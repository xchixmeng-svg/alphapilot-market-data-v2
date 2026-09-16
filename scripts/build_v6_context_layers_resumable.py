#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

import build_v6_context_layers as base

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "v6-context-stage"
IND_CACHE = CACHE / "industry"
MACRO_CACHE = CACHE / "macro"
REV_CACHE = CACHE / "revenue"
VAL_CACHE = CACHE / "valuation"
SNAP_CACHE = CACHE / "company_snapshot"
for p in (IND_CACHE, MACRO_CACHE, REV_CACHE, VAL_CACHE, SNAP_CACHE):
    p.mkdir(parents=True, exist_ok=True)

# Keep each run bounded. Successful units are checkpointed and restored by Actions cache.
IND_WORKERS = int(os.getenv("V6_IND_WORKERS", "2"))
MAX_IND_REQUESTS = int(os.getenv("V6_MAX_IND_REQUESTS", "700"))
MACRO_MIN_SERIES = 6
_macro_success_count = 0


def code4_fixed(x):
    s = str(x or "").strip().replace("=", "").replace('"', "")
    if re.fullmatch(r"[1-9]\d{3}\.0", s):
        s = s[:-2]
    return s if re.fullmatch(r"[1-9]\d{3}", s) else None


# Apply the parser correction to every downstream base function without mutating source on the runner.
base.code4 = code4_fixed


def _write_json_gz(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def _read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _safe_cached_rows(path: Path):
    if not path.exists():
        return None
    try:
        rows = _read_json_gz(path)
        if isinstance(rows, list) and rows:
            return rows
    except Exception:
        pass
    path.unlink(missing_ok=True)
    return None


# ---------- A) Macro: cache each successful FRED series independently ----------
def build_macro_resumable():
    global _macro_success_count
    frames = []
    failures = []
    for short, sid in base.FRED_SERIES.items():
        p = MACRO_CACHE / f"{short}_{sid}.csv.gz"
        df = None
        if p.exists():
            try:
                df = pd.read_csv(p)
                if list(df.columns) != ["date", short] or df.empty:
                    raise ValueError("invalid cached macro shape")
                print("[MACRO CACHE]", short, sid, len(df), flush=True)
            except Exception:
                p.unlink(missing_ok=True)
                df = None
        if df is None:
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                r = base.get(url, timeout=20, tries=2)
                df = pd.read_csv(io.BytesIO(r.content)).iloc[:, :2].copy()
                df.columns = ["date", short]
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df[short] = pd.to_numeric(df[short], errors="coerce")
                df = df[(df["date"].dt.year >= base.START_YEAR) & (df["date"].dt.date <= base.END_DATE)]
                if df.empty:
                    raise RuntimeError("empty FRED response")
                df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(p, index=False, compression="gzip")
                print("[MACRO FETCH+SAVE]", short, sid, len(df), flush=True)
            except Exception as e:
                failures.append({"series": sid, "name": short, "error": str(e)})
                print("[MACRO RETRY-NEXT-RUN]", short, sid, e, flush=True)
                continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[short] = pd.to_numeric(df[short], errors="coerce")
        frames.append(df[["date", short]])

    _macro_success_count = len(frames)
    if frames:
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="date", how="outer")
        out = out.sort_values("date")
        cols = [c for c in out.columns if c != "date"]
        out[cols] = out[cols].ffill()
    else:
        out = pd.DataFrame({"date": pd.date_range(date(base.START_YEAR, 1, 1), base.END_DATE, freq="D")})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    dest = base.OUT / "macro_daily.csv.gz"
    base.write_gz(dest, out.to_dict("records"))
    print("[MACRO CHECKPOINT] success_series", _macro_success_count, "/", len(base.FRED_SERIES), flush=True)
    return dest, failures, len(out)


# ---------- B) TWSE industry tape: one checkpoint per successful market date ----------
_original_parse_industry = base.parse_twse_industry


def _industry_cache_path(d: date) -> Path:
    return IND_CACHE / f"{d.isoformat()}.json.gz"


def _fetch_industry_checkpointed(d: date):
    p = _industry_cache_path(d)
    cached = _safe_cached_rows(p)
    if cached is not None:
        return cached, "cache"
    got = _original_parse_industry(d)
    if got:
        _write_json_gz(p, got)
        return got, "network"
    return [], "empty"


def build_industry_indices_resumable():
    dates = list(base.weekdays(date(base.START_YEAR, 1, 1), base.END_DATE))
    rows = []
    cached_dates = set()
    for d in dates:
        cached = _safe_cached_rows(_industry_cache_path(d))
        if cached is not None:
            rows.extend(cached)
            cached_dates.add(d.isoformat())

    missing = [d for d in dates if d.isoformat() not in cached_dates]
    batch = missing[:MAX_IND_REQUESTS]
    failures = []
    empty = 0
    fresh_dates = 0
    print(
        "[IND RESUME] cached_dates", len(cached_dates),
        "missing_weekdays", len(missing),
        "this_run_requests", len(batch),
        "workers", IND_WORKERS,
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=IND_WORKERS) as ex:
        fut = {ex.submit(_fetch_industry_checkpointed, d): d for d in batch}
        for i, f in enumerate(as_completed(fut), 1):
            d = fut[f]
            try:
                got, source = f.result()
                if got:
                    rows.extend(got)
                    if source == "network":
                        fresh_dates += 1
                else:
                    empty += 1
            except Exception as e:
                failures.append({"date": d.isoformat(), "error": str(e)})
            if i % 50 == 0 or i == len(fut):
                coverage = len({r["date"] for r in rows})
                print(
                    "[IND PROGRESS]", i, "/", len(fut),
                    "cached_start", len(cached_dates),
                    "fresh_saved", fresh_dates,
                    "coverage_dates", coverage,
                    "rows", len(rows),
                    "empty", empty,
                    "hard_fail", len(failures),
                    flush=True,
                )

    rows = sorted({(r["date"], r["index_name"]): r for r in rows}.values(), key=lambda r: (r["date"], r["index_name"]))
    coverage_dates = len({r["date"] for r in rows})
    print(
        "[IND CHECKPOINT] coverage_dates", coverage_dates,
        "fresh_saved", fresh_dates,
        "remaining_weekdays", max(0, len(missing) - len(batch)),
        flush=True,
    )
    if coverage_dates < 1800:
        raise RuntimeError(
            f"industry index coverage too low: dates={coverage_dates}; "
            f"checkpointed_successes_are_preserved; retry only missing dates next run"
        )
    dest = base.OUT / "twse_industry_index_daily.csv.gz"
    base.write_gz(dest, rows)
    return dest, failures, len(rows), coverage_dates, len({r["index_name"] for r in rows})


# ---------- C) Live snapshot: whole-output checkpoint keyed by END_DATE ----------
_original_company_snapshot = base.build_company_industry_snapshot


def build_company_snapshot_resumable():
    cache_file = SNAP_CACHE / f"company_industry_{base.END_DATE.isoformat()}.csv.gz"
    dest = base.OUT / "company_industry_live_snapshot.csv.gz"
    if cache_file.exists():
        shutil.copy2(cache_file, dest)
        df = pd.read_csv(dest)
        print("[COMPANY SNAPSHOT CACHE]", len(df), flush=True)
        return dest, len(df)
    p, n = _original_company_snapshot()
    shutil.copy2(p, cache_file)
    print("[COMPANY SNAPSHOT FETCH+SAVE]", n, flush=True)
    return p, n


# ---------- D) Revenue: checkpoint each month/market unit ----------
_original_fetch_revenue_month = base.fetch_revenue_month


def fetch_revenue_month_resumable(y, m, market):
    p = REV_CACHE / f"{y:04d}-{m:02d}-{market}.json.gz"
    cached = _safe_cached_rows(p)
    if cached is not None:
        return cached
    rows = _original_fetch_revenue_month(y, m, market)
    if rows:
        _write_json_gz(p, rows)
    return rows


# ---------- E) Valuation: checkpoint each market/date unit ----------
_original_twse_val = base.twse_val
_original_tpex_val = base.tpex_val


def _valuation_resumable(market, d, fn):
    p = VAL_CACHE / market / f"{d.isoformat()}.json.gz"
    cached = _safe_cached_rows(p)
    if cached is not None:
        return cached
    rows = fn(d)
    if rows:
        _write_json_gz(p, rows)
    return rows


def twse_val_resumable(d):
    return _valuation_resumable("TWSE", d, _original_twse_val)


def tpex_val_resumable(d):
    return _valuation_resumable("TPEX", d, _original_tpex_val)


# Patch the base builders/fetchers. base.main then keeps the preregistered aggregation/audit logic.
base.build_macro = build_macro_resumable
base.build_industry_indices = build_industry_indices_resumable
base.build_company_industry_snapshot = build_company_snapshot_resumable
base.fetch_revenue_month = fetch_revenue_month_resumable
base.twse_val = twse_val_resumable
base.tpex_val = tpex_val_resumable


def main():
    base.main()
    # Do not allow OOS to start with an under-covered macro layer. Industry has its own 1800-date gate.
    if _macro_success_count < MACRO_MIN_SERIES:
        raise RuntimeError(
            f"macro coverage too low after resumable staging: {_macro_success_count}/{len(base.FRED_SERIES)}; "
            "successful series are preserved; retry only missing series next run"
        )


if __name__ == "__main__":
    main()
