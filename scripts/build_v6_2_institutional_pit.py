#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data/history/2020-2025/institutional_2020_2025.parquet"
OHLCV_DIR = ROOT / "data/history/2020-2025"
OUT = ROOT / "v6_2_institutional_pit"
OUT.mkdir(exist_ok=True)

FIELDS = ["foreign_net", "trust_net", "dealer_net"]


def num(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "").replace("+", "")
    if s in {"", "--", "---", "null", "None"}:
        return None
    try:
        return int(round(float(s)))
    except Exception:
        return None


def parse_hist() -> pd.DataFrame:
    z = pd.read_parquet(HIST).copy()
    z["date"] = pd.to_numeric(z["date"], errors="coerce").astype("Int64")
    z["market"] = z["market"].astype(str).str.upper()
    z["code"] = z["code"].astype(str).str.strip()
    for c in FIELDS:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype("Int64")
    return z[z["date"].notna() & (z["date"] <= 20241231)].copy()


def load_ohlcv() -> pd.DataFrame:
    xs = []
    for y in range(2020, 2025):
        p = OHLCV_DIR / f"ohlcv_{y}.parquet"
        x = pd.read_parquet(p)
        cols = {str(c).lower(): c for c in x.columns}
        date_col = cols.get("date") or cols.get("trade_date")
        code_col = cols.get("code") or cols.get("stock_id")
        vol_col = cols.get("volume") or cols.get("tradevolume") or cols.get("trading_shares")
        if not all([date_col, code_col, vol_col]):
            raise RuntimeError(f"OHLCV schema unsupported for {p}: {list(x.columns)}")
        q = x[[date_col, code_col, vol_col]].copy()
        q.columns = ["date", "code", "volume"]
        q["date"] = pd.to_numeric(q["date"], errors="coerce").astype("Int64")
        q["code"] = q["code"].astype(str).str.strip()
        q["volume"] = pd.to_numeric(q["volume"], errors="coerce")
        if q.duplicated(["date","code"], keep=False).any():
            raise RuntimeError(f"OHLCV duplicate date/code keys in {p}")
        xs.append(q)
    return pd.concat(xs, ignore_index=True)


def trading_calendar(ohlcv: pd.DataFrame) -> list[int]:
    return sorted(int(x) for x in ohlcv.loc[ohlcv["date"].notna() & (ohlcv["date"] <= 20241231), "date"].unique())


def next_session_map(cal: list[int]) -> dict[int, int]:
    return {d: cal[i + 1] for i, d in enumerate(cal[:-1])}


def attach_pit_and_scale(z: pd.DataFrame, ohlcv: pd.DataFrame, market: str) -> tuple[pd.DataFrame, dict]:
    z = z[z["market"] == market].copy()
    if z.empty:
        raise RuntimeError(f"no rows for market={market}")
    cal = trading_calendar(ohlcv)
    ns = next_session_map(cal)
    z["available_session"] = z["date"].map(ns).astype("Int64")
    z = z[z["available_session"].notna()].copy()

    # Preserved historical OHLCV archive has no market column; date+code is unique.
    vol = ohlcv[["date", "code", "volume"]].copy()
    z = z.merge(vol, on=["date", "code"], how="left", validate="one_to_one")

    for c in FIELDS:
        z[c + "_ratio"] = np.where(z["volume"] > 0, z[c].astype("Float64") / z["volume"], np.nan)

    dup = int(z.duplicated(["date", "market", "code"], keep=False).sum())
    future = int((z["available_session"] <= z["date"]).sum())
    missing_avail = int(z["available_session"].isna().sum())
    ratio_nonnull = {c + "_ratio": int(z[c + "_ratio"].notna().sum()) for c in FIELDS}
    raw_nonnull = {c: int(z[c].notna().sum()) for c in FIELDS}
    audit = {
        "market": market,
        "rows": int(len(z)),
        "source_min_date": int(z["date"].min()),
        "source_max_date": int(z["date"].max()),
        "available_session_min": int(z["available_session"].min()),
        "available_session_max": int(z["available_session"].max()),
        "duplicate_source_key_violations": dup,
        "available_session_not_after_source_violations": future,
        "missing_available_session": missing_avail,
        "raw_nonnull": raw_nonnull,
        "ratio_nonnull": ratio_nonnull,
        "admission_rule": "decision_date >= available_session; source trade_date itself is never visible on same-day decision",
        "rolling_rule": "No rolling sums are materialized here. Any rolling institutional feature must reset at V6.1 price_segment_id boundary.",
    }
    return z, audit


def fetch_json(url, params=None, retries=5):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AlphaPilot-V6.2-TPEx-Rebuild"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8-sig"))
        except Exception as e:
            last = e
            time.sleep(min(8, 1.5 ** k))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {type(last).__name__}: {last}")


def fetch_tpex(d: int) -> pd.DataFrame:
    roc = str(int(str(d)[:4]) - 1911) + "/" + str(d)[4:6] + "/" + str(d)[6:8]
    url = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
    obj = fetch_json(url, {"l": "zh-tw", "o": "json", "d": roc, "se": "EW", "t": "D", "s": "0,asc"})
    aa = obj.get("aaData") if isinstance(obj, dict) else None
    if not isinstance(aa, list) or not aa:
        tables = obj.get("tables") if isinstance(obj, dict) else None
        aa = None
        if isinstance(tables, list):
            for t in tables:
                if not isinstance(t, dict):
                    continue
                data = t.get("data") or t.get("aaData")
                if isinstance(data, list) and data and isinstance(data[0], list) and len(data[0]) >= 23:
                    aa = data
                    break
    if not isinstance(aa, list):
        raise RuntimeError(f"TPEx table unavailable for {d}")
    rows = []
    for raw in aa:
        if not isinstance(raw, list) or len(raw) < 23:
            continue
        # TPEx columns after code/name are 7 buy/sell/net triplets:
        # [2:5] foreign+China EXCLUDING foreign dealers, [5:8] foreign dealers,
        # [8:11] foreign total, [11:14] trust, [14:17] dealer proprietary,
        # [17:20] dealer hedge, [20:23] dealer total.  Official TPEx notes say
        # foreign-dealer flow is already included in dealer flow and must be excluded
        # from the three-institution total to avoid double counting.  Therefore the
        # canonical foreign_net is raw[4], NOT foreign-total raw[10].
        foreign_ex_dealer_net = num(raw[4])
        foreign_dealer_net = num(raw[7])
        foreign_total_net = num(raw[10])
        trust_net = num(raw[13])
        dealer_prop_net = num(raw[16])
        dealer_hedge_net = num(raw[19])
        dealer_total_net = num(raw[22])
        # Fail closed if the endpoint's positional semantics drift.
        if None not in (foreign_ex_dealer_net, foreign_dealer_net, foreign_total_net):
            if foreign_ex_dealer_net + foreign_dealer_net != foreign_total_net:
                raise RuntimeError(f"TPEx foreign mapping identity failed for {d} code={raw[0]}")
        if None not in (dealer_prop_net, dealer_hedge_net, dealer_total_net):
            if dealer_prop_net + dealer_hedge_net != dealer_total_net:
                raise RuntimeError(f"TPEx dealer mapping identity failed for {d} code={raw[0]}")
        rows.append({
            "date": d,
            "market": "TPEX",
            "code": str(raw[0]).strip(),
            "foreign_net": foreign_ex_dealer_net,
            "trust_net": trust_net,
            "dealer_net": dealer_total_net,
        })
    if not rows:
        raise RuntimeError(f"TPEx parsed zero rows for {d}")
    return pd.DataFrame(rows)


def build_twse():
    hist = parse_hist()
    ohlcv = load_ohlcv()
    z, audit = attach_pit_and_scale(hist, ohlcv, "TWSE")
    out = OUT / "twse_institutional_pit_2020_2024.parquet"
    z.to_parquet(out, index=False)
    audit["output"] = str(out.relative_to(ROOT))
    audit["status"] = "PASS" if (
        audit["duplicate_source_key_violations"] == 0
        and audit["available_session_not_after_source_violations"] == 0
        and all(v > 0 for v in audit["raw_nonnull"].values())
    ) else "FAIL"
    (OUT / "TWSE_PIT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[TWSE PIT]", json.dumps(audit, ensure_ascii=False), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("TWSE PIT layer failed audit")


def rebuild_tpex_year(year: int):
    hist = parse_hist()
    ohlcv = load_ohlcv()
    cal = trading_calendar(ohlcv)
    dates = [d for d in cal if int(str(d)[:4]) == year]
    checkpoint = OUT / f"tpex_rebuild_{year}.parquet"
    errors_path = OUT / f"tpex_rebuild_{year}_errors.json"
    done = pd.DataFrame()
    if checkpoint.exists():
        done = pd.read_parquet(checkpoint)
    done_dates = set(int(x) for x in done["date"].unique()) if len(done) else set()
    errors = []
    chunks = [done] if len(done) else []
    for i, d in enumerate(dates, 1):
        if d in done_dates:
            continue
        try:
            q = fetch_tpex(d)
            chunks.append(q)
        except Exception as e:
            errors.append({"date": d, "error": f"{type(e).__name__}: {e}"})
        if i % 15 == 0 or i == len(dates):
            cur = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            if len(cur):
                cur = cur.drop_duplicates(["date", "market", "code"], keep="last").sort_values(["date", "code"])
                cur.to_parquet(checkpoint, index=False)
                chunks = [cur]
            errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        time.sleep(0.12)

    z = pd.read_parquet(checkpoint)
    expected = set(dates)
    actual = set(int(x) for x in z["date"].unique())
    missing_dates = sorted(expected - actual)
    dup = int(z.duplicated(["date", "market", "code"], keep=False).sum())
    nonnull = {c: int(z[c].notna().sum()) for c in FIELDS}
    summary = {
        "year": year,
        "expected_trading_dates": len(expected),
        "rebuilt_dates": len(actual),
        "missing_dates": missing_dates,
        "fetch_errors": errors,
        "rows": int(len(z)),
        "duplicate_source_key_violations": dup,
        "nonnull": nonnull,
        "status": "PASS_RAW_REBUILD" if not missing_dates and dup == 0 and all(v > 0 for v in nonnull.values()) else "INCOMPLETE",
    }
    (OUT / f"TPEX_REBUILD_{year}_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[TPEX REBUILD]", json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["status"] != "PASS_RAW_REBUILD":
        raise RuntimeError(f"TPEx rebuild incomplete for {year}")


def assemble_tpex():
    ohlcv = load_ohlcv()
    xs = []
    for y in range(2020, 2025):
        p = OUT / f"tpex_rebuild_{y}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing TPEx checkpoint {p}")
        xs.append(pd.read_parquet(p))
    raw = pd.concat(xs, ignore_index=True)
    z, audit = attach_pit_and_scale(raw, ohlcv, "TPEX")
    out = OUT / "tpex_institutional_pit_2020_2024.parquet"
    z.to_parquet(out, index=False)
    audit["output"] = str(out.relative_to(ROOT))
    audit["status"] = "PASS" if (
        audit["duplicate_source_key_violations"] == 0
        and audit["available_session_not_after_source_violations"] == 0
        and all(v > 0 for v in audit["raw_nonnull"].values())
    ) else "FAIL"
    (OUT / "TPEX_PIT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[TPEX PIT]", json.dumps(audit, ensure_ascii=False), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("TPEx PIT layer failed audit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["twse", "tpex-year", "tpex-assemble"])
    ap.add_argument("--year", type=int)
    a = ap.parse_args()
    if a.mode == "twse":
        build_twse()
    elif a.mode == "tpex-year":
        if a.year not in range(2020, 2025):
            raise SystemExit("--year must be 2020..2024")
        rebuild_tpex_year(a.year)
    else:
        assemble_tpex()


if __name__ == "__main__":
    main()
