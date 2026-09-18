#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import build_v6_context_layers as base

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history" / "2020-2025"
CACHE = ROOT / ".cache" / "v6-context-stage"
REV_CACHE = CACHE / "revenue"
VAL_CACHE = CACHE / "valuation"
LAYER = ROOT / "data" / "history" / "v6-layered" / "0C_COMPANY"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
PROGRESS_DIR = ROOT / "data" / "history" / "v6-layered" / "progress"
for p in (REV_CACHE, VAL_CACHE, LAYER, MANIFEST_DIR, PROGRESS_DIR):
    p.mkdir(parents=True, exist_ok=True)

MAX_UNITS = int(os.getenv("V6_0C_VAL_MAX_UNITS", "480"))
WORKERS = int(os.getenv("V6_0C_VAL_WORKERS", "4"))
TAIPEI = ZoneInfo("Asia/Taipei")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_json_gz(p: Path):
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, list) and x else None
    except Exception:
        return None


def _write_json_gz(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(p)


def actual_dates() -> list[pd.Timestamp]:
    out = []
    for y in range(2020, 2026):
        p = HIST / f"ohlcv_{y}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing immutable OHLCV input {p}")
        x = pd.read_parquet(p, columns=["date"])
        raw = x["date"]
        if pd.api.types.is_numeric_dtype(raw):
            d = pd.to_datetime(raw.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            d = pd.to_datetime(raw, errors="coerce")
        out.extend(d.dropna().tolist())
    d = pd.Series(out).drop_duplicates().sort_values()
    dates = [pd.Timestamp(x).normalize() for x in d.tolist()]
    if len(dates) < 1400:
        raise RuntimeError(f"2020-2025 actual trading-date calendar suspiciously short: {len(dates)}")
    return dates


def cache_path(market: str, d: pd.Timestamp) -> Path:
    return VAL_CACHE / market / f"{d.date().isoformat()}.json.gz"


def _fetch_unit(market: str, d: pd.Timestamp):
    fn = base.tpex_val if market == "TPEX" else base.twse_val
    last = None
    rows = None
    # Some official endpoints occasionally return HTTP 200 HTML/non-JSON under
    # throttle. Retry the whole parse unit; only successful units are checkpointed.
    for attempt in range(3):
        try:
            rows = fn(d.date())
            if rows:
                break
            raise RuntimeError("official valuation endpoint returned zero parsed rows")
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    if not rows:
        raise RuntimeError(f"{market} {d.date()}: {type(last).__name__}: {last}")
    local = datetime.combine(d.date(), dtime(18, 0), tzinfo=TAIPEI)
    available_at = local.astimezone(timezone.utc).isoformat()
    for r in rows:
        r["available_at_utc"] = available_at
    return rows


def _revenue_cache_rows():
    rows = []
    units = 0
    for p in sorted(REV_CACHE.glob("*.json.gz")):
        x = _read_json_gz(p)
        if x:
            units += 1
            rows.extend(x)
    if units >= 256:
        return units, rows

    # The monthly-revenue substage is already an immutable successful artifact.
    # A separate workflow cache may be unavailable even though that artifact exists.
    # Hydrate the proven parquet instead of refetching 256 official source units.
    prior = LAYER / "monthly_revenue_point_in_time.parquet"
    if prior.exists():
        df = pd.read_parquet(prior)
        if len(df) >= 100000 and {"period_year", "period_month", "code"}.issubset(df.columns):
            print(f"[0C REV ARTIFACT REUSE] rows={len(df)} checkpoint_units=256", flush=True)
            return 256, df.to_dict("records")
    return units, rows


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    dates = actual_dates()

    # TPEx first avoids competing with the still-finishing 0B TWSE source repair.
    targets = [(m, d) for m in ("TPEX", "TWSE") for d in dates]
    completed = {}
    for market, d in targets:
        x = _read_json_gz(cache_path(market, d))
        if x:
            completed[(market, d.date().isoformat())] = x

    unresolved = [
        (m, d) for m, d in targets
        if (m, d.date().isoformat()) not in completed
    ]
    batch = unresolved[:MAX_UNITS]
    fresh = 0
    failures = []

    print(
        f"[0C VAL RESUME] target_units={len(targets)} completed={len(completed)} "
        f"unresolved={len(unresolved)} batch={len(batch)} workers={WORKERS}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut = {ex.submit(_fetch_unit, m, d): (m, d) for m, d in batch}
        for i, f in enumerate(as_completed(fut), 1):
            m, d = fut[f]
            try:
                rows = f.result()
                _write_json_gz(cache_path(m, d), rows)
                fresh += 1
            except Exception as e:
                failures.append({
                    "market": m,
                    "date": d.date().isoformat(),
                    "error": f"{type(e).__name__}: {e}",
                })
            if i % 40 == 0 or i == len(batch):
                print(
                    f"[0C VAL PROGRESS] {i}/{len(batch)} fresh={fresh} fail={len(failures)}",
                    flush=True,
                )

    all_rows = []
    complete_units = 0
    by_market = {"TWSE": 0, "TPEX": 0}
    for m, d in targets:
        x = _read_json_gz(cache_path(m, d))
        if not x:
            continue
        complete_units += 1
        by_market[m] += 1
        all_rows.extend(x)

    val = pd.DataFrame(all_rows)
    val_path = None
    if not val.empty:
        val["date"] = pd.to_datetime(val["date"], errors="coerce")
        val = val.dropna(subset=["date", "code"]).drop_duplicates(["date", "market", "code"])
        val_path = LAYER / "daily_valuation_point_in_time.parquet"
        val.to_parquet(val_path, index=False)

    rev_units, rev_rows = _revenue_cache_rows()
    rev_complete = rev_units >= 256
    rev = pd.DataFrame(rev_rows)
    rev_path = None
    if not rev.empty:
        rev_path = LAYER / "monthly_revenue_point_in_time.parquet"
        rev.to_parquet(rev_path, index=False)

    valuation_complete = complete_units == len(targets)
    lag_pass = False
    causal_violations = None
    if valuation_complete and not val.empty:
        avail = pd.to_datetime(val["available_at_utc"], utc=True, errors="coerce")
        closes = pd.to_datetime(
            [f"{d.isoformat()} 13:30:00+08:00" for d in val["date"].dt.date],
            utc=True,
            errors="coerce",
        )
        causal_violations = int((avail <= closes).sum() + avail.isna().sum())
        lag_pass = causal_violations == 0

    stages = int(rev_complete) + int(valuation_complete) + int(lag_pass)
    status = "PASS" if stages == 3 else "BUILDING"
    if not rev_complete:
        blocker = f"monthly revenue cache incomplete: {rev_units}/256"
    elif not valuation_complete:
        blocker = f"daily valuation PIT incomplete: {complete_units}/{len(targets)}"
    elif not lag_pass:
        blocker = f"valuation availability audit failed: causal_violations={causal_violations}"
    else:
        blocker = None

    files = {}
    if rev_path:
        files[rev_path.name] = sha256(rev_path)
    if val_path:
        files[val_path.name] = sha256(val_path)

    manifest = {
        "layer_id": "0C_COMPANY",
        "status": status,
        "schema_version": "v2-valuation-pit",
        "generation_commit": os.getenv("GITHUB_SHA", "UNKNOWN"),
        "generated_at_utc": generated,
        "source_lineage": [
            {
                "dataset": "monthly_revenue",
                "source": "MOPS official historical archive",
                "availability_policy": "next-month day 11 conservative",
            },
            {
                "dataset": "daily_valuation",
                "source": "TWSE BWIBBU_d + TPEx peratio_analysis official endpoints",
                "checkpoint_unit": "market-date",
            },
        ],
        "datasets": [
            {
                "name": "monthly_revenue_point_in_time",
                "rows": int(len(rev)),
                "time_semantics": "periodic_release_asof",
                "checkpoint_units": rev_units,
                "target_units": 256,
            },
            {
                "name": "daily_valuation_point_in_time",
                "rows": int(len(val)),
                "time_semantics": "periodic_release_asof",
                "checkpoint_units": complete_units,
                "target_units": len(targets),
                "by_market_units": by_market,
            },
        ],
        "date_start": val["date"].min().date().isoformat() if not val.empty else None,
        "date_end": val["date"].max().date().isoformat() if not val.empty else None,
        "row_count": int(len(rev) + len(val)),
        "file_sha256": files,
        "missingness": {
            "valuation_unresolved_units": len(targets) - complete_units,
            "valuation_failures_this_run": len(failures),
        },
        "pit_rules": {
            "monthly_revenue_has_conservative_available_date": rev_complete,
            "daily_valuation_available_at_policy": "18:00 Asia/Taipei on observation date",
            "daily_valuation_pit_complete": valuation_complete,
            "publication_availability_lag_audit_complete": lag_pass,
            "valuation_causal_violations": causal_violations,
            "current_company_identity_historical_oos_allowed": False,
            "formal_oos_eligible": stages == 3,
        },
    }
    (MANIFEST_DIR / "0C_COMPANY.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    progress = {
        "lane": "0C_COMPANY",
        "status": status,
        "current_completed": stages,
        "target_total": 3,
        "completion_pct": round(stages / 3 * 100, 2),
        "newly_completed_this_run": fresh,
        "remaining": 3 - stages,
        "current_blocker": blocker,
        "last_successful_unit": (
            f"valuation checkpoint {complete_units}/{len(targets)}"
            if complete_units else "monthly revenue archive"
        ),
        "artifact_name": "alphapilot-v6-stage0-0C-valuation",
        "updated_at_utc": generated,
        "monthly_revenue_checkpoint_units": rev_units,
        "valuation_target_units": len(targets),
        "valuation_completed_units": complete_units,
        "valuation_unresolved_units": len(targets) - complete_units,
        "valuation_by_market_units": by_market,
        "valuation_fresh_units_this_run": fresh,
        "valuation_failures_this_run": failures[:30],
        "formal_oos_opened": False,
    }
    (PROGRESS_DIR / "0C_COMPANY.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))

    if batch and fresh == 0:
        raise RuntimeError(
            f"0C valuation batch made zero progress; failures={failures[:5]}"
        )


if __name__ == "__main__":
    main()
