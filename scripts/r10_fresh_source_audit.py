#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh-source audit for AlphaPilot R10 historical validation.

This script verifies that cached/backtest inputs agree with fresh official
TWSE/TPEx responses on deterministic sample dates.  It never reads or compares
performance targets, Golden orders, exits, trades, or NAV references.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import r10_fast_validation as fast

bt = fast.bt
core = fast.core
ROOT = Path(__file__).resolve().parent.parent
OUT = bt.OUT_ROOT / "latest" / "true_validation2021_2025"


def _norm(q: pd.DataFrame) -> pd.DataFrame:
    if q.empty:
        return q.copy()
    z = q.copy()
    z["code"] = z["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for c in ("open", "high", "low", "close", "volume"):
        if c in z:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    return z


def _inst_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["market", "code", "foreign_net", "trust_net", "dealer_net"])
    q = pd.DataFrame(rows).copy()
    q["code"] = q["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    for c in ("foreign_net", "trust_net", "dealer_net"):
        q[c] = pd.to_numeric(q[c], errors="coerce").fillna(0.0)
    return q[["market", "code", "foreign_net", "trust_net", "dealer_net"]].drop_duplicates(["market", "code"], keep="last")


def main() -> None:
    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = _norm(bt.load_scenario_ohlcv(cfg))
    lo, hi = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    dates = np.array(sorted(int(x) for x in raw.date.unique() if lo <= int(x) <= hi), dtype=int)
    if len(dates) < 1000:
        raise RuntimeError(f"unexpectedly short evaluation calendar: {len(dates)}")

    # Ten deterministic, widely separated real trading days from the exact
    # backtest calendar. No hand-picked outcome dates.
    idx = np.linspace(0, len(dates) - 1, 10, dtype=int)
    sample_dates = [int(dates[i]) for i in idx]

    day_audits = []
    inst_audits = []
    hard_failures = []

    for di in sample_dates:
        d = bt.dt_from_int(di)
        hist = _norm(raw[raw.date.eq(di)][["code", "open", "high", "low", "close", "volume"]])
        fresh = _norm(core.official_day(d)[["code", "open", "high", "low", "close", "volume"]])
        if fresh.empty:
            hard_failures.append(f"{di} official OHLCV empty")
            continue
        m = hist.merge(fresh, on="code", suffixes=("_hist", "_fresh"), how="inner")
        if m.empty:
            hard_failures.append(f"{di} no OHLCV code intersection")
            continue
        px_bad = pd.Series(False, index=m.index)
        for c in ("open", "high", "low", "close"):
            px_bad |= (m[f"{c}_hist"] - m[f"{c}_fresh"]).abs() > 1e-8
        vol_bad = (m["volume_hist"] - m["volume_fresh"]).abs() > 0.5
        mismatch = px_bad | vol_bad
        # Restrict the coverage statistic to simple 4-digit securities used by
        # the R10 stock universe; index/special instruments are irrelevant here.
        hist_codes = set(hist.code[hist.code.str.fullmatch(r"\d{4}")])
        fresh_codes = set(fresh.code[fresh.code.str.fullmatch(r"\d{4}")])
        inter = hist_codes & fresh_codes
        coverage = len(inter) / max(1, len(hist_codes))
        day_audits.append({
            "date": di,
            "historical_rows": int(len(hist)),
            "official_rows": int(len(fresh)),
            "intersection_rows": int(len(m)),
            "four_digit_coverage": float(coverage),
            "ohlcv_mismatch_rows": int(mismatch.sum()),
            "sample_mismatches": m.loc[mismatch].head(5).to_dict("records"),
        })
        if coverage < 0.90:
            hard_failures.append(f"{di} OHLCV code coverage {coverage:.2%} < 90%")
        if int(mismatch.sum()) != 0:
            hard_failures.append(f"{di} OHLCV mismatches={int(mismatch.sum())}")

        # Fresh institutional response vs the exact daily cache consumed by the
        # historical engine. These are official TWSE/TPEx calls, not the cache.
        fresh_rows: list[dict] = []
        fresh_rows.extend(core.twse_inst(d))
        time.sleep(0.15)
        fresh_rows.extend(core.tpex_inst(d))
        fi = _inst_frame(fresh_rows)
        cache_path = fast.INST_DAY_CACHE / f"{di}.csv"
        if not cache_path.exists():
            hard_failures.append(f"{di} institutional cache missing")
            continue
        ci = _inst_frame(pd.read_csv(cache_path, dtype={"code": str}).to_dict("records"))
        im = ci.merge(fi, on=["market", "code"], suffixes=("_cache", "_fresh"), how="inner")
        if im.empty:
            hard_failures.append(f"{di} institutional no code intersection")
            continue
        ibad = pd.Series(False, index=im.index)
        for c in ("foreign_net", "trust_net", "dealer_net"):
            ibad |= (im[f"{c}_cache"] - im[f"{c}_fresh"]).abs() > 0.5
        cache_keys = set(zip(ci.market.astype(str), ci.code.astype(str)))
        fresh_keys = set(zip(fi.market.astype(str), fi.code.astype(str)))
        icoverage = len(cache_keys & fresh_keys) / max(1, len(cache_keys))
        inst_audits.append({
            "date": di,
            "cache_rows": int(len(ci)),
            "official_rows": int(len(fi)),
            "intersection_rows": int(len(im)),
            "key_coverage": float(icoverage),
            "institutional_mismatch_rows": int(ibad.sum()),
            "sample_mismatches": im.loc[ibad].head(5).to_dict("records"),
        })
        if icoverage < 0.90:
            hard_failures.append(f"{di} institutional key coverage {icoverage:.2%} < 90%")
        if int(ibad.sum()) != 0:
            hard_failures.append(f"{di} institutional mismatches={int(ibad.sum())}")
        time.sleep(0.15)

    report = {
        "status": "PASS" if not hard_failures else "FAIL",
        "purpose": "fresh official input verification only",
        "performance_reference_used": False,
        "sample_method": "10 evenly spaced dates from actual 2021-2025 backtest trading calendar",
        "sample_dates": sample_dates,
        "ohlcv": day_audits,
        "institutional": inst_audits,
        "failures": hard_failures,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fresh_source_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
