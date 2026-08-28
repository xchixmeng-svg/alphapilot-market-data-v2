#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated audit for reconstructing the formal 0050 ETF Total Return series.

This script does NOT run or steer R10. It only checks whether a mechanical
reconstruction from raw 0050 closes + official cash distributions + the 2025
4-for-1 split reproduces the archived R6.2 benchmark fingerprints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import r10_fast_validation as fast

bt = fast.bt

# Ex-dividend date -> cash distribution per share as published for 0050.
# Post-split cash is converted to old-share-equivalent units below together
# with the price series so the return formula has one continuous unit basis.
CASH_DISTRIBUTIONS = {
    20210122: 3.05,
    20210721: 0.35,
    20220121: 3.20,
    20220718: 1.80,
    20230130: 2.60,
    20230718: 1.90,
    20240117: 3.00,
    20240716: 1.00,
    20250117: 2.70,
    20250721: 0.36,
}

# Archived corrected R6.2 workbook fingerprints. These are used only to audit
# the benchmark reconstruction, never as a trading-engine input.
EXPECTED_ANNUAL = {
    2021: 0.19911078492215806,
    2022: -0.2134159350721403,
    2023: 0.2739778912491486,
    2024: 0.4867147167748145,
    2025: 0.36853420406222615,
}

# The archived benchmark itself had these source gaps. Its formal rule is
# past-only forward fill; no future/backward fill.
FORMAL_MISSING = {
    20230525,
    20250206,
    20250611, 20250612, 20250613, 20250616, 20250617,
}

SPLIT_DATE = 20250618
SPLIT_FACTOR = 4.0
BASE_DATE = 20210104
BASE_NAV = 2_000_000.0
OUT = bt.CACHE_ROOT / "0050_total_return_reconstruction_audit"


def _sha(df: pd.DataFrame) -> str:
    s = df[["date", "nav"]].to_csv(index=False, float_format="%.12f", lineterminator="\n")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = bt.load_scenario_ohlcv(cfg).copy()
    raw["code"] = raw.code.astype(str).str.strip().str.zfill(4)
    q = raw[raw.code.eq("0050")][["date", "close"]].copy()
    q["date"] = pd.to_numeric(q.date, errors="coerce")
    q["close"] = pd.to_numeric(q.close, errors="coerce")
    q = q.dropna().astype({"date": int}).drop_duplicates("date", keep="last").sort_values("date")
    q = q[q.date.between(BASE_DATE, 20251231)].copy()
    if q.empty or BASE_DATE not in set(q.date):
        raise RuntimeError("raw 0050 base date missing")

    # Continuous old-share-equivalent price basis across the 4-for-1 split.
    q["px"] = q.close.astype(float)
    q.loc[q.date >= SPLIT_DATE, "px"] *= SPLIT_FACTOR
    q["div"] = q.date.map(CASH_DISTRIBUTIONS).fillna(0.0).astype(float)
    q.loc[q.date >= SPLIT_DATE, "div"] *= SPLIT_FACTOR

    # Total return with cash distribution reinvested on its ex-date close:
    # TR_t / TR_t-1 = (P_t + D_t) / P_t-1.
    q["gross"] = (q.px + q.div) / q.px.shift(1)
    q.loc[q.date.eq(BASE_DATE), "gross"] = 1.0
    q["nav_raw"] = BASE_NAV * q.gross.cumprod()

    # Align to the full market calendar from raw data and reproduce the formal
    # benchmark gaps with past-only ffill.
    cal = pd.DataFrame({"date": sorted(int(x) for x in raw.date.unique() if BASE_DATE <= int(x) <= 20251231)})
    src = q[["date", "nav_raw"]].copy()
    src.loc[src.date.isin(FORMAL_MISSING), "nav_raw"] = np.nan
    out = cal.merge(src, on="date", how="left").sort_values("date")
    out["benchmark_source_missing"] = out.nav_raw.isna()
    out["nav"] = out.nav_raw.ffill()
    if out.nav.isna().any():
        raise RuntimeError(f"reconstruction has leading missing values: {out[out.nav.isna()].date.head().tolist()}")

    annual = {}
    prev = float(out.iloc[0].nav)
    for y in range(2021, 2026):
        g = out[out.date.astype(str).str[:4].astype(int).eq(y)]
        if g.empty:
            raise RuntimeError(f"year {y} missing")
        end = float(g.iloc[-1].nav)
        annual[y] = end / prev - 1.0
        prev = end

    diffs = {y: annual[y] - EXPECTED_ANNUAL[y] for y in EXPECTED_ANNUAL}
    audit = {
        "base_date": BASE_DATE,
        "base_nav": BASE_NAV,
        "split": {"date": SPLIT_DATE, "factor": SPLIT_FACTOR},
        "cash_distributions": CASH_DISTRIBUTIONS,
        "formal_missing": sorted(FORMAL_MISSING),
        "annual_reconstructed": annual,
        "annual_expected": EXPECTED_ANNUAL,
        "annual_delta": diffs,
        "max_abs_annual_delta": max(abs(x) for x in diffs.values()),
        "end_nav": float(out.iloc[-1].nav),
        "sha256": _sha(out),
        "rows": int(len(out)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out[["date", "nav", "benchmark_source_missing"]].to_csv(
        OUT / "benchmark_0050_total_return_reconstructed.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    # Tight enough to prove the convention rather than merely approximate it.
    if audit["max_abs_annual_delta"] > 5e-6:
        raise RuntimeError(
            "0050 Total Return reconstruction fingerprint mismatch; "
            f"max annual delta={audit['max_abs_annual_delta']:.9f}. "
            "Do not use this reconstruction in R10."
        )
    print("0050 TOTAL RETURN RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
