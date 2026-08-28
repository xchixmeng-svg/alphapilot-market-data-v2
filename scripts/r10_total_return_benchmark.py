#!/usr/bin/env python3
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# Formal archived 0050 ETF cash distributions (ex-date, per share).
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

# Formal benchmark-source gaps documented by the corrected R6.2 archive.
FORMAL_MISSING = {
    20230525,
    20250206,
    20250611, 20250612, 20250613, 20250616, 20250617,
}

EXPECTED_ANNUAL = {
    2021: 0.19911078492215806,
    2022: -0.2134159350721403,
    2023: 0.2739778912491486,
    2024: 0.4867147167748145,
    2025: 0.36853420406222615,
}

BASE_DATE = 20210104
BASE_NAV = 2_000_000.0
SPLIT_DATE = 20250618
SPLIT_FACTOR = 4.0


def _hash(q: pd.DataFrame) -> str:
    z = q[["date", "mkt"]].sort_values("date").copy()
    payload = z.to_csv(index=False, float_format="%.12f", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_0050_total_return(all_dates: list[int], raw: pd.DataFrame):
    """Rebuild the formal PIT-safe 0050 ETF Total Return benchmark.

    2021-2025 is reconstructed from raw 0050 closes, archived cash
    distributions and the 2025 4-for-1 split. The result exactly reproduces
    the corrected R6.2 annual benchmark fingerprints. The archived source gaps
    are then forward-filled only. For pre-2021 warm-up, follow the original
    R6.2 source exactly: scale raw 0050 close to the first benchmark NAV.
    """
    rr = raw.copy()
    rr["code"] = rr.code.astype(str).str.strip().str.zfill(4)
    q = rr[rr.code.eq("0050")][["date", "close"]].copy()
    q["date"] = pd.to_numeric(q.date, errors="coerce")
    q["close"] = pd.to_numeric(q.close, errors="coerce")
    q = q.dropna().astype({"date": int}).drop_duplicates("date", keep="last").sort_values("date")
    if q.empty or BASE_DATE not in set(q.date):
        raise RuntimeError("raw 0050 base date missing")

    q["px"] = q["close"].astype(float)
    q.loc[q.date >= SPLIT_DATE, "px"] *= SPLIT_FACTOR
    q["cash_dividend"] = q.date.map(CASH_DISTRIBUTIONS).fillna(0.0).astype(float)
    q.loc[q.date >= SPLIT_DATE, "cash_dividend"] *= SPLIT_FACTOR

    live = q[q.date >= BASE_DATE].copy()
    live["gross"] = (live["px"] + live["cash_dividend"]) / live["px"].shift(1)
    live.loc[live.date.eq(BASE_DATE), "gross"] = 1.0
    live["mkt"] = BASE_NAV * live["gross"].cumprod()

    # Reproduce archived source gaps before the formal past-only ffill.
    live.loc[live.date.isin(FORMAL_MISSING), "mkt"] = np.nan

    # Original R6.2 warm-up rule: before the first benchmark row, use raw 0050
    # close scaled to the benchmark's first NAV. No future values are used.
    base_px = float(q.loc[q.date.eq(BASE_DATE), "close"].iloc[0])
    scale = BASE_NAV / base_px
    warm = q[q.date < BASE_DATE][["date", "close"]].copy()
    warm["mkt"] = warm["close"].astype(float) * scale

    src = pd.concat([warm[["date", "mkt"]], live[["date", "mkt"]]], ignore_index=True)
    src = src.drop_duplicates("date", keep="last").sort_values("date")
    cal = pd.DataFrame({"date": sorted(set(int(x) for x in all_dates))})
    out = cal.merge(src, on="date", how="left").sort_values("date")
    out["mkt"] = out["mkt"].ffill()  # formal rule: past-only; no bfill
    if out["mkt"].isna().any():
        bad = out[out.mkt.isna()].date.head(10).tolist()
        raise RuntimeError(f"0050 Total Return unavailable without future fill: {bad}")

    # Independent benchmark fingerprint gate; never used as a strategy target.
    eval_out = out[out.date >= BASE_DATE].copy()
    annual = {}
    prev = float(eval_out.iloc[0].mkt)
    years = eval_out.date.astype(str).str[:4].astype(int)
    for y in range(2021, 2026):
        g = eval_out[years.eq(y)]
        if g.empty:
            raise RuntimeError(f"0050 TR year missing: {y}")
        end = float(g.iloc[-1].mkt)
        annual[y] = end / prev - 1.0
        prev = end
    delta = {y: annual[y] - EXPECTED_ANNUAL[y] for y in EXPECTED_ANNUAL}
    max_delta = max(abs(x) for x in delta.values())
    if max_delta > 5e-6:
        raise RuntimeError(f"0050 TR fingerprint mismatch max_delta={max_delta}")

    meta = {
        "source": "0050 ETF Total Return reconstructed from raw 0050 close + archived cash distributions + 2025 4-for-1 split",
        "series": "0050 ETF Total Return",
        "base_date": BASE_DATE,
        "base_nav": BASE_NAV,
        "formal_missing_dates": sorted(FORMAL_MISSING),
        "fill_policy": "forward-fill past known value only; no backward fill",
        "aligned_dates": int(len(out)),
        "annual_fingerprint": annual,
        "annual_expected": EXPECTED_ANNUAL,
        "annual_max_abs_delta": max_delta,
        "end_nav": float(eval_out.iloc[-1].mkt),
        "sha256": _hash(eval_out),
    }
    return out[["date", "mkt"]], meta
