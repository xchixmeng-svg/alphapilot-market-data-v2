#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

import build_r10_scan as core

CACHE = Path('.stress_cache') / 'tai50_total_return_2020_2025.csv'


def _hash(q: pd.DataFrame) -> str:
    z = q[['date','mkt']].sort_values('date').copy()
    payload = z.to_csv(index=False, float_format='%.12g').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_tai50_total_return(eval_dates: list[int], start_year: int = 2020, end_year: int = 2025):
    """Return a PIT-safe Taiwan 50 total-return benchmark aligned to stock dates.

    Source is the official TWSE TAI50I report already used by the live scanner.
    Missing index rows are forward-filled only. No backward fill/future row is
    permitted. The historical performance ledger is never read here.
    """
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        src = pd.read_csv(CACHE)
        src['date'] = pd.to_numeric(src['date'], errors='coerce').astype('Int64')
        src['mkt'] = pd.to_numeric(src['mkt'], errors='coerce')
        src = src.dropna().astype({'date': int}).drop_duplicates('date', keep='last').sort_values('date')
    else:
        parts = []
        for y in range(start_year, end_year + 1):
            # core.tai50() requests every month in the target year and extracts
            # the field containing both "50" and "報酬" from TWSE TAI50I.
            q = core.tai50(date(y, 12, 31))
            if q is None or q.empty:
                raise RuntimeError(f'TWSE Taiwan50 total-return history empty for {y}')
            parts.append(q[['date','mkt']].copy())
        src = pd.concat(parts, ignore_index=True)
        src['date'] = pd.to_numeric(src['date'], errors='coerce')
        src['mkt'] = pd.to_numeric(src['mkt'], errors='coerce')
        src = src.dropna().astype({'date': int}).drop_duplicates('date', keep='last').sort_values('date')
        src.to_csv(CACHE, index=False, encoding='utf-8-sig')

    cal = pd.DataFrame({'date': sorted(set(int(x) for x in eval_dates))})
    out = cal.merge(src[['date','mkt']], on='date', how='left').sort_values('date')
    out['mkt'] = out['mkt'].ffill()  # past-only; never bfill
    if out['mkt'].isna().any():
        bad = out[out.mkt.isna()].date.head(10).tolist()
        raise RuntimeError(f'Taiwan50 total-return unavailable without future fill: {bad}')

    used = src[src.date.between(int(cal.date.min()), int(cal.date.max()))].copy()
    meta = {
        'source': core.TWSE_TAI50,
        'series': 'Taiwan 50 total-return index (field containing 50 + 報酬)',
        'source_rows': int(len(used)),
        'aligned_dates': int(len(out)),
        'fill_policy': 'forward-fill past known value only; no backward fill',
        'sha256': _hash(used),
    }
    return out[['date','mkt']], meta
