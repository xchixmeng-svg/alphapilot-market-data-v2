#!/usr/bin/env python3
"""Performance-only wrapper for AlphaPilot AI V12.

This file does not change routing, labels, thresholds, R10 rules, or success gates.
It replaces only the V11 shadow data-access implementation: repeated full-DataFrame
boolean scans become a one-time limited-column MultiIndex plus cached per-code views.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'scripts' / 'backtest_ai_causal_router_v12.py'
spec = importlib.util.spec_from_file_location('ai_v12', src)
v12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v12)

BUY_FEE, SELL_FEE, SELL_TAX = 0.000855, 0.000855, 0.003
BUY_ADVERSE, SELL_ADVERSE, MIN_HOLD = 0.005, 0.005, 3
_INDEXED = {}
_CODE = {}


def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def floor_tick(p):
    t = tick(p)
    return round(np.floor((p + 1e-10) / t) * t, 4)


def ceil_tick(p):
    t = tick(p)
    return round(np.ceil((p - 1e-10) / t) * t, 4)


def buy_fill(open_, low_, limit_):
    if open_ <= limit_:
        return min(ceil_tick(open_ * (1 + BUY_ADVERSE)), limit_)
    return limit_ if low_ <= limit_ else None


def _indexed(px):
    token = id(px)
    if token not in _INDEXED:
        work = px[['code', 'date', 'open', 'low', 'close', 'aclose']].copy()
        work['r7_exposure'] = px['r7_exposure'].to_numpy() if 'r7_exposure' in px.columns else 1.0
        work['r7_hard'] = px['r7_hard'].to_numpy() if 'r7_hard' in px.columns else False
        work['amount_ratio'] = px['amount_ratio'].to_numpy() if 'amount_ratio' in px.columns else np.nan
        work['code'] = work['code'].astype(str)
        work['date'] = work['date'].astype(int)
        _INDEXED[token] = work.set_index(['code', 'date']).sort_index()
        print(f'FAST_SHADOW_INDEX rows={len(work)} codes={work.code.nunique()}', flush=True)
    return token, _INDEXED[token]


def _code_frame(px, code):
    token, idx = _indexed(px)
    key = (token, str(code))
    if key not in _CODE:
        try:
            _CODE[key] = idx.xs(str(code), level='code', drop_level=True)
        except KeyError:
            _CODE[key] = None
    return _CODE[key]


def _row(g, d):
    if g is None or int(d) not in g.index:
        return None, False
    r = g.loc[int(d)]
    if isinstance(r, pd.DataFrame):
        return None, True
    return r, False


def shadow_return_fast(px, date_to_i, dates, signal_date, code, strategy):
    """Semantically equivalent to V11 shadow_return, with indexed lookups."""
    i = date_to_i.get(int(signal_date))
    if i is None or i + 1 >= len(dates):
        return np.nan
    g = _code_frame(px, str(code))
    sr, sdup = _row(g, int(signal_date))
    er, edup = _row(g, dates[i + 1])
    if sr is None or er is None or sdup or edup:
        return np.nan

    lim = floor_tick(float(sr.close) * (0.98 if strategy == 'R7' else 0.995))
    fill = buy_fill(float(er.open), float(er.low), lim)
    if fill is None or fill <= 0:
        return np.nan
    adj_factor = float(er.aclose) / float(er.close) if float(er.close) > 0 else np.nan
    if not np.isfinite(adj_factor):
        return np.nan
    entry_idx = fill * adj_factor
    peak, state, hold = float(er.aclose), None, 0
    exit_idx = np.nan

    for j in range(i + 1, len(dates) - 1):
        r, dup = _row(g, dates[j])
        if dup or r is None:
            continue
        hold += 1
        idx = float(r.aclose)
        peak = max(peak, idx)
        if hold < MIN_HOLD:
            continue
        ret = idx / entry_idx - 1.0
        reason = False
        if strategy == 'R7':
            reason = (ret <= -0.12) or (float(r.get('r7_exposure', 1.0)) <= 0.0) or (not bool(r.get('r7_hard', False)))
        else:
            ar = r.get('amount_ratio', np.nan)
            if state is None and ret >= 0.40 and np.isfinite(ar) and ar >= 2.0:
                state = 'runner'
            if state == 'runner' and ret >= 0.80:
                state = 'mega' if np.isfinite(ar) and ar >= 1.2 else 'target'
            if ret <= -0.10:
                reason = True
            elif state == 'target' and (ret >= 2.00 or idx <= peak * 0.80):
                reason = True
            elif state == 'mega' and idx <= peak * 0.84:
                reason = True
            elif state == 'runner' and idx <= peak * 0.86:
                reason = True
            elif state is None and ret >= 0.50 and idx <= peak * 0.88:
                reason = True
            elif hold >= (120 if state is not None else 60):
                reason = True
        if reason:
            xr, xdup = _row(g, dates[j + 1])
            if xr is None or xdup:
                return np.nan
            raw_exit = floor_tick(float(xr.open) * (1 - SELL_ADVERSE))
            af = float(xr.aclose) / float(xr.close) if float(xr.close) > 0 else np.nan
            if not np.isfinite(af):
                return np.nan
            exit_idx = raw_exit * af
            break

    if not np.isfinite(exit_idx):
        return np.nan
    gross_ratio = exit_idx / entry_idx
    return gross_ratio * (1 - SELL_FEE - SELL_TAX) / (1 + BUY_FEE) - 1.0


def self_test():
    dates = [20260102, 20260105, 20260106, 20260107, 20260108, 20260109]
    px = pd.DataFrame({
        'code': ['0001'] * 6,
        'date': dates,
        'open': [100, 98, 97, 96, 95, 94],
        'low': [99, 97, 96, 95, 94, 93],
        'close': [100, 98, 97, 96, 95, 94],
        'aclose': [100, 98, 97, 96, 95, 94],
        'r7_exposure': [1, 1, 1, 1, 1, 1],
        'r7_hard': [True, True, True, False, False, False],
        'amount_ratio': [1, 1, 1, 1, 1, 1],
    })
    didx = {d: i for i, d in enumerate(dates)}
    slow = v12.v11.shadow_return(px, didx, dates, 20260102, '0001', 'R7')
    fast = shadow_return_fast(px, didx, dates, 20260102, '0001', 'R7')
    assert (np.isnan(slow) and np.isnan(fast)) or np.isclose(slow, fast, rtol=0, atol=1e-12), (slow, fast)
    print(f'FAST_SHADOW_SELFTEST PASS slow={slow} fast={fast}', flush=True)


if __name__ == '__main__':
    self_test()
    v12.v11.shadow_return = shadow_return_fast
    print('FAST_SHADOW_PATCH ACTIVE', flush=True)
    v12.main()
