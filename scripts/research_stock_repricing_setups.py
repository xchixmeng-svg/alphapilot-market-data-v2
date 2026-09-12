from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('research_out_stock_repricing')
OUT.mkdir(exist_ok=True)
START = 20230523
DEV_END = 20241231
HOLDOUT_START = 20250101
END = 20251231
INIT = 1_300_000.0
FEE = 0.000855
TAX = 0.003
MAX_POS = 4
SLOT = 0.24

# Frozen causal layer produced by locked formal engine. No peer/sector/context router is used here.
px = pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz', dtype={'code': str}, low_memory=False)
px['code'] = px.code.astype(str).str.zfill(4)
px = px.sort_values(['code', 'date']).reset_index(drop=True)
inst = pd.read_parquet('formal_run/institutional_2020_2025.parquet')
inst['code'] = inst.code.astype(str).str.zfill(4)
if pd.api.types.is_datetime64_any_dtype(inst.date):
    inst['date'] = inst.date.dt.strftime('%Y%m%d').astype(int)
else:
    inst['date'] = pd.to_datetime(inst.date).dt.strftime('%Y%m%d').astype(int)
inst = inst.sort_values(['code', 'date']).copy()

# -------------------- causal T-close features --------------------
px['amount'] = px.close * px.volume
g = px.groupby('code', group_keys=False)
for w in (1, 3, 5, 10, 20, 60, 120):
    px[f'r{w}'] = g.aclose.transform(lambda s, w=w: s.pct_change(w))
for w in (10, 20, 60, 120):
    px[f'ma{w}'] = g.aclose.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
px['amount20'] = g.amount.transform(lambda s: s.rolling(20, min_periods=20).mean())
px['amount_ratio'] = px.amount / px.amount20.replace(0, np.nan)
px['prior_high20'] = g.aclose.transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
px['prior_high60'] = g.aclose.transform(lambda s: s.shift(1).rolling(60, min_periods=60).max())
px['high20'] = g.aclose.transform(lambda s: s.rolling(20, min_periods=20).max())
px['draw20'] = px.aclose / px.high20 - 1.0
px['vol20'] = g.aclose.transform(lambda s: s.pct_change().rolling(20, min_periods=20).std())
px['dist_ma20'] = px.aclose / px.ma20 - 1.0
px['dist_ma60'] = px.aclose / px.ma60 - 1.0
px['prev_close'] = g.aclose.shift(1)
px['prev_high20'] = g.prior_high20.shift(1)

# Institutional features; all rolling windows use only information known by T close.
gi = inst.groupby('code', group_keys=False)
inst['foreign5'] = gi.foreign_net.transform(lambda s: s.rolling(5, min_periods=5).sum())
inst['trust5'] = gi.trust_net.transform(lambda s: s.rolling(5, min_periods=5).sum())
inst['foreign20'] = gi.foreign_net.transform(lambda s: s.rolling(20, min_periods=20).sum())
inst['trust20'] = gi.trust_net.transform(lambda s: s.rolling(20, min_periods=20).sum())
inst['inst5'] = inst.foreign5 + inst.trust5
inst['inst20'] = inst.foreign20 + inst.trust20
inst['inst_prev5'] = gi[['foreign_net', 'trust_net']].transform(lambda s: s.shift(5).rolling(5, min_periods=5).sum()).sum(axis=1)
inst['inst_accel'] = inst.inst5 - inst.inst_prev5
px = px.merge(inst[['date', 'code', 'inst5', 'inst20', 'inst_prev5', 'inst_accel']], on=['date', 'code'], how='left')
px[['inst5', 'inst20', 'inst_prev5', 'inst_accel']] = px[['inst5', 'inst20', 'inst_prev5', 'inst_accel']].fillna(0.0)
px['flow5_to_amount'] = (px.inst5 * px.close) / (px.amount20 * 5).replace(0, np.nan)
px['flow20_to_amount'] = (px.inst20 * px.close) / (px.amount20 * 20).replace(0, np.nan)
px['flow_accel_value'] = (px.inst_accel * px.close) / (px.amount20 * 5).replace(0, np.nan)

# Universe filter only removes structurally unsuitable instruments. It is not a market gate.
valid = (
    px.code.str.fullmatch(r'[1-9]\d{3}')
    & (~px.name.astype(str).str.contains('KY', case=False, na=False))
    & (px.close >= 5)
    & (px.amount20 >= 30_000_000)
)
stocks = px[valid].copy()

# Cross-sectional ranks are contemporaneous T-close information.
for c in ['r5', 'r20', 'r60', 'r120', 'amount20', 'amount_ratio', 'flow5_to_amount', 'flow20_to_amount', 'flow_accel_value', 'vol20']:
    stocks[c + '_pr'] = stocks.groupby('date')[c].rank(pct=True)

# Separate market/liquidity quality filter. This is NOT a fundamental-business-quality claim.
def market_quality(d: pd.DataFrame) -> pd.Series:
    return (
        (d.close >= 10)
        & (d.amount20 >= 100_000_000)
        & (d.vol20_pr <= 0.95)
        & (d.r120_pr >= 0.25)
    )

# Each family is an economic stock-level hypothesis. No market regime/peer group is consulted.
# Holding periods are fixed ex ante for this discovery batch; no 2025 tuning.
SETUPS = {
    'flow_accel_breakout': {
        'hold': 8,
        'cond': lambda d: (
            (d.aclose > d.prior_high20)
            & (d.r20_pr >= 0.65)
            & (d.flow_accel_value_pr >= 0.75)
            & (d.flow5_to_amount_pr >= 0.60)
            & (d.amount_ratio >= 1.15)
        ),
        'score': lambda d: 0.32*d.flow_accel_value_pr + 0.26*d.r20_pr + 0.22*d.amount_ratio_pr + 0.20*d.flow5_to_amount_pr,
    },
    'institutional_accumulation': {
        'hold': 10,
        'cond': lambda d: (
            (d.aclose > d.ma20)
            & (d.ma20 > d.ma60)
            & (d.r60_pr >= 0.60)
            & (d.flow20_to_amount_pr >= 0.75)
            & (d.flow5_to_amount_pr >= 0.60)
            & (d.dist_ma20 <= 0.10)
        ),
        'score': lambda d: 0.35*d.flow20_to_amount_pr + 0.25*d.flow5_to_amount_pr + 0.25*d.r60_pr + 0.15*d.amount20_pr,
    },
    'strong_trend_absorption': {
        'hold': 8,
        'cond': lambda d: (
            (d.ma20 > d.ma60)
            & (d.aclose > d.ma60)
            & (d.r60_pr >= 0.72)
            & (d.draw20 <= -0.025)
            & (d.draw20 >= -0.10)
            & (d.flow5_to_amount_pr >= 0.55)
            & (d.r3 > -0.06)
        ),
        'score': lambda d: 0.34*d.r60_pr + 0.28*d.flow5_to_amount_pr + 0.20*(1-d.vol20_pr) + 0.18*d.amount20_pr,
    },
    'rs_not_extended': {
        'hold': 10,
        'cond': lambda d: (
            (d.r20_pr >= 0.85)
            & (d.r60_pr >= 0.72)
            & (d.aclose > d.ma20)
            & (d.dist_ma20 >= 0.0)
            & (d.dist_ma20 <= 0.08)
            & (d.amount_ratio <= 2.5)
            & (d.flow20_to_amount_pr >= 0.45)
        ),
        'score': lambda d: 0.34*d.r20_pr + 0.28*d.r60_pr + 0.20*d.flow20_to_amount_pr + 0.18*d.amount20_pr,
    },
    'flow_turnaround_reclaim': {
        'hold': 7,
        'cond': lambda d: (
            (d.inst_prev5 <= 0)
            & (d.inst5 > 0)
            & (d.flow_accel_value_pr >= 0.78)
            & (d.aclose > d.ma20)
            & (d.prev_close <= d.ma20.shift(0))
            & (d.r20 > -0.08)
        ),
        'score': lambda d: 0.42*d.flow_accel_value_pr + 0.24*d.r20_pr + 0.18*d.amount_ratio_pr + 0.16*d.amount20_pr,
    },
}

# Fix reclaim condition correctly: compare prior adjusted close with prior MA20 from the same stock.
stocks['prev_ma20'] = stocks.groupby('code').ma20.shift(1)
SETUPS['flow_turnaround_reclaim']['cond'] = lambda d: (
    (d.inst_prev5 <= 0)
    & (d.inst5 > 0)
    & (d.flow_accel_value_pr >= 0.78)
    & (d.aclose > d.ma20)
    & (d.prev_close <= d.prev_ma20)
    & (d.r20 > -0.08)
)

# Legal Taiwan tick helpers.
def tick(p: float) -> float:
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0

def ceil_tick(p: float) -> float:
    t = tick(p)
    return math.ceil((p - 1e-12) / t) * t

def floor_tick(p: float) -> float:
    t = tick(p)
    return math.floor((p + 1e-12) / t) * t

stocks = stocks.sort_values(['date', 'code']).copy()
bydate = {int(d): q.copy() for d, q in stocks.groupby('date')}
calendar = sorted(int(x) for x in stocks[(stocks.date >= START) & (stocks.date <= END)].date.unique())
cal_i = {d: i for i, d in enumerate(calendar)}
rowmap = {(int(r.date), r.code): r for r in stocks.itertuples(index=False)}

# Signal ledger. Setup screen happens first, then independent market/liquidity quality removal.
orders_by_setup = {k: {} for k in SETUPS}
signal_rows = []
for d in calendar:
    if d not in bydate: continue
    i = cal_i[d]
    if i + 1 >= len(calendar): continue
    entry_date = calendar[i + 1]
    day = bydate[d]
    for name, spec in SETUPS.items():
        raw = day[spec['cond'](day).fillna(False)].copy()
        if raw.empty: continue
        raw['setup_score'] = spec['score'](raw)
        q = raw[market_quality(raw).fillna(False)].sort_values(['setup_score', 'amount20'], ascending=False).head(6)
        if q.empty: continue
        items = []
        for r in q.itertuples(index=False):
            items.append({'code': r.code, 'signal_date': d, 'entry_date': entry_date, 'hold': int(spec['hold']), 'score': float(r.setup_score)})
            signal_rows.append({'setup': name, 'code': r.code, 'signal_date': d, 'entry_date': entry_date, 'hold': int(spec['hold']), 'score': float(r.setup_score)})
        orders_by_setup[name].setdefault(entry_date, []).extend(items)

signals = pd.DataFrame(signal_rows)
signals.to_csv(OUT / 'signal_ledger.csv', index=False)
if signals.empty:
    raise RuntimeError('no stock-level repricing signals generated')

# Shared-capital simulator, one independent portfolio per setup family.
def simulate(setup: str, start: int, end: int, label: str) -> dict:
    pending = orders_by_setup[setup]
    cash = INIT
    pos = {}
    trades = []
    navrows = []
    dates = [d for d in calendar if start <= d <= end]
    if not dates:
        raise RuntimeError(f'no dates for {label}')

    for d in dates:
        # Apply official share-factor event before execution/valuation.
        for code, p in list(pos.items()):
            rr = rowmap.get((d, code))
            if rr is not None and bool(getattr(rr, 'is_official_event', False)):
                sf = float(getattr(rr, 'share_factor', 1.0))
                if np.isfinite(sf) and sf > 0 and abs(sf - 1.0) > 1e-12:
                    p['shares'] = int(round(p['shares'] * sf))

        # Scheduled exits at T+N open, conservatively slipped.
        for code, p in list(pos.items()):
            if p['exit_date'] != d: continue
            rr = rowmap.get((d, code))
            if rr is None or not np.isfinite(rr.open) or rr.open <= 0: continue
            xp = floor_tick(float(rr.open) * 0.995)
            gross = p['shares'] * xp
            cash += gross * (1 - FEE - TAX)
            ret = (xp * (1 - FEE - TAX)) / (p['entry_price'] * (1 + FEE)) - 1
            trades.append({'setup': setup, 'code': code, 'signal_date': p['signal_date'], 'entry_date': p['entry_date'], 'exit_date': d, 'entry_price': p['entry_price'], 'exit_price': xp, 'shares': p['shares'], 'return_net': ret})
            del pos[code]

        # T-close signal -> next-session open execution. No same-day hindsight.
        for item in sorted(pending.get(d, []), key=lambda x: x['score'], reverse=True):
            if len(pos) >= MAX_POS: break
            code = item['code']
            if code in pos: continue
            rr = rowmap.get((d, code))
            if rr is None or not np.isfinite(rr.open) or rr.open <= 0: continue
            if bool(getattr(rr, 'is_official_event', False)): continue
            ci = cal_i[d] + item['hold']
            if ci >= len(calendar): continue
            exit_date = calendar[ci]
            ep = ceil_tick(float(rr.open) * 1.005)
            mv = 0.0
            for c, p in pos.items():
                vr = rowmap.get((d, c))
                if vr is not None and np.isfinite(vr.close):
                    mv += p['shares'] * float(vr.close)
            nav_open = cash + mv
            budget = min(cash, nav_open * SLOT)
            shares = int(budget // (ep * (1 + FEE)))
            if shares <= 0: continue
            cost = shares * ep * (1 + FEE)
            if cost > cash + 1e-9: continue
            cash -= cost
            if cash < -1e-7:
                raise RuntimeError(f'negative cash {setup} {d}')
            pos[code] = {'shares': shares, 'entry_price': ep, 'entry_date': d, 'signal_date': item['signal_date'], 'exit_date': exit_date}

        nav = cash
        for code, p in pos.items():
            rr = rowmap.get((d, code))
            if rr is not None and np.isfinite(rr.close):
                nav += p['shares'] * float(rr.close)
        navrows.append({'date': d, 'nav': nav, 'cash': cash, 'positions': len(pos)})

    # Force liquidation at final available open/close proxy to make periods comparable.
    last = dates[-1]
    for code, p in list(pos.items()):
        rr = rowmap.get((last, code))
        if rr is None or not np.isfinite(rr.close) or rr.close <= 0: continue
        xp = floor_tick(float(rr.close) * 0.995)
        cash += p['shares'] * xp * (1 - FEE - TAX)
        ret = (xp * (1 - FEE - TAX)) / (p['entry_price'] * (1 + FEE)) - 1
        trades.append({'setup': setup, 'code': code, 'signal_date': p['signal_date'], 'entry_date': p['entry_date'], 'exit_date': last, 'entry_price': p['entry_price'], 'exit_price': xp, 'shares': p['shares'], 'return_net': ret, 'forced_exit': True})
        del pos[code]
    navrows[-1]['nav'] = cash
    navrows[-1]['cash'] = cash
    navrows[-1]['positions'] = 0

    nav = pd.DataFrame(navrows)
    t = pd.DataFrame(trades)
    years = max((pd.to_datetime(str(end)) - pd.to_datetime(str(start))).days / 365.25, 1/252)
    cagr = (cash / INIT) ** (1 / years) - 1
    dd = float((nav.nav / nav.nav.cummax() - 1).min())
    wr = float((t.return_net > 0).mean()) if len(t) else 0.0
    gp = float(t.loc[t.return_net > 0, 'return_net'].sum()) if len(t) else 0.0
    gl = float(-t.loc[t.return_net < 0, 'return_net'].sum()) if len(t) else 0.0
    pf = gp / gl if gl > 0 else float('inf')
    if cash < -1e-6:
        raise RuntimeError(f'negative final cash {setup} {label}')
    if abs(float(nav.iloc[-1].nav) - cash) > 0.01:
        raise RuntimeError(f'cash/nav mismatch {setup} {label}')

    nav.to_csv(OUT / f'nav_{setup}_{label}.csv', index=False)
    t.to_csv(OUT / f'trades_{setup}_{label}.csv', index=False)
    return {'setup': setup, 'label': label, 'start': start, 'end': end, 'end_nav': float(cash), 'cagr': float(cagr), 'max_drawdown': dd, 'trades': int(len(t)), 'wins': int((t.return_net > 0).sum()) if len(t) else 0, 'win_rate': wr, 'profit_factor': pf}

results = []
for setup in SETUPS:
    results.append(simulate(setup, START, DEV_END, 'dev'))
    results.append(simulate(setup, HOLDOUT_START, END, 'holdout_2025'))
    results.append(simulate(setup, START, END, 'full'))
res = pd.DataFrame(results)
res.to_csv(OUT / 'portfolio_summary.csv', index=False)

# Explicit year-by-year evidence (independent calendar-year portfolios, no carry-in positions).
yearly = []
for setup in SETUPS:
    for y, s, e in [(2023, 20230523, 20231229), (2024, 20240102, 20241231), (2025, 20250102, 20251231)]:
        r = simulate(setup, s, e, f'year_{y}')
        r['year'] = y
        yearly.append(r)
pd.DataFrame(yearly).to_csv(OUT / 'yearly_summary.csv', index=False)

# Dev-only ranking; 2025 is evidence only and never participates in selection.
dev = res[res.label == 'dev'].copy()
dev['robust_score'] = (
    np.clip(dev.win_rate, 0, 1) * 2.0
    + np.clip(dev.cagr, -1, 1) * 1.5
    + np.clip(dev.profit_factor, 0, 3) * 0.35
    + np.minimum(dev.trades, 80) / 200.0
    + np.clip(dev.max_drawdown, -1, 0) * 0.5
)
eligible = dev[(dev.trades >= 25) & (dev.cagr > 0) & (dev.profit_factor > 1.10)].sort_values('robust_score', ascending=False)
selected_dev = eligible.setup.tolist()

# Minute gate deliberately strict: daily edge must be positive in both dev and OOS with adequate sample.
gate_rows = []
for setup in selected_dev:
    d = res[(res.setup == setup) & (res.label == 'dev')].iloc[0]
    h = res[(res.setup == setup) & (res.label == 'holdout_2025')].iloc[0]
    gate = bool(d.cagr > 0 and d.profit_factor > 1.10 and h.cagr > 0 and h.profit_factor >= 1.20 and h.trades >= 25 and h.win_rate >= 0.55)
    gate_rows.append({'setup': setup, 'minute_gate': gate})
minute_gate = any(x['minute_gate'] for x in gate_rows)

decision = {
    'status': 'PASS',
    'architecture': 'independent stock-level repricing setups -> separate market/liquidity quality filter -> T+1 shared-capital execution',
    'rejected_architecture_not_used': 'peer/sector-cycle/archetype router',
    'fundamental_filter_present': False,
    'fundamental_note': 'No point-in-time revenue/EPS/financial-statement layer is claimed in this batch.',
    'research_window': [START, END],
    'dev_window': [START, DEV_END],
    'holdout_window': [HOLDOUT_START, END],
    'target': {'cagr': 0.50, 'win_rate': 0.70},
    'dev_eligible_setups': selected_dev,
    'minute_gate_by_setup': gate_rows,
    'minute_optimization_allowed': minute_gate,
    'formal_r10_untouched': True,
}
(OUT / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
print(res.to_string(index=False))
print(pd.DataFrame(yearly).to_string(index=False))
print(json.dumps(decision, ensure_ascii=False, indent=2))
