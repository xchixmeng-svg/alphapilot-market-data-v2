from __future__ import annotations

import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_CTX = Path('research_out_context_cycle')
BASE_EV = Path('research_out_continuous_context')
OUT = Path('research_out_sector_cycle_router')
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
MAX_EXPOSURE = 0.95
ADV_CAP = 0.02
K = 45
MIN_HISTORY_DAYS = 80
MIN_STAGE1 = 18
MIN_STAGE2 = 12
TOP_POLICIES = 2


def sector_proxy(code: str) -> str:
    code = str(code).zfill(4)
    if code.startswith('28'):
        return 'financial'
    if code.startswith(('23', '24')):
        return 'electronics'
    if code.startswith('26'):
        return 'transport'
    if code.startswith('20'):
        return 'materials'
    return 'other'


def pf_of(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    return pos / neg if neg > 0 else (99.0 if pos > 0 else 0.0)


def edge_score(r: pd.Series) -> float:
    if len(r) == 0:
        return -999.0
    wr = float((r > 0).mean())
    mean = float(r.mean())
    med = float(r.median())
    p10 = float(r.quantile(.10))
    pf = pf_of(r)
    # No target forcing. Reward repeatability and payoff, penalize left-tail instability.
    return 2.4 * wr + 7.5 * mean + 1.4 * med + .16 * min(pf, 3.0) + 1.8 * min(p10, 0.0)


ctx = pd.read_csv(BASE_CTX / 'daily_context_cycle.csv')
ev = pd.read_csv(BASE_EV / 'candidate_events.csv', dtype={'code': str})
ev['code'] = ev.code.astype(str).str.zfill(4)
ev['sector'] = ev.code.map(sector_proxy)

px = pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz', dtype={'code': str}, low_memory=False)
px['code'] = px.code.astype(str).str.zfill(4)
px = px.sort_values(['code', 'date']).reset_index(drop=True)
px['vol20'] = px.groupby('code').volume.transform(lambda s: s.rolling(20, min_periods=20).mean())

# Exclude candidate trades that cross any official corporate-action date. This research stage
# intentionally avoids needing a second independent corporate-action position-adjustment engine.
def as_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in {'1', 'true', 't', 'yes', 'y'}
    return bool(x) if pd.notna(x) else False

if 'is_official_event' in px.columns:
    action_rows = px[px.is_official_event.map(as_bool)][['code', 'date']]
else:
    action_rows = pd.DataFrame(columns=['code', 'date'])
action_dates = {c: sorted(q.date.astype(int).tolist()) for c, q in action_rows.groupby('code')}


def crosses_action(code: str, start: int, end: int) -> bool:
    a = action_dates.get(code, [])
    if not a:
        return False
    i = bisect.bisect_left(a, int(start))
    return i < len(a) and a[i] <= int(end)

mask_clean = [not crosses_action(r.code, int(r.entry_date), int(r.exit_date)) for r in ev.itertuples(index=False)]
ev = ev[np.array(mask_clean, dtype=bool)].copy()

# Signal-day execution/liquidity information known by T close.
sig_liq = px[['date', 'code', 'vol20', 'close']].rename(columns={'date': 'signal_date', 'close': 'signal_close'})
ev = ev.merge(sig_liq, on=['signal_date', 'code'], how='left')
ev = ev[np.isfinite(ev.vol20) & (ev.vol20 > 0)].copy()

# Continuous context vector. Both base *_z and causal cycle *_cz columns were standardized
# using only prior observations by the upstream scripts.
FEATURES = [c for c in ctx.columns if c.endswith('_z') or c.endswith('_cz')]
ctx['ready'] = ctx[FEATURES].notna().sum(axis=1) >= max(8, int(.72 * len(FEATURES)))
ctx_map = {
    int(r.date): np.array([getattr(r, c) for c in FEATURES], dtype=float)
    for r in ctx.itertuples(index=False)
    if bool(r.ready)
}
ctx_dates = sorted(ctx_map)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < max(8, int(.68 * len(a))):
        return np.inf
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def neighbor_dates(d: int) -> list[int]:
    if d not in ctx_map:
        return []
    prior = [x for x in ctx_dates if x < d]
    if len(prior) < MIN_HISTORY_DAYS:
        return []
    cur = ctx_map[d]
    ds = sorted((distance(cur, ctx_map[x]), x) for x in prior)
    return [x for di, x in ds if np.isfinite(di)][:K]


# Stage 1 chooses sector + playbook from loose-quality outcomes only.
# Stage 2 separately chooses how strict the company/stock quality filter should be.
def choose_policies(d: int) -> list[dict]:
    nei = neighbor_dates(d)
    if len(nei) < 20:
        return []
    known = ev[(ev.signal_date.isin(nei)) & (ev.exit_date < d)].copy()
    if known.empty:
        return []

    loose = known[known.quality == 'loose']
    stage1 = []
    for (sector, pb), q in loose.groupby(['sector', 'playbook']):
        if len(q) < MIN_STAGE1:
            continue
        r = q.return_net.astype(float)
        pf = pf_of(r)
        mean = float(r.mean())
        med = float(r.median())
        wr = float((r > 0).mean())
        if mean <= 0 or pf <= 1.05 or med <= -0.025:
            continue
        stage1.append({
            'sector': sector, 'playbook': pb, 'n1': int(len(q)),
            'mean1': mean, 'median1': med, 'wr1': wr, 'pf1': pf,
            'score1': edge_score(r)
        })
    if not stage1:
        return []
    stage1 = sorted(stage1, key=lambda x: (x['score1'], x['n1']), reverse=True)[:TOP_POLICIES]

    chosen = []
    for s in stage1:
        same = known[(known.sector == s['sector']) & (known.playbook == s['playbook'])]
        qrows = []
        for quality, q in same.groupby('quality'):
            if len(q) < MIN_STAGE2:
                continue
            r = q.return_net.astype(float)
            pf = pf_of(r)
            mean = float(r.mean())
            med = float(r.median())
            wr = float((r > 0).mean())
            if mean <= 0 or pf <= 1.05 or med <= -0.03:
                continue
            qrows.append({
                'quality': quality, 'n2': int(len(q)), 'mean2': mean,
                'median2': med, 'wr2': wr, 'pf2': pf, 'score2': edge_score(r)
            })
        if not qrows:
            continue
        qpick = max(qrows, key=lambda x: (x['score2'], x['n2']))
        chosen.append({**s, **qpick, 'neighbors': len(nei)})
    return chosen


calendar = sorted(int(x) for x in px[(px.date >= START) & (px.date <= END)].date.unique())
route_cache = {d: choose_policies(d) for d in calendar}

# Build orders only after market/cycle -> sector/playbook -> separate quality filtering.
orders = []
for d in calendar:
    policies = route_cache.get(d, [])
    if not policies:
        continue
    for p in policies:
        q = ev[(ev.signal_date == d) & (ev.sector == p['sector']) & (ev.playbook == p['playbook']) & (ev.quality == p['quality'])].copy()
        if q.empty:
            continue
        q = q.sort_values('score', ascending=False).head(3)
        for r in q.itertuples(index=False):
            orders.append({
                'signal_date': int(d), 'entry_date': int(r.entry_date), 'exit_date': int(r.exit_date),
                'code': r.code, 'sector': p['sector'], 'playbook': p['playbook'], 'quality': p['quality'],
                'candidate_score': float(r.score), 'entry_price': float(r.entry_price), 'exit_price': float(r.exit_price),
                'vol20': float(r.vol20), 'route_score': float(p['score1']), 'quality_score': float(p['score2']),
                'hist_sector_n': int(p['n1']), 'hist_sector_wr': float(p['wr1']), 'hist_sector_pf': float(p['pf1']),
                'hist_quality_n': int(p['n2']), 'hist_quality_wr': float(p['wr2']), 'hist_quality_pf': float(p['pf2'])
            })
orders = pd.DataFrame(orders)
if orders.empty:
    orders = pd.DataFrame(columns=['signal_date','entry_date','exit_date','code','sector','playbook','quality','candidate_score','entry_price','exit_price','vol20','route_score','quality_score'])
orders.to_csv(OUT / 'routed_orders.csv', index=False)

px_idx = {(int(r.date), r.code): r for r in px.itertuples(index=False)}


def mark_price(date: int, code: str, field: str, fallback: float) -> float:
    r = px_idx.get((int(date), code))
    if r is None:
        return fallback
    v = getattr(r, field, np.nan)
    return float(v) if np.isfinite(v) and v > 0 else fallback


def simulate(start: int, end: int, label: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    days = [d for d in calendar if start <= d <= end]
    if not days:
        raise RuntimeError(f'no calendar for {label}')
    entry_map = defaultdict(list)
    if not orders.empty:
        for r in orders[(orders.entry_date >= start) & (orders.entry_date <= end)].itertuples(index=False):
            entry_map[int(r.entry_date)].append(r)
    cash = INIT
    positions: dict[str, dict] = {}
    trades = []
    nav_rows = []
    min_cash = INIT
    max_positions = 0
    max_exposure = 0.0

    for d in days:
        # T+1 open exits first.
        for code in list(positions):
            p = positions[code]
            if int(p['exit_date']) != d:
                continue
            xp = float(p['exit_price'])
            proceeds = xp * p['shares'] * (1 - FEE - TAX)
            cash += proceeds
            ret = proceeds / p['cost'] - 1.0
            trades.append({**p, 'exit_fill_date': d, 'realized_return': ret, 'proceeds': proceeds})
            del positions[code]

        # Pre-buy open NAV / exposure for shared-capital sizing.
        pos_open_value = 0.0
        for code, p in positions.items():
            op = mark_price(d, code, 'open', p['last_price'])
            p['last_price'] = op
            pos_open_value += op * p['shares']
        open_nav = cash + pos_open_value

        # Orders were chosen at prior T close. No same-day reselection.
        todays = sorted(entry_map.get(d, []), key=lambda r: (float(r.route_score) + float(r.quality_score), float(r.candidate_score)), reverse=True)
        for r in todays:
            if len(positions) >= MAX_POS:
                break
            if r.code in positions:
                continue
            ep = float(r.entry_price)
            current_exposure = sum(mark_price(d, c, 'open', p['last_price']) * p['shares'] for c, p in positions.items())
            max_new_exposure_cash = max(0.0, MAX_EXPOSURE * open_nav - current_exposure)
            budget = min(SLOT * open_nav, cash, max_new_exposure_cash)
            if budget <= ep * (1 + FEE):
                continue
            shares_cash = int(math.floor(budget / (ep * (1 + FEE))))
            shares_adv = int(math.floor(ADV_CAP * float(r.vol20)))
            shares = max(0, min(shares_cash, shares_adv))
            if shares < 1:
                continue
            cost = ep * shares * (1 + FEE)
            if cost > cash + 1e-8:
                continue
            cash -= cost
            positions[r.code] = {
                'code': r.code, 'sector': r.sector, 'playbook': r.playbook, 'quality': r.quality,
                'signal_date': int(r.signal_date), 'entry_date': int(r.entry_date), 'exit_date': int(r.exit_date),
                'entry_price': ep, 'exit_price': float(r.exit_price), 'shares': int(shares), 'cost': float(cost),
                'route_score': float(r.route_score), 'quality_score': float(r.quality_score),
                'hist_sector_n': int(r.hist_sector_n), 'hist_sector_wr': float(r.hist_sector_wr), 'hist_sector_pf': float(r.hist_sector_pf),
                'hist_quality_n': int(r.hist_quality_n), 'hist_quality_wr': float(r.hist_quality_wr), 'hist_quality_pf': float(r.hist_quality_pf),
                'last_price': ep,
            }
            if cash < -1e-6:
                raise AssertionError(f'negative cash {cash} on {d}')

        # Close mark-to-market. Missing/suspended day carries last known mark.
        pos_close_value = 0.0
        for code, p in positions.items():
            cp = mark_price(d, code, 'close', p['last_price'])
            p['last_price'] = cp
            pos_close_value += cp * p['shares']
        nav = cash + pos_close_value
        exposure = pos_close_value / nav if nav > 0 else np.nan
        nav_rows.append({'date': d, 'cash': cash, 'market_value': pos_close_value, 'nav': nav, 'positions': len(positions), 'exposure': exposure})
        min_cash = min(min_cash, cash)
        max_positions = max(max_positions, len(positions))
        max_exposure = max(max_exposure, exposure if np.isfinite(exposure) else 0.0)

    navdf = pd.DataFrame(nav_rows)
    tr = pd.DataFrame(trades)
    end_nav = float(navdf.iloc[-1].nav)
    days_elapsed = max(1, (pd.to_datetime(str(end)) - pd.to_datetime(str(start))).days)
    cagr = (end_nav / INIT) ** (365.25 / days_elapsed) - 1 if end_nav > 0 else -1.0
    dd = navdf.nav / navdf.nav.cummax() - 1.0
    wr = float((tr.realized_return > 0).mean()) if len(tr) else 0.0
    pf = pf_of(tr.realized_return) if len(tr) else 0.0
    result = {
        'label': label, 'start': start, 'end': end, 'initial': INIT, 'end_nav': end_nav,
        'cagr': float(cagr), 'max_drawdown': float(dd.min()), 'trades': int(len(tr)),
        'wins': int((tr.realized_return > 0).sum()) if len(tr) else 0, 'win_rate': wr, 'profit_factor': float(pf),
        'min_cash': float(min_cash), 'max_positions': int(max_positions), 'max_exposure': float(max_exposure),
        'cash_nonnegative': bool(min_cash >= -1e-6), 'position_cap_pass': bool(max_positions <= MAX_POS),
        'exposure_cap_pass': bool(max_exposure <= MAX_EXPOSURE + .02),
    }
    return result, tr, navdf


full, full_tr, full_nav = simulate(START, END, 'full_2023m5_2025')
dev, dev_tr, dev_nav = simulate(START, DEV_END, 'dev_2023m5_2024')
hold, hold_tr, hold_nav = simulate(HOLDOUT_START, END, 'holdout_2025')

full_tr.to_csv(OUT / 'trades_full.csv', index=False)
full_nav.to_csv(OUT / 'nav_full.csv', index=False)
dev_tr.to_csv(OUT / 'trades_dev.csv', index=False)
hold_tr.to_csv(OUT / 'trades_holdout2025.csv', index=False)

route_counts = Counter()
for d, ps in route_cache.items():
    for p in ps:
        route_counts[(p['sector'], p['playbook'], p['quality'])] += 1
route_df = pd.DataFrame([
    {'sector': k[0], 'playbook': k[1], 'quality': k[2], 'selected_days': v}
    for k, v in route_counts.items()
]).sort_values('selected_days', ascending=False) if route_counts else pd.DataFrame(columns=['sector','playbook','quality','selected_days'])
route_df.to_csv(OUT / 'route_selection_counts.csv', index=False)

summary = pd.DataFrame([dev, hold, full])
summary.to_csv(OUT / 'portfolio_summary.csv', index=False)

target_full = bool(full['cagr'] >= .50 and full['win_rate'] >= .70 and full['trades'] >= 30)
target_hold = bool(hold['cagr'] >= .50 and hold['win_rate'] >= .70 and hold['trades'] >= 15)
valid = bool(full['cash_nonnegative'] and full['position_cap_pass'] and full['exposure_cap_pass'] and hold['cash_nonnegative'])
decision = {
    'status': 'PASS' if valid else 'FAIL_ACCOUNTING',
    'architecture': 'continuous market+cycle context -> sector/playbook -> separate quality filter -> shared-capital T+1 portfolio',
    'semantic_bull_bear_labels_used': False,
    'future_leakage_policy': 'neighbors and policy statistics use only dates before T; outcomes require exit_date < T',
    'corporate_action_policy': 'candidate intervals crossing official corporate-action dates are excluded at this research stage',
    'sector_proxy_policy': 'fixed ex-ante code-range proxies: financial 28xx, electronics 23xx/24xx, transport 26xx, materials 20xx, other otherwise',
    'full': full,
    'dev': dev,
    'holdout': hold,
    'target': {'cagr': .50, 'win_rate': .70},
    'target_hit_full': target_full,
    'target_hit_holdout': target_hold,
    'minute_stage_allowed': bool(valid and target_full and target_hold),
    'formal_r10_untouched': True,
}
(OUT / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
print(summary.to_string(index=False))
print('\nTOP ROUTES')
print(route_df.head(20).to_string(index=False))
print(json.dumps(decision, ensure_ascii=False, indent=2))
