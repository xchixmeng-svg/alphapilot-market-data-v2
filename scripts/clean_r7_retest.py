#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from collections.abc import Mapping, Iterator

import numpy as np
import pandas as pd

from clean_event_loop import Bar, BuyIntent, SellIntent, DecisionContext, PortfolioEngine, floor_tick
from clean_features import residual_adjust

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.clean_cache'
OUT = ROOT / 'clean_results'
OUT.mkdir(exist_ok=True)
INITIAL = 1_000_000.0
EVAL_START = 20210101
EVAL_END = 20251231

# Archived PIT-safe 0050 total-return reconstruction used by the historical R7 line.
CASH_DISTRIBUTIONS = {
    20210122: 3.05, 20210721: 0.35, 20220121: 3.20, 20220718: 1.80,
    20230130: 2.60, 20230718: 1.90, 20240117: 3.00, 20240716: 1.00,
    20250117: 2.70, 20250721: 0.36,
}
FORMAL_MISSING = {20230525, 20250206, 20250611, 20250612, 20250613, 20250616, 20250617}
BASE_DATE = 20210104
BASE_NAV = 2_000_000.0
SPLIT_DATE = 20250618
SPLIT_FACTOR = 4.0


def load_0050_total_return(all_dates: list[int], raw: pd.DataFrame) -> pd.DataFrame:
    rr = raw.copy()
    rr['code'] = rr.code.astype(str).str.strip().str.zfill(4)
    q = rr[rr.code.eq('0050')][['date','close']].copy()
    q['date'] = pd.to_numeric(q.date, errors='coerce')
    q['close'] = pd.to_numeric(q.close, errors='coerce')
    q = q.dropna().astype({'date':int}).drop_duplicates('date', keep='last').sort_values('date')
    if q.empty or BASE_DATE not in set(q.date):
        raise RuntimeError('raw 0050 base date missing')
    q['px'] = q['close'].astype(float)
    q.loc[q.date >= SPLIT_DATE, 'px'] *= SPLIT_FACTOR
    q['cash_dividend'] = q.date.map(CASH_DISTRIBUTIONS).fillna(0.0).astype(float)
    q.loc[q.date >= SPLIT_DATE, 'cash_dividend'] *= SPLIT_FACTOR
    live = q[q.date >= BASE_DATE].copy()
    live['gross'] = (live['px'] + live['cash_dividend']) / live['px'].shift(1)
    live.loc[live.date.eq(BASE_DATE), 'gross'] = 1.0
    live['mkt'] = BASE_NAV * live['gross'].cumprod()
    live.loc[live.date.isin(FORMAL_MISSING), 'mkt'] = np.nan
    base_px = float(q.loc[q.date.eq(BASE_DATE), 'close'].iloc[0])
    warm = q[q.date < BASE_DATE][['date','close']].copy()
    warm['mkt'] = warm['close'].astype(float) * (BASE_NAV / base_px)
    src = pd.concat([warm[['date','mkt']], live[['date','mkt']]], ignore_index=True)
    src = src.drop_duplicates('date', keep='last').sort_values('date')
    cal = pd.DataFrame({'date': sorted(set(int(x) for x in all_dates))})
    out = cal.merge(src, on='date', how='left').sort_values('date')
    out['mkt'] = out['mkt'].ffill()
    if out.mkt.isna().any():
        raise RuntimeError('0050 total return unavailable without future fill')
    g = out.mkt
    out['mkt_ma60'] = g.rolling(60, min_periods=60).mean()
    out['mkt_ma120'] = g.rolling(120, min_periods=120).mean()
    out['mkt_ret20'] = g.pct_change(20, fill_method=None)
    out['mkt_ret60'] = g.pct_change(60, fill_method=None)
    return out


def build_r7_features(raw: pd.DataFrame) -> pd.DataFrame:
    q, _ = residual_adjust(raw)
    q['code'] = q.code.astype(str).str.strip().str.zfill(4)
    q['amount'] = q.close.astype(float) * q.volume.astype(float)
    g = q.groupby('code', sort=False)
    q['ret1'] = g.aclose.pct_change(1, fill_method=None)
    q['ret10'] = g.aclose.pct_change(10, fill_method=None)
    q['ret20'] = g.aclose.pct_change(20, fill_method=None)
    q['ret60'] = g.aclose.pct_change(60, fill_method=None)
    q['ma60'] = g.aclose.rolling(60, min_periods=60).mean().reset_index(level=0, drop=True)
    q['ma120'] = g.aclose.rolling(120, min_periods=120).mean().reset_index(level=0, drop=True)
    q['high60'] = g.ahigh.rolling(60, min_periods=60).max().reset_index(level=0, drop=True)
    q['avgamt5'] = g.amount.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    q['avgamt20'] = g.amount.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    sign_amt = np.sign(q['ret1'].fillna(0.0)) * q['amount']
    q['flow_num'] = sign_amt.groupby(q.code).rolling(20, min_periods=20).sum().reset_index(level=0, drop=True)
    q['flow_den'] = q.amount.groupby(q.code).rolling(20, min_periods=20).sum().reset_index(level=0, drop=True)
    q['flow20'] = q.flow_num / q.flow_den.replace(0, np.nan)
    rng = (q.high - q.low).replace(0, np.nan)
    q['clv'] = ((2*q.close - q.high - q.low) / rng).fillna(0.0).clip(-1,1)
    clv_amt = q.clv * q.amount
    q['clv_num'] = clv_amt.groupby(q.code).rolling(20, min_periods=20).sum().reset_index(level=0, drop=True)
    q['clvflow20'] = q.clv_num / q.flow_den.replace(0, np.nan)
    q['amt_acc'] = q.avgamt5 / q.avgamt20.replace(0, np.nan) - 1.0
    q['nearhigh'] = q.aclose / q.high60.replace(0, np.nan)

    all_dates = sorted(int(x) for x in q.date.unique())
    bm = load_0050_total_return(all_dates, raw)
    q = q.merge(bm, on='date', how='left', validate='many_to_one')
    q['rel20'] = q.ret20 - q.mkt_ret20
    q['rel60'] = q.ret60 - q.mkt_ret60

    ordinary = q.code.str.fullmatch(r'\d{4}') & ~q.code.str.startswith('00') & ~q['name'].astype(str).str.contains('KY', case=False, na=False)
    u = q[ordinary].copy()
    liquid = u.avgamt20 >= 30_000_000.0
    u['breadth_flag'] = np.where(liquid, (u.aclose > u.ma60).astype(float), np.nan)
    breadth = u.groupby('date').breadth_flag.mean().rename('breadth60')
    adv = u.assign(adv_flag=np.where(liquid, (u.ret10 > 0).astype(float), np.nan)).groupby('date').adv_flag.mean().rename('advance10')
    md = pd.concat([breadth, adv], axis=1).reset_index()
    md['breadth20mean'] = md.breadth60.rolling(20, min_periods=20).mean()
    u = u.merge(md, on='date', how='left', validate='many_to_one')

    for c in ['ret10','rel20','rel60','flow20','amt_acc','clvflow20','nearhigh']:
        u[f'pct_{c}'] = u.groupby('date')[c].rank(pct=True, method='average')
    u['score'] = (
        0.26*u.pct_ret10 + 0.22*u.pct_rel20 + 0.10*u.pct_rel60 +
        0.14*u.pct_flow20 + 0.12*u.pct_amt_acc + 0.08*u.pct_clvflow20 + 0.08*u.pct_nearhigh
    )
    return u.sort_values(['date','code']).reset_index(drop=True)


class DayBars(Mapping):
    def __init__(self, q: pd.DataFrame): self.q = q
    def __len__(self): return len(self.q)
    def __iter__(self) -> Iterator[str]: return iter(self.q.index)
    def __getitem__(self, code):
        r = self.q.loc[code]
        if isinstance(r, pd.DataFrame): r = r.iloc[-1]
        return Bar(str(int(r.date)), str(code), str(r['name']), float(r.open), float(r.high), float(r.low), float(r.close))
    def get(self, code, default=None):
        try: return self[code]
        except KeyError: return default


class BarStore:
    def __init__(self, raw: pd.DataFrame):
        self.by_date = {str(int(d)): z.set_index('code', drop=False) for d,z in raw.groupby('date', sort=True)}
    def __getitem__(self, date): return DayBars(self.by_date[date])


def alloc_weights(n: int, exposure: float, cap: float = 0.20) -> list[float]:
    if n <= 0 or exposure <= 0: return []
    raw = np.array([(n-i)**2.5 for i in range(n)], dtype=float)
    w = raw / raw.sum() * exposure
    active = np.ones(n, dtype=bool)
    while True:
        over = (w > cap + 1e-12) & active
        if not over.any(): break
        excess = float((w[over] - cap).sum())
        w[over] = cap; active[over] = False
        if excess <= 1e-12 or not active.any(): break
        rem = raw[active]
        w[active] += excess * rem / rem.sum()
    return [float(min(x, cap)) for x in w]


def shares_for_target(nav: float, limit: float, target_pct: float) -> int:
    target = min(nav * target_pct, nav * 0.20)
    if target <= 0 or limit <= 0: return 0
    one_lot = limit * 1000
    if one_lot <= target + 1e-9:
        return int(math.floor(target / one_lot)) * 1000
    return int(math.floor(target / limit / 100.0)) * 100


class R7CleanStrategy:
    def __init__(self, feat: pd.DataFrame):
        self.by_date = {str(int(d)): z.set_index('code', drop=False) for d,z in feat.groupby('date', sort=True)}
        self.session = 0
        self.last_regime = None

    def _regime(self, sample):
        if sample is None: return ('BEAR', 0.0, 0)
        m = float(sample.mkt); ma60=float(sample.mkt_ma60); ma120=float(sample.mkt_ma120)
        r20=float(sample.mkt_ret20); r60=float(sample.mkt_ret60)
        b=float(sample.breadth60); bm=float(sample.breadth20mean); a=float(sample.advance10)
        if not all(np.isfinite(x) for x in [m,ma60,ma120,r20,r60,b]): return ('BEAR',0.0,0)
        if r20 <= -0.08 or (m < ma120 and r60 < 0 and b < 0.40): return ('BEAR',0.0,0)
        if m < ma120*1.02 and r20 > 0 and b > 0.42 and np.isfinite(bm) and b > bm: return ('REPAIR',0.60,2)
        if m > ma60 and m > ma120 and r20 > 0 and r60 > 0 and b >= 0.60 and np.isfinite(a) and a >= 0.52: return ('STRONG_BULL',1.00,4)
        if m > ma120 and r60 > 0 and b >= 0.45: return ('NORMAL_BULL',0.80,3)
        if m > ma120*0.98 and b >= 0.38: return ('WEAK',0.20,2)
        return ('BEAR',0.0,0)

    def decide(self, ctx: DecisionContext):
        self.session += 1
        date = str(int(ctx.date.replace('-',''))) if '-' in ctx.date else ctx.date
        q = self.by_date.get(date)
        sample = None if q is None or q.empty else q.iloc[0]
        regime, exposure, nslots = self._regime(sample)
        regime_changed = self.last_regime is None or regime != self.last_regime
        rebalance = self.session == 1 or ((self.session-1) % 15 == 0) or regime_changed
        self.last_regime = regime
        if q is None or q.empty:
            return [], []
        eligible = q[(q.avgamt20 >= 30_000_000.0) & (q.aclose > q.ma120) & (q.nearhigh >= 0.78) & q.score.notna()].copy().reset_index(drop=True)
        eligible = eligible.sort_values(['score','code'], ascending=[False,True])
        ranks = {str(c): i+1 for i,c in enumerate(eligible.code.astype(str).tolist())}

        sells = []
        for code,pos in ctx.positions.items():
            r = q.loc[code] if code in q.index else None
            if isinstance(r, pd.DataFrame): r = r.iloc[-1]
            if r is not None and float(r.aclose) <= float(pos.entry_price)*0.88:
                sells.append(SellIntent(code, True, reason='R7_HARD_STOP'))
        already = {x.code for x in sells}
        if rebalance:
            if nslots == 0:
                for code in ctx.positions:
                    if code not in already: sells.append(SellIntent(code, True, reason='R7_REGIME_EXIT'))
            else:
                for code in ctx.positions:
                    if code in already: continue
                    rank = ranks.get(code, 10**9)
                    if rank > 2*nslots:
                        sells.append(SellIntent(code, True, reason='R7_RANK_EXIT'))
        already = {x.code for x in sells}

        # Original R7 exposure trim: full-position exits from weakest ranks.
        if exposure <= 0:
            pass
        else:
            mv = sum(p.shares * (float(q.loc[c].close) if c in q.index and not isinstance(q.loc[c], pd.DataFrame) else p.last_close) for c,p in ctx.positions.items() if c not in already)
            limit_mv = ctx.nav * exposure * 1.03
            if mv > limit_mv + 1e-6:
                weakest = sorted([c for c in ctx.positions if c not in already], key=lambda c: ranks.get(c,10**9), reverse=True)
                for c in weakest:
                    if mv <= limit_mv + 1e-6: break
                    p=ctx.positions[c]; px=float(q.loc[c].close) if c in q.index and not isinstance(q.loc[c],pd.DataFrame) else p.last_close
                    sells.append(SellIntent(c, True, reason='R7_EXPOSURE_EXIT')); already.add(c); mv -= p.shares*px

        if not rebalance or nslots <= 0:
            return sells, []
        surviving = [c for c in ctx.positions if c not in already]
        capacity = max(0, nslots - len(surviving))
        if capacity <= 0:
            return sells, []
        candidates = eligible[~eligible.code.astype(str).isin(set(ctx.positions) | already)].head(capacity)
        if candidates.empty:
            return sells, []
        weights = alloc_weights(nslots, exposure, 0.20)
        start_rank = len(surviving)
        buys=[]
        for j,r in enumerate(candidates.itertuples(index=False)):
            slot_index = min(start_rank+j, len(weights)-1)
            target_pct = weights[slot_index] if weights else 0.0
            limit = floor_tick(float(r.aclose)*0.98)
            shares = shares_for_target(ctx.nav, limit, target_pct)
            if shares > 0:
                buys.append(BuyIntent(str(r.code), shares, limit, 'R7_DEEP_LIMIT'))
        return sells, buys


def annual_stats(nav_rows):
    rows = pd.DataFrame([{'date':int(x.date),'nav':float(x.nav)} for x in nav_rows])
    rows['year'] = rows.date.astype(str).str[:4].astype(int)
    out={}; prev=INITIAL
    for y in sorted(rows.year.unique()):
        z=rows[rows.year.eq(y)]; end=float(z.iloc[-1].nav)
        vals=np.r_[prev,z.nav.to_numpy(float)]; peaks=np.maximum.accumulate(vals)
        out[str(y)]={'return':end/prev-1.0,'max_dd':float(np.min(vals/peaks-1.0)),'end_nav':end}; prev=end
    return out


def main():
    raw = pd.concat([pd.read_parquet(CACHE/f'ohlcv_{y}.parquet') for y in range(2020,2026)], ignore_index=True)
    raw['code'] = raw.code.astype(str).str.strip().str.zfill(4)
    feat = build_r7_features(raw)
    eval_raw = raw[raw.date.between(EVAL_START,EVAL_END) & ~raw.code.str.startswith('00')].copy()
    eval_dates = [str(int(x)) for x in sorted(eval_raw.date.unique())]
    if len(eval_dates) < 1200: raise RuntimeError(f'too few eval dates: {len(eval_dates)}')
    print(json.dumps({'stage':'START','dates':len(eval_dates),'initial_capital':INITIAL,'strategy':'R7 logic under Clean Contract','clean_single_name_cap':0.20}), flush=True)
    result = PortfolioEngine(INITIAL).run(eval_dates, BarStore(eval_raw), R7CleanStrategy(feat))
    end=float(result['end_nav']); cagr=(end/INITIAL)**(1/5)-1
    ann=annual_stats(result['nav_rows'])
    led=pd.DataFrame([asdict(x) for x in result['ledger']])
    orders=pd.DataFrame(result['order_log'])
    nav=pd.DataFrame([asdict(x) for x in result['nav_rows']])
    summary={
        'status':'PASS' if float(result['max_drawdown']) > -0.20 else 'REJECT_DD_GATE',
        'strategy':'R7 historical selection/exit logic revalidated under current Clean Contract',
        'initial_capital':INITIAL,'evaluation':'2021-2025','warmup':'2020',
        'end_nav':end,'total_return':end/INITIAL-1.0,'cagr':float(cagr),
        'max_drawdown':float(result['max_drawdown']),'completed_trades':int(result['completed_trades']),
        'orders':int(result['orders']),'annual':ann,
        'contract_overrides_vs_historical_r7':{
            'single_name_cap':'20% Clean Contract instead of old R7 30%',
            'commission':'0.1425% each side, no discount','sell_execution':'T+1 open x 0.98 adverse model',
            'shares':'100-share increments / board-lot preference','buy_execution':'precommitted T+1 limit with 0.5% adverse open fill model'
        },
        'note':'This is a clean-engine revalidation, not the obsolete archived R7 performance claim.'
    }
    (OUT/'r7_clean_retest_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    led.to_csv(OUT/'r7_clean_retest_trades.csv',index=False)
    orders.to_csv(OUT/'r7_clean_retest_orders.csv',index=False)
    nav.to_csv(OUT/'r7_clean_retest_nav.csv',index=False)
    print(json.dumps(summary,ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
