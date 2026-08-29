#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from clean_features import residual_adjust
from clean_r7_retest import build_r7_features, floor_tick

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.clean_cache'
OUT = ROOT / 'clean_results'
OUT.mkdir(exist_ok=True)

INITIAL = 1_300_000.0
EVAL_START = 20210104
EVAL_END = 20251231
BUY_FEE = 0.000855
SELL_FEE = 0.000855
SELL_TAX = 0.003
SELL_SLIP = 0.005
MAX_POSITIONS = 5
MAX_SINGLE = 0.25
MAX_EXPOSURE = 0.95
ADV_CAP = 0.02
R7_BASE = 0.22
R05_BASE = 0.20


def tick(price: float) -> float:
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1.0
    return 5.0


def ceil_tick(price: float) -> float:
    t = tick(price)
    q = math.floor((price + 1e-12) / t)
    base = q * t
    return round(base if abs(base-price) < 1e-10 else (q+1)*t, 8)


def load_raw() -> pd.DataFrame:
    parts=[]
    for y in range(2020, 2026):
        p=CACHE/f'ohlcv_{y}.parquet'
        if not p.exists(): raise RuntimeError(f'missing cache {p}')
        parts.append(pd.read_parquet(p))
    q=pd.concat(parts,ignore_index=True)
    q['code']=q.code.astype(str).str.strip().str.zfill(4)
    return q.drop_duplicates(['date','code'],keep='last').sort_values(['code','date']).reset_index(drop=True)


def build_r05_features(raw: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    q,_=residual_adjust(raw)
    q['code']=q.code.astype(str).str.strip().str.zfill(4)
    q['amount']=q.close.astype(float)*q.volume.astype(float)
    g=q.groupby('code',sort=False)
    q['ret20']=g.aclose.pct_change(20,fill_method=None)
    q['ma20']=g.aclose.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['avgamt5']=g.amount.rolling(5,min_periods=5).mean().reset_index(level=0,drop=True)
    q['avgamt20']=g.amount.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['avgvol20']=g.volume.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['amt_ratio']=q.avgamt5/q.avgamt20.replace(0,np.nan)
    q['ma20_gap']=q.aclose/q.ma20.replace(0,np.nan)-1.0
    q['prior60high']=g.ahigh.shift(1).groupby(q.code).rolling(60,min_periods=60).max().reset_index(level=0,drop=True)
    q['prior10high']=g.ahigh.shift(1).groupby(q.code).rolling(10,min_periods=10).max().reset_index(level=0,drop=True)
    q['near_prior60']=q.aclose/q.prior60high.replace(0,np.nan)-1.0
    rng=(q.high-q.low).replace(0,np.nan)
    q['clv']=((2*q.close-q.high-q.low)/rng).fillna(0.0).clip(-1,1)
    ca=q.clv*q.amount
    for n in (5,10):
        num=ca.groupby(q.code).rolling(n,min_periods=n).sum().reset_index(level=0,drop=True)
        den=q.amount.groupby(q.code).rolling(n,min_periods=n).sum().reset_index(level=0,drop=True)
        q[f'clvflow{n}']=num/den.replace(0,np.nan)

    ii=inst.copy()
    ii['code']=ii.code.astype(str).str.strip().str.zfill(4)
    ii=ii.groupby(['date','code'],as_index=False)[['foreign_net','trust_net']].sum()
    q=q.merge(ii,on=['date','code'],how='left')
    q[['foreign_net','trust_net']]=q[['foreign_net','trust_net']].fillna(0.0)
    gg=q.groupby('code',sort=False)
    q['foreign3']=gg.foreign_net.rolling(3,min_periods=3).sum().reset_index(level=0,drop=True)
    q['foreign10']=gg.foreign_net.rolling(10,min_periods=10).sum().reset_index(level=0,drop=True)
    q['trust5']=gg.trust_net.rolling(5,min_periods=5).sum().reset_index(level=0,drop=True)

    r7=build_r7_features(raw)
    market=r7[['date','mkt','mkt_ma60','mkt_ret20','mkt_ret60']].drop_duplicates('date')
    q=q.merge(market,on='date',how='left',validate='many_to_one')

    ordinary=q.code.str.fullmatch(r'\d{4}') & ~q.code.str.startswith('00') & ~q['name'].astype(str).str.contains('KY',case=False,na=False)
    u=q[ordinary].copy()
    for c in ['clvflow10','amt_ratio','clvflow5','foreign3','foreign10','trust5']:
        u[f'pct_{c}']=u.groupby('date')[c].rank(pct=True,method='average')
    u['score']=(
        0.5251*u.pct_clvflow10 + 0.2465*u.pct_amt_ratio + 0.0683*u.pct_clvflow5 +
        0.0628*u.pct_foreign3 - 0.0778*u.pct_foreign10 + 0.0195*u.pct_trust5 -
        0.2000*u.ma20_gap.clip(lower=0.0)
    )
    u['risk_on']=(u.mkt>u.mkt_ma60)&(u.mkt_ret20>0)&(u.mkt_ret60>0)
    return u.sort_values(['date','code']).reset_index(drop=True)


@dataclass
class Pos:
    code: str
    name: str
    strategy: str
    shares: int
    entry_date: int
    entry_price: float
    cost: float
    hold: int = 1
    peak_return: float = 0.0
    runner: str = 'BASE'


@dataclass
class BuyOrder:
    decision_date: int
    execute_date: int
    strategy: str
    code: str
    name: str
    shares: int
    limit: float
    target_cash: float
    reason: str


@dataclass
class SellOrder:
    decision_date: int
    execute_date: int
    code: str
    reason: str


def share_size(target: float, limit: float, adv20: float) -> int:
    if not np.isfinite(target) or target<=0 or limit<=0: return 0
    if limit*1000 <= target+1e-9:
        s=int(target//(limit*1000))*1000
    else:
        s=int(target//limit)
    if np.isfinite(adv20) and adv20>0:
        s=min(s,int(math.floor(adv20*ADV_CAP)))
    if s>=1000: s=(s//1000)*1000
    return max(0,int(s))


def annual_returns(nav: pd.DataFrame) -> dict[str,float]:
    out={}
    for y,z in nav.groupby(nav.date.astype(str).str[:4]):
        start=float(z.iloc[0].nav); end=float(z.iloc[-1].nav)
        out[str(y)]=end/start-1.0
    return out


class R10:
    def __init__(self, raw: pd.DataFrame, r7: pd.DataFrame, r05: pd.DataFrame):
        self.raw_by_date={int(d):z.set_index('code',drop=False) for d,z in raw.groupby('date',sort=True)}
        self.r7_by_date={int(d):z.set_index('code',drop=False) for d,z in r7.groupby('date',sort=True)}
        self.r05_by_date={int(d):z.set_index('code',drop=False) for d,z in r05.groupby('date',sort=True)}
        self.cash=INITIAL; self.positions={}; self.pending_buys={}; self.pending_sells={}
        self.nav_rows=[]; self.trades=[]; self.orders=[]; self.peak=INITIAL
        self.r7_session=0; self.r7_last_regime=None
        self.no_new_until_idx=-1; self.force_cooldown_until=-1

    def r7_regime(self,s):
        if s is None: return ('BEAR',0.0,0)
        vals=[s.mkt,s.mkt_ma60,s.mkt_ma120,s.mkt_ret20,s.mkt_ret60,s.breadth60]
        if not all(np.isfinite(float(x)) for x in vals): return ('BEAR',0.0,0)
        m,ma60,ma120,r20,r60,b=map(float,vals); bm=float(s.breadth20mean); a=float(s.advance10)
        if r20<=-0.08 or (m<ma120 and r60<0 and b<0.40): return ('BEAR',0.0,0)
        if m<ma120*1.02 and r20>0 and b>0.42 and np.isfinite(bm) and b>bm: return ('REPAIR',0.60,2)
        if m>ma60 and m>ma120 and r20>0 and r60>0 and b>=0.60 and np.isfinite(a) and a>=0.52: return ('STRONG_BULL',1.00,4)
        if m>ma120 and r60>0 and b>=0.45: return ('BULL',0.80,3)
        if m>ma120*0.98 and b>=0.38: return ('CAUTIOUS',0.20,2)
        return ('BEAR',0.0,0)

    def marks(self,date):
        bars=self.raw_by_date.get(date)
        mv=0.0
        for c,p in self.positions.items():
            if bars is not None and c in bars.index: mv+=p.shares*float(bars.loc[c].close)
            else: mv+=p.shares*p.entry_price
        return mv

    def execute(self,date,next_date):
        bars=self.raw_by_date.get(date)
        if bars is None: return
        # Binding sells execute first; 0.5% adverse to T+1 open.
        for o in self.pending_sells.pop(date,[]):
            p=self.positions.get(o.code)
            if p is None: continue
            if o.code not in bars.index or float(bars.loc[o.code].open)<=0:
                if next_date is not None: self.pending_sells.setdefault(next_date,[]).append(SellOrder(o.decision_date,next_date,o.code,o.reason))
                self.orders.append({'decision_date':o.decision_date,'execute_date':date,'side':'SELL','code':o.code,'filled':False,'reason':'NO_BAR_DEFERRED'})
                continue
            raw_open=float(bars.loc[o.code].open); px=floor_tick(raw_open*(1-SELL_SLIP))
            gross=px*p.shares; fee=gross*SELL_FEE; tax=gross*SELL_TAX; proceeds=gross-fee-tax
            pnl=proceeds-p.cost
            self.cash+=proceeds
            self.trades.append({'strategy':p.strategy,'code':p.code,'name':p.name,'entry_date':p.entry_date,'exit_date':date,'shares':p.shares,'buy_price':p.entry_price,'sell_open':raw_open,'sell_price':px,'net_pnl':pnl,'net_return':pnl/p.cost if p.cost else 0.0,'hold_sessions':p.hold,'exit_reason':o.reason})
            self.orders.append({'decision_date':o.decision_date,'execute_date':date,'side':'SELL','code':o.code,'shares':p.shares,'filled':True,'fill_price':px,'reason':o.reason})
            del self.positions[o.code]

        nav_pre=self.cash+self.marks(date)
        for o in self.pending_buys.pop(date,[]):
            if o.code in self.positions: continue
            if o.code not in bars.index:
                self.orders.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'limit_price':o.limit,'filled':False,'reason':'NO_BAR'})
                continue
            b=bars.loc[o.code]; op=float(b.open); lo=float(b.low)
            if op<=0 or lo<=0 or (op>o.limit and lo>o.limit):
                self.orders.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'limit_price':o.limit,'filled':False,'reason':'LIMIT_NOT_TOUCHED'})
                continue
            px=ceil_tick(min(op,o.limit) if op<=o.limit else o.limit)
            px=min(px,o.limit); gross=px*o.shares; fee=gross*BUY_FEE; cost=gross+fee
            if gross>nav_pre*MAX_SINGLE+1e-6 or cost>self.cash+1e-6:
                self.orders.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'limit_price':o.limit,'filled':False,'reason':'CAP_OR_CASH'})
                continue
            self.cash-=cost
            self.positions[o.code]=Pos(o.code,o.name,o.strategy,o.shares,date,px,cost)
            self.orders.append({'decision_date':o.decision_date,'execute_date':date,'side':'BUY','code':o.code,'shares':o.shares,'limit_price':o.limit,'filled':True,'fill_price':px,'target_cash':o.target_cash,'reason':o.reason})

    def decide(self,idx,date,next_date,nav,dd):
        r7q=self.r7_by_date.get(date); r05q=self.r05_by_date.get(date)
        self.r7_session+=1
        sample=None if r7q is None or r7q.empty else r7q.iloc[0]
        regime,r7_exposure,r7_slots=self.r7_regime(sample)
        regime_changed=self.r7_last_regime is None or regime!=self.r7_last_regime
        r7_rebalance=(self.r7_session==1 or (self.r7_session-1)%15==0 or regime_changed)
        self.r7_last_regime=regime

        mult=1.0
        if dd<=-0.15: mult=0.40
        elif dd<=-0.09: mult=0.45
        elif dd<=-0.06: mult=0.85

        r7_elig=pd.DataFrame()
        ranks={}
        if r7q is not None and not r7q.empty:
            r7_elig=r7q[(r7q.avgamt20>=30_000_000)&(r7q.aclose>r7q.ma120)&(r7q.nearhigh>=0.78)&r7q.score.notna()].copy().reset_index(drop=True)
            r7_elig=r7_elig.sort_values(['score','code'],ascending=[False,True])
            ranks={str(c):i+1 for i,c in enumerate(r7_elig.code.astype(str).tolist())}

        sells=[]
        # Strategy-owned exits.
        for c,p in list(self.positions.items()):
            if p.strategy=='R7':
                rr=None if r7q is None or c not in r7q.index else r7q.loc[c]
                if isinstance(rr,pd.DataFrame): rr=rr.iloc[-1]
                if rr is not None and float(rr.aclose)<=p.entry_price*0.88: sells.append((c,'R7_HARD_STOP'))
                elif r7_rebalance and (r7_slots==0 or ranks.get(c,10**9)>2*r7_slots): sells.append((c,'R7_REGIME_EXIT' if r7_slots==0 else 'R7_RANK_EXIT'))
            else:
                rr=None if r05q is None or c not in r05q.index else r05q.loc[c]
                if isinstance(rr,pd.DataFrame): rr=rr.iloc[-1]
                if rr is None: continue
                ret=float(rr.aclose)/p.entry_price-1.0
                p.peak_return=max(p.peak_return,ret)
                ar=float(rr.amt_ratio) if np.isfinite(rr.amt_ratio) else 0.0
                if p.runner!='MEGA' and ret>=0.80 and ar>=1.2: p.runner='MEGA'
                elif p.runner=='BASE' and ret>=0.40 and ar>=2.0: p.runner='SUPER'
                if ret<=-0.10: sells.append((c,'R05_HARD_STOP'))
                else:
                    trail=None
                    if p.runner=='MEGA': trail=0.16
                    elif p.runner=='SUPER': trail=0.14
                    elif p.peak_return>=0.50: trail=0.12
                    if trail is not None and ret <= (1+p.peak_return)*(1-trail)-1: sells.append((c,f'R05_{p.runner}_TRAIL'))
                    elif p.runner=='BASE' and p.hold>=60: sells.append((c,'R05_MAX_HOLD_60'))
                    elif p.runner!='BASE' and p.hold>=120: sells.append((c,'R05_RUNNER_MAX_HOLD_120'))
                    elif p.runner!='MEGA' and ret>=2.0: sells.append((c,'R05_TARGET_200'))
                    elif p.runner=='BASE' and p.peak_return>=0.20 and ret<=0.0: sells.append((c,'R05_20PCT_PROTECT'))

        sellcodes={c for c,_ in sells}

        # Preserve R7's source-regime exposure exit on R7 positions.
        if r7_rebalance and r7_exposure>0:
            r7_mv=0.0
            bars=self.raw_by_date.get(date)
            for c,p in self.positions.items():
                if p.strategy=='R7' and c not in sellcodes:
                    px=float(bars.loc[c].close) if bars is not None and c in bars.index else p.entry_price
                    r7_mv+=p.shares*px
            limit_mv=nav*r7_exposure*1.03
            if r7_mv>limit_mv:
                weak=sorted([c for c,p in self.positions.items() if p.strategy=='R7' and c not in sellcodes],key=lambda x:ranks.get(x,10**9),reverse=True)
                for c in weak:
                    if r7_mv<=limit_mv: break
                    p=self.positions[c]; px=float(bars.loc[c].close) if bars is not None and c in bars.index else p.entry_price
                    sells.append((c,'R7_EXPOSURE_EXIT')); sellcodes.add(c); r7_mv-=p.shares*px

        # Portfolio force-reduce: DD<=-14%, target about 50%, 15-day cooldown, 10-day no-new window.
        if dd<=-0.14 and idx>self.force_cooldown_until:
            self.force_cooldown_until=idx+15; self.no_new_until_idx=max(self.no_new_until_idx,idx+10)
            bars=self.raw_by_date.get(date)
            cur=[]; mv=0.0
            for c,p in self.positions.items():
                if c in sellcodes: continue
                px=float(bars.loc[c].close) if bars is not None and c in bars.index else p.entry_price
                mv+=p.shares*px
                score=-1e9
                if p.strategy=='R7' and r7q is not None and c in r7q.index: score=float(r7q.loc[c].score)
                if p.strategy=='R0.5' and r05q is not None and c in r05q.index: score=float(r05q.loc[c].score)
                cur.append((score,c,p.shares*px))
            for _,c,val in sorted(cur):
                if mv<=nav*0.50: break
                sells.append((c,'PORTFOLIO_DD_FORCE_REDUCE')); sellcodes.add(c); mv-=val

        for c,reason in sells:
            self.pending_sells.setdefault(next_date,[]).append(SellOrder(date,next_date,c,reason))

        if idx<=self.no_new_until_idx: return
        survivors=[c for c in self.positions if c not in sellcodes]
        capacity=MAX_POSITIONS-len(survivors)
        if capacity<=0: return

        bars=self.raw_by_date.get(date)
        if bars is None: return
        current_mv=sum(self.positions[c].shares*float(bars.loc[c].close) for c in survivors if c in bars.index)
        exposure_room=max(0.0,nav*MAX_EXPOSURE-current_mv)
        cash_room=self.cash
        if exposure_room<=0 or cash_room<=0: return

        candidates=[]
        held=set(self.positions)|sellcodes
        if r7_rebalance and r7_slots>0:
            active_r7=sum(1 for c in survivors if self.positions[c].strategy=='R7')
            room=max(0,r7_slots-active_r7)
            for r in r7_elig[~r7_elig.code.astype(str).isin(held)].head(room).itertuples(index=False):
                candidates.append(('R7',float(r.score),str(r.code),str(r.name),float(r.aclose),float(r.avgamt20),float(getattr(r,'avgvol20',np.nan)) if hasattr(r,'avgvol20') else np.nan))

        if r05q is not None and not r05q.empty and bool(r05q.iloc[0].risk_on):
            active05=sum(1 for c in survivors if self.positions[c].strategy=='R0.5')
            room=max(0,3-active05)
            elig=r05q[(r05q.aclose>=10)&(r05q.aclose<=40)&(r05q.avgamt20>=50_000_000)&(r05q.amt_ratio>=1.0)&(r05q.ret20>=0)&(r05q.ret20<=0.20)&(r05q.ma20_gap<=0.18)&(r05q.near_prior60>=-0.15)&(r05q.aclose>r05q.prior10high)&r05q.score.notna()].copy().reset_index(drop=True)
            elig=elig.sort_values(['score','code'],ascending=[False,True])
            for r in elig[~elig.code.astype(str).isin(held)].head(room).itertuples(index=False):
                candidates.append(('R0.5',float(r.score),str(r.code),str(r.name),float(r.aclose),float(r.avgamt20),float(r.avgvol20)))

        # Same-day cross-strategy competition: normalized source score, deterministic code tie-break.
        candidates=sorted(candidates,key=lambda x:(-x[1],x[0],x[2]))[:capacity]
        reserved=0.0; planned_mv=current_mv
        for strat,score,code,name,aclose,avgamt,avgvol in candidates:
            if code in held: continue
            limit=floor_tick(aclose*(0.98 if strat=='R7' else 0.995))
            base=(R7_BASE if strat=='R7' else R05_BASE)*nav*mult
            target=min(base,nav*MAX_SINGLE,max(0.0,nav*MAX_EXPOSURE-planned_mv),max(0.0,cash_room-reserved))
            s=share_size(target,limit,avgvol)
            while s>0 and limit*s*(1+BUY_FEE)>cash_room-reserved+1e-6:
                s=s-1000 if s>=2000 else s-1
            if s<=0: continue
            gross=limit*s; reserve=gross*(1+BUY_FEE)
            if gross>nav*MAX_SINGLE+1e-6: continue
            self.pending_buys.setdefault(next_date,[]).append(BuyOrder(date,next_date,strat,code,name,s,limit,target,'R10_FIXED_T1_LIMIT'))
            reserved+=reserve; planned_mv+=gross; held.add(code)

    def run(self,dates):
        for idx,date in enumerate(dates):
            next_date=dates[idx+1] if idx+1<len(dates) else None
            self.execute(date,next_date)
            bars=self.raw_by_date.get(date)
            mv=self.marks(date); nav=self.cash+mv; self.peak=max(self.peak,nav); dd=nav/self.peak-1.0
            self.nav_rows.append({'date':date,'cash':self.cash,'market_value':mv,'nav':nav,'peak_nav':self.peak,'drawdown':dd,'holdings':len(self.positions),'exposure':mv/nav if nav>0 else 0.0})
            for p in self.positions.values():
                if p.entry_date!=date: p.hold+=1
            if next_date is not None: self.decide(idx,date,next_date,nav,dd)
        return float(self.nav_rows[-1]['nav'])


def main():
    raw=load_raw()
    instp=CACHE/'institutional_2020_2025.parquet'
    if not instp.exists(): raise RuntimeError('missing institutional cache')
    inst=pd.read_parquet(instp)
    r7=build_r7_features(raw)
    adv=raw[['date','code','volume']].copy(); adv['avgvol20']=adv.groupby('code').volume.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    r7=r7.merge(adv[['date','code','avgvol20']],on=['date','code'],how='left',validate='one_to_one')
    r05=build_r05_features(raw,inst)
    dates=sorted(int(x) for x in raw.date.unique() if EVAL_START<=int(x)<=EVAL_END)
    sim=R10(raw,r7,r05); end=sim.run(dates)
    nav=pd.DataFrame(sim.nav_rows); trades=pd.DataFrame(sim.trades); orders=pd.DataFrame(sim.orders)
    years=(pd.to_datetime(nav.date.astype(str)).iloc[-1]-pd.to_datetime(nav.date.astype(str)).iloc[0]).days/365.25
    cagr=(end/INITIAL)**(1/years)-1 if years>0 else 0.0
    mdd=float(nav.drawdown.min())
    summary={
        'strategy':'AlphaPilot R10-MAX 0.5% Clean Retest',
        'status':'PASS' if mdd>-0.20 else 'REJECT_DD_GATE',
        'initial_nav':INITIAL,'ending_nav':end,'total_return':end/INITIAL-1.0,'cagr':cagr,'max_drawdown':mdd,
        'annual_returns':annual_returns(nav),'completed_trades':int(len(trades)),'orders':int(len(orders)),
        'strict_dd_lt_20_pass':bool(mdd>-0.20),'max_positions':int(nav.holdings.max()),'max_exposure':float(nav.exposure.max()),
        'contract':{'decision':'T close only','buy':'T+1 fixed limit; no chase','sell':'T+1 open with 0.5% adverse slippage','commission_each_side':BUY_FEE,'sell_tax':SELL_TAX,'single_stock_cap':MAX_SINGLE,'portfolio_exposure_cap':MAX_EXPOSURE,'max_positions':MAX_POSITIONS,'adv20_share_cap':ADV_CAP},
        'fidelity_note':'R10 locked PDF/XLSX rules implemented from source specification. R0.5 institutional strength factors are ranked cross-sectionally from official net-share data; same-day R7/R0.5 competition uses deterministic source-score ordering because the source document does not specify a cross-strategy normalization formula.'
    }
    (OUT/'r10_clean_retest_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    nav.to_csv(OUT/'r10_clean_retest_nav.csv',index=False)
    trades.to_csv(OUT/'r10_clean_retest_trades.csv',index=False)
    orders.to_csv(OUT/'r10_clean_retest_orders.csv',index=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
