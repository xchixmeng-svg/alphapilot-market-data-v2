#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Mapping
import math
import numpy as np
import pandas as pd

from clean_event_loop import BuyIntent, SellIntent, DecisionContext, floor_tick


@dataclass(frozen=True)
class Params:
    family: str = 'MOM_RS_FLOW'
    min_avgamt20: float = 30_000_000.0
    min_price: float = 10.0
    max_price: float = 3000.0
    score_threshold: float = 0.68
    limit_discount: float = 0.005
    max_position_pct: float = 0.18
    max_slots_strong: int = 5
    max_slots_neutral: int = 3
    max_slots_weak: int = 0
    exit_ma: int = 20
    trailing_pct: float = 0.10
    max_hold: int = 80
    momentum_decay_pct: float = 0.40


def _valid(v): return v is not None and np.isfinite(v)


def build_feature_store(features: pd.DataFrame) -> Dict[str,pd.DataFrame]:
    return {str(int(d)):q.set_index('code',drop=False) for d,q in features.groupby('date',sort=True)}


def size_shares(nav: float, price: float, target_pct: float) -> int:
    """Board-lot first; 100-share odd lots only when one board lot breaches 20% NAV."""
    if nav<=0 or price<=0: return 0
    hard_cap=nav*0.20
    target=min(nav*target_pct,hard_cap)
    one_lot=price*1000
    if one_lot <= hard_cap + 1e-9:
        lots=max(1,int(math.floor(target/one_lot)))
        while lots>0 and lots*one_lot>hard_cap+1e-9: lots-=1
        return max(0,lots*1000)
    shares=int(math.floor(target/price/100.0))*100
    return max(0,shares)


class ResearchStrategy:
    def __init__(self, features_or_store, params: Params):
        self.params=params
        self.by_date=features_or_store if isinstance(features_or_store,dict) else build_feature_store(features_or_store)
        self.peak: Dict[str,float]={}
        self.age: Dict[str,int]={}

    def _row(self,date: str,code: str):
        q=self.by_date.get(date)
        if q is None or code not in q.index: return None
        r=q.loc[code]
        return r.iloc[-1] if isinstance(r,pd.DataFrame) else r

    def _regime(self, sample) -> tuple[str,int]:
        p=self.params
        if sample is None: return 'WEAK',p.max_slots_weak
        close=float(sample.mkt_aclose) if _valid(sample.mkt_aclose) else np.nan
        ma60=float(sample.mkt_ma60) if _valid(sample.mkt_ma60) else np.nan
        ma120=float(sample.mkt_ma120) if _valid(sample.mkt_ma120) else np.nan
        r20=float(sample.mkt_ret20) if _valid(sample.mkt_ret20) else np.nan
        r60=float(sample.mkt_ret60) if _valid(sample.mkt_ret60) else np.nan
        if all(np.isfinite(x) for x in (close,ma60,ma120,r20,r60)) and close>ma60>ma120 and r20>0 and r60>0:
            return 'STRONG',p.max_slots_strong
        if all(np.isfinite(x) for x in (close,ma60,r20)) and close>ma60 and r20>-0.03:
            return 'NEUTRAL',p.max_slots_neutral
        return 'WEAK',p.max_slots_weak

    def _score(self,q: pd.DataFrame) -> pd.Series:
        p=self.params
        if p.family=='MOM_RS_FLOW':
            return 0.24*q.pct_rs20 + 0.20*q.pct_rs60 + 0.18*q.pct_ret20 + 0.12*q.pct_ret60 + 0.10*q.pct_foreign5 + 0.08*q.pct_trust5 + 0.08*q.pct_amt_ratio
        if p.family=='BREAK_FLOW':
            br=(q.break20.clip(-.10,.10)+.10)/.20
            return 0.24*q.pct_rs20 + 0.18*q.pct_ret20 + 0.18*br.clip(0,1) + 0.14*q.pct_amt_ratio + 0.12*q.pct_foreign5 + 0.08*q.pct_trust5 + 0.06*q.clv.add(1).div(2)
        if p.family=='PULLBACK_RS':
            pull=(1-(q.dist_ma20.abs()/0.10)).clip(0,1)
            return 0.28*q.pct_rs60 + 0.18*q.pct_rs20 + 0.16*q.pct_ret60 + 0.16*pull + 0.10*q.pct_foreign20 + 0.06*q.pct_trust20 + 0.06*q.pct_amt_ratio
        raise ValueError(p.family)

    def _candidates(self,date: str,held: set[str]) -> pd.DataFrame:
        p=self.params; q=self.by_date.get(date)
        if q is None or q.empty: return pd.DataFrame()
        x=q
        required=['close','avgamt20','ma20','ma60','ma120','rs20','rs60','pct_rs20','pct_rs60','pct_ret20','pct_ret60','pct_amt_ratio']
        for c in required:
            if c not in x.columns: return pd.DataFrame()
        mask=(x.close.between(p.min_price,p.max_price))&(x.avgamt20>=p.min_avgamt20)&(x.close>x.ma60)&(x.ma60>x.ma120)
        mask &= x.rs20.notna() & x.rs60.notna()
        if p.family=='BREAK_FLOW': mask &= (x.aclose>=x.prior_high20*.985)
        elif p.family=='PULLBACK_RS': mask &= x.dist_ma20.between(-.04,.08)
        z=x[mask & ~x.code.isin(held)].copy()
        if z.empty:return z
        z['score']=self._score(z)
        return z[z.score>=p.score_threshold].sort_values(['score','code'],ascending=[False,True])

    def decide(self,ctx: DecisionContext) -> tuple[Sequence[SellIntent],Sequence[BuyIntent]]:
        p=self.params; date=str(int(ctx.date.replace('-',''))) if '-' in ctx.date else ctx.date
        q=self.by_date.get(date)
        sample=q.iloc[0] if q is not None and not q.empty else None
        regime,max_slots=self._regime(sample)
        sells=[]
        for code,pos in ctx.positions.items():
            r=self._row(date,code)
            if r is None: continue
            close=float(r.close); self.peak[code]=max(self.peak.get(code,pos.entry_price),close); self.age[code]=self.age.get(code,0)+1
            ma_col=f'ma{p.exit_ma}'; ma=float(r[ma_col]) if ma_col in r and _valid(r[ma_col]) else np.nan
            rs_pct=float(r.pct_rs20) if 'pct_rs20' in r and _valid(r.pct_rs20) else 0.5
            reason=None
            if regime=='WEAK': reason='MARKET_WEAK'
            elif np.isfinite(ma) and close<ma: reason=f'CLOSE_BELOW_MA{p.exit_ma}'
            elif close <= self.peak[code]*(1-p.trailing_pct): reason='CLOSE_TRAIL'
            elif rs_pct < p.momentum_decay_pct: reason='MOMENTUM_DECAY'
            elif self.age[code]>=p.max_hold: reason='TIME'
            if reason: sells.append(SellIntent(code,full_exit=True,reason=reason))
        held=set(ctx.positions)
        capacity=max(0,max_slots-len(held))
        if capacity<=0: return sells,[]
        c=self._candidates(date,held)
        buys=[]
        for r in c.head(capacity).itertuples(index=False):
            limit=floor_tick(float(r.close)*(1-p.limit_discount))
            shares=size_shares(ctx.nav,limit,p.max_position_pct)
            if shares<=0: continue
            buys.append(BuyIntent(str(r.code),shares,limit,f'{p.family}_SCORE'))
        return sells,buys
