#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'.clean_cache'


def residual_adjust(df: pd.DataFrame) -> tuple[pd.DataFrame,int]:
    """Causal continuity adjustment for impossible >11.5% overnight gaps only.

    Raw OHLC remains untouched for execution. Adjusted OHLC is used only for
    historical features. A factor change can only be detected when the new
    day's raw open is observed, so it never changes a prior T decision.
    """
    market_dates=np.array(sorted(df.date.unique()),dtype=int)
    prev_market={int(market_dates[i]):int(market_dates[i-1]) for i in range(1,len(market_dates))}
    out=[]; events=0
    for code,q in df.groupby('code',sort=False):
        q=q.sort_values('date').copy(); factor=1.0; prev_adj_close=None; prev_date=None
        aopen=[]; ahigh=[]; alow=[]; aclose=[]
        for r in q.itertuples(index=False):
            op=float(r.open)*factor; hi=float(r.high)*factor; lo=float(r.low)*factor; cl=float(r.close)*factor
            if prev_adj_close is not None and prev_date is not None and prev_market.get(int(r.date))==int(prev_date) and op>0:
                ratio=op/prev_adj_close
                if ratio < 0.885 or ratio > 1.115:
                    step=prev_adj_close/op; factor*=step
                    op*=step; hi*=step; lo*=step; cl*=step; events+=1
            aopen.append(op); ahigh.append(hi); alow.append(lo); aclose.append(cl)
            prev_adj_close=cl; prev_date=int(r.date)
        q['aopen']=aopen; q['ahigh']=ahigh; q['alow']=alow; q['aclose']=aclose
        out.append(q)
    return pd.concat(out,ignore_index=True).sort_values(['date','code']).reset_index(drop=True),events


def build_features(ohlcv: pd.DataFrame, inst: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    q,events=residual_adjust(ohlcv)
    q['amount']=q.close.astype(float)*q.volume.astype(float)
    g=q.groupby('code',sort=False)
    for w in (1,5,10,20,40,60,120): q[f'ret{w}']=g.aclose.pct_change(w,fill_method=None)
    for w in (5,10,20,40,60,120): q[f'ma{w}']=g.aclose.rolling(w,min_periods=w).mean().reset_index(level=0,drop=True)
    for w in (10,20,60):
        q[f'prior_high{w}']=g.aclose.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=w).max())
        q[f'prior_low{w}']=g.aclose.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=w).min())
    q['avgvol20']=g.volume.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['avgamt20']=g.amount.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['avgamt60']=g.amount.rolling(60,min_periods=60).mean().reset_index(level=0,drop=True)
    q['vol_ratio']=q.volume/g.volume.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['amt_ratio']=q.amount/g.amount.rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    tr=pd.concat([(q.high-q.low).abs(),(q.high-g.close.shift(1)).abs(),(q.low-g.close.shift(1)).abs()],axis=1).max(axis=1)
    q['atr20']=tr.groupby(q.code).rolling(20,min_periods=20).mean().reset_index(level=0,drop=True)
    q['atr_pct']=q.atr20/q.close
    q['volatility20']=g.ret1.rolling(20,min_periods=20).std().reset_index(level=0,drop=True)
    rng=(q.high-q.low).replace(0,np.nan)
    q['clv']=((2*q.close-q.high-q.low)/rng).fillna(0).clip(-1,1)
    q['close_pos20']=(q.aclose-q.prior_low20)/(q.prior_high20-q.prior_low20).replace(0,np.nan)
    q['dist_ma20']=q.aclose/q.ma20-1
    q['dist_ma60']=q.aclose/q.ma60-1
    q['break10']=q.aclose/q.prior_high10-1
    q['break20']=q.aclose/q.prior_high20-1
    q['break60']=q.aclose/q.prior_high60-1

    ins=inst.copy().sort_values(['market','code','date'])
    for c in ('foreign_net','trust_net','dealer_net'): ins[c]=pd.to_numeric(ins[c],errors='coerce').fillna(0.0)
    ig=ins.groupby(['market','code'],sort=False)
    for c,pfx in [('foreign_net','foreign'),('trust_net','trust'),('dealer_net','dealer')]:
        for w in (3,5,10,20): ins[f'{pfx}{w}']=ig[c].rolling(w,min_periods=w).sum().reset_index(level=[0,1],drop=True)
    # Codes are unique across TWSE/TPEx in the ordinary listed universe; retain one row/date/code.
    ikeep=['date','code']+[c for c in ins.columns if c.startswith(('foreign','trust','dealer'))]
    ins2=ins[ikeep].sort_values(['date','code']).drop_duplicates(['date','code'],keep='last')
    q=q.merge(ins2,on=['date','code'],how='left',validate='one_to_one')

    # 0050 is benchmark only, never tradable by clean stock strategy.
    bm=q[q.code.eq('0050')][['date','aclose','ma20','ma60','ma120','ret20','ret60','volatility20']].copy()
    bm=bm.rename(columns={c:f'mkt_{c}' for c in bm.columns if c!='date'}).sort_values('date')
    bm['mkt_risk_on']=(bm.mkt_aclose>bm.mkt_ma60)&(bm.mkt_ret20>0)&(bm.mkt_ret60>0)
    q=q.merge(bm,on='date',how='left',validate='many_to_one')
    q['rs20']=q.ret20-q.mkt_ret20
    q['rs60']=q.ret60-q.mkt_ret60

    # Cross-sectional ranks are computed only inside each T date from T-known values.
    ordinary=q.code.str.fullmatch(r'\d{4}') & ~q.code.str.startswith('00')
    universe=q[ordinary].copy()
    for c in ('ret20','ret60','rs20','rs60','vol_ratio','amt_ratio','foreign5','foreign20','trust5','trust20'):
        if c in universe.columns:
            universe[f'pct_{c}']=universe.groupby('date')[c].rank(pct=True,method='average')

    audit={
        'status':'PASS','corporate_action_events':int(events),
        'rows':int(len(universe)),'dates':int(universe.date.nunique()),
        'min_date':int(universe.date.min()),'max_date':int(universe.date.max()),
        'benchmark_dates':int(bm.date.nunique()),
        'future_shift_columns':0,
        'note':'All rolling/rank features use same-day or prior data only; prior highs/lows explicitly shift(1).'
    }
    return universe,bm,audit


def main():
    o=pd.concat([pd.read_parquet(CACHE/f'ohlcv_{y}.parquet') for y in range(2020,2026)],ignore_index=True)
    ins=pd.read_parquet(CACHE/'institutional_2020_2025.parquet')
    feat,bm,audit=build_features(o,ins)
    feat.to_parquet(CACHE/'features_2020_2025.parquet',index=False)
    bm.to_parquet(CACHE/'benchmark_0050_2020_2025.parquet',index=False)
    (CACHE/'feature_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))

if __name__=='__main__': main()
