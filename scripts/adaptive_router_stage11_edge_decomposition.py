#!/usr/bin/env python3
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
REF=ROOT/'reference_bundle'; OUT=ROOT/'adaptive_router_stage11'; OUT.mkdir(exist_ok=True)
DISCOVERY=[2016,2017,2018]; OOS=[2019,2020]
EDGES=[
 ('E1','vol20','DISPERSION_HIGH',-1),('E2','vol20','BREADTH_HIGH',-1),
 ('E3','range20','DISPERSION_HIGH',-1),('E4','vol20','MARKET_MOM_HIGH',-1),
 ('E5','range20','BREADTH_HIGH',-1),('E6','range20','MARKET_MOM_HIGH',-1),
 ('E7','break60','BREADTH_HIGH',1),('E8','break60','DISPERSION_HIGH',1)]

def norm_code(s): return s.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)

def load_ohlcv():
    parts=[]
    for y in range(2015,2020):
        zp=REF/'ohlcv_raw'/f'yearly_{y}.zip'
        with zipfile.ZipFile(zp) as z:
            member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
            with z.open(member) as f: parts.append(pd.read_csv(f,dtype={'code':str},low_memory=False))
    parts.append(pd.read_parquet(REF/'formal_2020'/'ohlcv_2020.parquet'))
    d=pd.concat(parts,ignore_index=True); d.columns=[str(c).strip().lower() for c in d.columns]
    d['code']=norm_code(d.code)
    if pd.api.types.is_datetime64_any_dtype(d.date): d['date']=d.date.dt.strftime('%Y%m%d').astype(int)
    else: d['date']=pd.to_numeric(d.date.astype(str).str.replace('-','',regex=False),errors='coerce').astype('Int64')
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['date','code','open','high','low','close','volume']).copy(); d['date']=d.date.astype(int)
    d=d[d.code.str.fullmatch(r'[1-9]\d{3}')].copy()
    return d.sort_values(['code','date']).drop_duplicates(['code','date'],keep='last')

def prepare():
    d=load_ohlcv(); g=d.groupby('code',group_keys=False)
    d['prev_close']=g.close.shift(1); d['bridge']=d.prev_close/d.open
    d['action_flag']=d.prev_close.notna() & ((d.bridge<0.82)|(d.bridge>1.18)); d.loc[~d.action_flag,'bridge']=1.0
    d.loc[(d.bridge<0.1)|(d.bridge>10),'bridge']=1.0; d['adj_factor']=g.bridge.cumprod()
    for c in ['open','high','low','close']: d['adj_'+c]=d[c]*d.adj_factor
    g=d.groupby('code',group_keys=False)
    d['amount']=d.close*d.volume; d['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean())
    d['liq_pct']=d.groupby('date')['amt20'].rank(pct=True)
    d['r20']=g.adj_close.transform(lambda s:s.pct_change(20,fill_method=None)); d['r60']=g.adj_close.transform(lambda s:s.pct_change(60,fill_method=None))
    d['ma60']=g.adj_close.transform(lambda s:s.rolling(60,min_periods=60).mean()); d['dist_ma60']=d.adj_close/d.ma60-1
    d['prior60']=g.adj_close.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max()); d['break60']=d.adj_close/d.prior60-1
    ret1=g.adj_close.transform(lambda s:s.pct_change(fill_method=None)); d['vol20']=ret1.groupby(d.code).transform(lambda s:s.rolling(20,min_periods=20).std())
    hi=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).max()); lo=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).min())
    d['range20']=(hi-lo)/d.ma60.replace(0,np.nan)
    for f in [5,10,20,40]: d[f'fwd{f}']=g.adj_close.transform(lambda s,f=f:s.shift(-f)/s-1)
    d['action_recent']=g.action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool)
    d['eligible']=(d.liq_pct>=0.35)&~d.action_recent
    cs=d[d.eligible].groupby('date').agg(breadth=('dist_ma60',lambda s:float((s>0).mean())),market_mom=('r20','median'),dispersion=('r20','std')).sort_index()
    for c in ['breadth','market_mom','dispersion']:
        cs[c+'_q70']=cs[c].shift(1).rolling(252,min_periods=120).quantile(.70)
    ctx=pd.DataFrame(index=cs.index)
    ctx['BREADTH_HIGH']=cs.breadth>=cs.breadth_q70
    ctx['MARKET_MOM_HIGH']=cs.market_mom>=cs.market_mom_q70
    ctx['DISPERSION_HIGH']=cs.dispersion>=cs.dispersion_q70
    return d,ctx

def decompose(d,ctx):
    rows=[]
    for eid,feat,cname,direction in EDGES:
        on=ctx[cname]
        for y in DISCOVERY+OOS:
            yy=d[(d.date//10000==y)&d.eligible].copy()
            yy=yy[yy.date.map(on).fillna(False)]
            for dt,x in yy.groupby('date'):
                x=x[[feat,'fwd5','fwd10','fwd20','fwd40']].dropna(subset=[feat,'fwd20'])
                if len(x)<40: continue
                pct=x[feat].rank(pct=True)
                # 'favored' is the side implied by Stage9 direction
                favored=(pct<=0.20) if direction<0 else (pct>=0.80)
                opposed=(pct>=0.80) if direction<0 else (pct<=0.20)
                for horizon in [5,10,20,40]:
                    f=f'fwd{horizon}'
                    a=x.loc[favored,f].dropna(); b=x.loc[opposed,f].dropna()
                    if len(a)<5 or len(b)<5: continue
                    rows.append({'edge_id':eid,'feature':feat,'context':cname,'direction':direction,'year':y,'date':dt,'horizon':horizon,
                                 'fav_mean':float(a.mean()),'fav_median':float(a.median()),'fav_win':float((a>0).mean()),
                                 'opp_mean':float(b.mean()),'opp_median':float(b.median()),'opp_win':float((b>0).mean()),
                                 'spread_mean':float(a.mean()-b.mean()),'fav_n':len(a),'opp_n':len(b)})
    return pd.DataFrame(rows)

def main():
    d,ctx=prepare(); z=decompose(d,ctx)
    if z.empty: raise SystemExit('no decomposition rows')
    z.to_csv(OUT/'daily_edge_decomposition.csv',index=False)
    agg=z.groupby(['edge_id','feature','context','direction','year','horizon']).agg(
        fav_mean=('fav_mean','mean'),fav_median=('fav_median','median'),fav_win=('fav_win','mean'),
        opp_mean=('opp_mean','mean'),opp_median=('opp_median','median'),opp_win=('opp_win','mean'),
        spread_mean=('spread_mean','mean'),days=('date','nunique')).reset_index()
    def classify(r):
        if r.fav_mean>0 and r.spread_mean>0: return 'LONG_CANDIDATE'
        if r.fav_mean<=0 and r.opp_mean<r.fav_mean: return 'RELATIVE_ONLY'
        if r.fav_mean>0 and r.spread_mean<=0: return 'NO_RANK_EDGE'
        return 'AVOID_FILTER'
    agg['classification']=agg.apply(classify,axis=1)
    agg.to_csv(OUT/'edge_year_horizon_summary.csv',index=False)
    h20=agg[agg.horizon==20].copy()
    pivot=h20.pivot_table(index=['edge_id','feature','context','direction'],columns='year',values=['fav_mean','spread_mean','fav_win'],aggfunc='first')
    pivot.columns=[f'{a}_{b}' for a,b in pivot.columns]; pivot=pivot.reset_index()
    def verdict(r):
        years=[2016,2017,2018,2019,2020]
        fav=[r.get(f'fav_mean_{y}',np.nan) for y in years]
        spr=[r.get(f'spread_mean_{y}',np.nan) for y in years]
        long_ok=sum(pd.notna(v) and v>0 for v in fav)
        rank_ok=sum(pd.notna(v) and v>0 for v in spr)
        if long_ok>=4 and rank_ok>=4: return 'PROMISING_LONG_FILTER'
        if rank_ok>=4 and long_ok<4: return 'RELATIVE_RANK_ONLY'
        if long_ok<=2: return 'AVOID_AS_LONG'
        return 'UNSTABLE'
    pivot['verdict']=pivot.apply(verdict,axis=1)
    pivot.to_csv(OUT/'edge_final_verdict.csv',index=False)
    manifest={'stage':'11_edge_decomposition','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,
              'transaction_parameters_retuned':False,'stage10_failure_used_to_change_thresholds':False,'context_thresholds_past_only':True,
              'scale_invariant_liquidity_rank':True,'edge_count':len(EDGES),'horizons':[5,10,20,40],
              'pre2020_corp_action_method':'inferred_bridge_approximate'}
    json.dump(manifest,open(OUT/'stage11_manifest.json','w'),indent=2)
    print('STAGE11_COMPLETE'); print(json.dumps(manifest,indent=2)); print('\nFINAL VERDICTS'); print(pivot.to_string(index=False))

if __name__=='__main__': main()
