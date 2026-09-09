#!/usr/bin/env python3
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
REF=ROOT/'reference_bundle'; OUT=ROOT/'adaptive_router_stage9'; OUT.mkdir(exist_ok=True)
DISCOVERY_YEARS=[2016,2017,2018]; OOS1=2019; OOS2=2020
FWD=20
FEATURES=['r20','r60','vol20','range20','dist_ma60','break60','inst_ratio']
CONTEXTS=['BREADTH_HIGH','BREADTH_LOW','DISPERSION_HIGH','DISPERSION_LOW','MARKET_MOM_HIGH','MARKET_MOM_LOW']

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

def load_inst():
    parts=[]
    for y in range(2015,2020):
        p=REF/'institutional_2015_2019'/f'institutional_{y}.csv.gz'; parts.append(pd.read_csv(p,dtype={'code':str},low_memory=False))
    x=pd.read_parquet(REF/'formal_2020'/'institutional_2020_2025.parquet')
    yr=x.date.dt.year if pd.api.types.is_datetime64_any_dtype(x.date) else pd.to_datetime(x.date).dt.year
    parts.append(x[yr==2020].copy())
    d=pd.concat(parts,ignore_index=True); d.columns=[str(c).strip().lower() for c in d.columns]; d['code']=norm_code(d.code)
    d['date']=pd.to_datetime(d.date).dt.strftime('%Y%m%d').astype(int)
    for c in ['foreign_net','trust_net']:
        if c not in d: d[c]=0.0
        d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    return d[['date','code','foreign_net','trust_net']].drop_duplicates(['date','code'],keep='last')

def prepare():
    d=load_ohlcv().merge(load_inst(),on=['date','code'],how='left')
    d[['foreign_net','trust_net']]=d[['foreign_net','trust_net']].fillna(0)
    g=d.groupby('code',group_keys=False)
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
    d['inst5']=g.apply(lambda x:(x.foreign_net+x.trust_net).rolling(5,min_periods=3).sum(),include_groups=False).reset_index(level=0,drop=True)
    vma20=g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean()); d['inst_ratio']=d.inst5/vma20.replace(0,np.nan)
    d['fwd20']=g.adj_close.transform(lambda s:s.shift(-FWD)/s-1)
    d['action_recent']=g.action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool)
    d['eligible']=(d.liq_pct>=0.35)&(d.close>0)&~d.action_recent
    # market structure from same-day cross-section only; thresholds are rolling past-only percentiles
    cs=d[d.eligible].groupby('date').agg(
        breadth=('dist_ma60',lambda s:float((s>0).mean())),
        market_mom=('r20','median'),
        dispersion=('r20','std')
    ).sort_index()
    for c in ['breadth','market_mom','dispersion']:
        cs[c+'_q70']=cs[c].shift(1).rolling(252,min_periods=120).quantile(.70)
        cs[c+'_q30']=cs[c].shift(1).rolling(252,min_periods=120).quantile(.30)
    ctx=pd.DataFrame(index=cs.index)
    ctx['BREADTH_HIGH']=cs.breadth>=cs.breadth_q70; ctx['BREADTH_LOW']=cs.breadth<=cs.breadth_q30
    ctx['DISPERSION_HIGH']=cs.dispersion>=cs.dispersion_q70; ctx['DISPERSION_LOW']=cs.dispersion<=cs.dispersion_q30
    ctx['MARKET_MOM_HIGH']=cs.market_mom>=cs.market_mom_q70; ctx['MARKET_MOM_LOW']=cs.market_mom<=cs.market_mom_q30
    return d,cs,ctx

def daily_ic(df,feature):
    vals=[]
    for dt,x in df.groupby('date'):
        x=x[['fwd20',feature]].dropna()
        if len(x)<40: continue
        vals.append((dt,x[feature].corr(x.fwd20,method='spearman')))
    return pd.DataFrame(vals,columns=['date','ic'])

def main():
    d,cs,ctx=prepare(); detail=[]; summary=[]
    for feat in FEATURES:
        ic=daily_ic(d[d.eligible],feat)
        if ic.empty: continue
        ic['year']=ic.date.astype(str).str[:4].astype(int)
        for cname in CONTEXTS:
            mask=ctx[cname].rename('on').reset_index().rename(columns={'index':'date'})
            z=ic.merge(mask,on='date',how='left'); z=z[z.on.fillna(False)]
            for y in DISCOVERY_YEARS+[OOS1,OOS2]:
                yy=z[z.year==y]
                med=float(yy.ic.median()) if len(yy) else np.nan; mean=float(yy.ic.mean()) if len(yy) else np.nan
                detail.append({'feature':feat,'context':cname,'year':y,'median_ic':med,'mean_ic':mean,'days':len(yy)})
            pre=pd.DataFrame([r for r in detail if r['feature']==feat and r['context']==cname and r['year'] in DISCOVERY_YEARS])
            if len(pre)==3 and pre.median_ic.notna().all():
                signs=np.sign(pre.median_ic.values); same=bool((signs==signs[0]).all() and signs[0]!=0)
                minabs=float(np.abs(pre.median_ic).min()); avg=float(pre.median_ic.mean()); direction=int(np.sign(avg)) if avg!=0 else 0
                summary.append({'feature':feat,'context':cname,'same_sign_3y':same,'min_abs_year_ic':minabs,'avg_discovery_ic':avg,'direction':direction})
    det=pd.DataFrame(detail); summ=pd.DataFrame(summary)
    robust=summ[(summ.same_sign_3y)&(summ.min_abs_year_ic>=0.02)].copy().sort_values('min_abs_year_ic',ascending=False)
    # freeze before OOS: no OOS fields used in selection
    evalrows=[]
    for _,r in robust.iterrows():
        for y in [OOS1,OOS2]:
            q=det[(det.feature==r.feature)&(det.context==r.context)&(det.year==y)]
            if len(q):
                med=float(q.median_ic.iloc[0]); evalrows.append({'feature':r.feature,'context':r.context,'direction':int(r.direction),'year':y,'median_ic':med,'same_direction':bool(np.sign(med)==r.direction) if pd.notna(med) and med!=0 else False,'days':int(q.days.iloc[0])})
    ev=pd.DataFrame(evalrows)
    robust.to_csv(OUT/'robust_context_edges_frozen_pre2019.csv',index=False)
    det.to_csv(OUT/'context_factor_year_detail.csv',index=False)
    ev.to_csv(OUT/'oos_context_edge_validation.csv',index=False)
    cs.reset_index().to_csv(OUT/'daily_market_structure.csv',index=False)
    oos1_pass=0; oos2_pass=0
    if len(ev):
        p=ev[ev.year==OOS1]; oos1_pass=int(p.same_direction.sum())
        q=ev.merge(p[['feature','context','same_direction']],on=['feature','context'],suffixes=('','_oos1'))
        q=q[(q.year==OOS2)&(q.same_direction_oos1)]
        oos2_pass=int(q.same_direction.sum())
    manifest={'stage':'9_context_conditioned_edge','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'oos1_2019_used_for_discovery':False,'oos2_2020_used_for_discovery':False,'context_thresholds_past_only':True,'scale_invariant_liquidity_rank':True,'feature_count':len(FEATURES),'context_count':len(CONTEXTS),'robust_pre2019_edges':int(len(robust)),'oos1_same_direction':oos1_pass,'oos2_after_oos1_same_direction':oos2_pass,'pre2020_corp_action_method':'inferred_bridge_approximate'}
    json.dump(manifest,open(OUT/'stage9_manifest.json','w'),indent=2)
    print('ADAPTIVE_ROUTER_STAGE9_COMPLETE'); print(json.dumps(manifest,indent=2)); print('\nFROZEN CONTEXT EDGES'); print(robust.to_string(index=False) if len(robust) else 'NONE'); print('\nOOS'); print(ev.to_string(index=False) if len(ev) else 'NONE')

if __name__=='__main__': main()
