#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
REF=ROOT/'reference_bundle'
OUT=ROOT/'adaptive_router_stage8'; OUT.mkdir(exist_ok=True)
START_CAPITAL=1_300_000.0
FEE=0.000855; SELL_TAX=0.003; BUY_SLIP=0.005; SELL_SLIP=0.005
MAX_POS=4; MAX_STOCK=0.25; TOTAL_EXPOSURE=0.90
DISCOVERY=(20160101,20181231); OOS1=(20190101,20191231); OOS2=(20200101,20201231)
FEATURES=['r5','r20','r60','r120','ma20_dist','ma60_dist','ma120_dist','break20','break60','vol_ratio_log','range20','dd60','vol20','amount_rel','inst_ratio']


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
    return d.sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)

def load_inst():
    parts=[]
    for y in range(2015,2020):
        parts.append(pd.read_csv(REF/'institutional_2015_2019'/f'institutional_{y}.csv.gz',dtype={'code':str},low_memory=False))
    x=pd.read_parquet(REF/'formal_2020'/'institutional_2020_2025.parquet')
    dt=pd.to_datetime(x.date); parts.append(x[dt.dt.year==2020].copy())
    d=pd.concat(parts,ignore_index=True); d.columns=[str(c).strip().lower() for c in d.columns]
    d['code']=norm_code(d.code); d['date']=pd.to_datetime(d.date).dt.strftime('%Y%m%d').astype(int)
    for c in ['foreign_net','trust_net','dealer_net','total_net']:
        if c not in d: d[c]=0.0
        d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0.0)
    return d[['date','code','foreign_net','trust_net','dealer_net','total_net']].drop_duplicates(['date','code'],keep='last')

def prepare():
    d=load_ohlcv().merge(load_inst(),on=['date','code'],how='left')
    for c in ['foreign_net','trust_net','dealer_net','total_net']: d[c]=d[c].fillna(0.0)
    g=d.groupby('code',group_keys=False)
    d['prev_close']=g.close.shift(1); d['bridge']=d.prev_close/d.open
    d['action_flag']=d.prev_close.notna() & ((d.bridge<0.82)|(d.bridge>1.18)); d.loc[~d.action_flag,'bridge']=1.0
    d.loc[(d.bridge<0.10)|(d.bridge>10.0),'bridge']=1.0
    d['adj_factor']=g.bridge.cumprod()
    for c in ['open','high','low','close']: d['adj_'+c]=d[c]*d.adj_factor
    g=d.groupby('code',group_keys=False)
    d['ret1']=g.adj_close.pct_change(fill_method=None)
    for n in [5,20,60,120]: d[f'r{n}']=g.adj_close.pct_change(n,fill_method=None)
    for n in [20,60,120]: d[f'ma{n}']=g.adj_close.transform(lambda s,n=n:s.rolling(n,min_periods=n).mean())
    for n in [20,60]: d[f'prior{n}']=g.adj_close.transform(lambda s,n=n:s.shift(1).rolling(n,min_periods=n).max())
    d['ma20_dist']=d.adj_close/d.ma20-1; d['ma60_dist']=d.adj_close/d.ma60-1; d['ma120_dist']=d.adj_close/d.ma120-1
    d['break20']=d.adj_close/d.prior20-1; d['break60']=d.adj_close/d.prior60-1
    d['vma20']=g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean()); d['vol_ratio']=d.volume/d.vma20.replace(0,np.nan); d['vol_ratio_log']=np.log(d.vol_ratio.clip(lower=0.05))
    d['range20']=g.adj_close.transform(lambda s:(s.shift(1).rolling(20,min_periods=20).max()-s.shift(1).rolling(20,min_periods=20).min())/s.rolling(20,min_periods=20).mean())
    d['dd60']=d.adj_close/d.prior60-1
    d['vol20']=g.ret1.transform(lambda s:s.rolling(20,min_periods=20).std())
    d['amount']=d.close*d.volume; d['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean()); d['amount_rel']=d.amount/d.amt20.replace(0,np.nan)-1
    d['inst_net']=d.foreign_net+d.trust_net; d['inst5']=g.inst_net.transform(lambda s:s.rolling(5,min_periods=3).sum()); d['inst_ratio']=d.inst5/d.vma20.replace(0,np.nan)
    d['action_recent']=g.action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool)
    d['liq_pct']=d.groupby('date')['amt20'].rank(pct=True)
    d['eligible']=(d.liq_pct>=0.30)&d.ma120.notna()&~d.action_recent&(d.close>0)
    for c in FEATURES: d[c+'_pct']=d.groupby('date')[c].rank(pct=True)
    d['fwd20']=g.adj_close.shift(-20)/d.adj_close-1
    d['fwd20_date']=g.date.shift(-20)
    d['fwd20_pct']=d.groupby('date')['fwd20'].rank(pct=True)
    breadth=d.groupby('date').apply(lambda x: float(((x.adj_close>x.ma60)&x.ma60.notna()).mean()),include_groups=False).rename('breadth_ma60')
    d=d.merge(breadth,on='date',how='left')
    return d.sort_values(['date','code']).reset_index(drop=True)

def daily_ic(frame, feature):
    out=[]
    for dt,x in frame.groupby('date'):
        z=x[[feature+'_pct','fwd20_pct']].dropna()
        if len(z)<200: continue
        c=z[feature+'_pct'].corr(z.fwd20_pct)
        if pd.notna(c): out.append((int(dt),float(c)))
    return out

def discover_factors(d):
    rows=[]
    for feat in FEATURES:
        yr=[]
        for y in [2016,2017,2018]:
            x=d[(d.date>=y*10000+101)&(d.date<=y*10000+1231)&d.eligible&(d.fwd20_date<=y*10000+1231)].copy()
            dates=sorted(x.date.unique())[::5]; x=x[x.date.isin(dates)]
            vals=[v for _,v in daily_ic(x,feat)]
            med=float(np.median(vals)) if vals else np.nan; mean=float(np.mean(vals)) if vals else np.nan
            yr.append(med); rows.append({'feature':feat,'year':y,'median_daily_ic':med,'mean_daily_ic':mean,'sample_dates':len(vals)})
        a=np.array(yr,dtype=float); valid=np.isfinite(a)
        signs=np.sign(a[valid]); same=bool(valid.sum()==3 and np.all(signs==signs[0]) and signs[0]!=0)
        score=float(np.nanmean(np.abs(a))) if valid.any() else np.nan
        rows.append({'feature':feat,'year':'SUMMARY','median_daily_ic':float(np.nanmean(a)),'mean_daily_ic':score,'sample_dates':int(valid.sum()),'same_sign_3y':same,'min_abs_year_ic':float(np.nanmin(np.abs(a))) if valid.any() else np.nan})
    f=pd.DataFrame(rows); s=f[f.year=='SUMMARY'].copy()
    robust=s[(s.same_sign_3y==True)&(s.min_abs_year_ic>=0.005)&(s.mean_daily_ic>=0.010)].copy()
    robust['direction']=np.sign(robust.median_daily_ic).astype(int)
    robust=robust.sort_values(['mean_daily_ic','min_abs_year_ic'],ascending=False)
    return f,robust

def make_variants(robust):
    feats=robust.feature.tolist(); direction=dict(zip(robust.feature,robust.direction))
    vars=[]
    if len(feats)<2: return vars,direction
    for k in sorted(set([2,min(3,len(feats)),min(5,len(feats))])):
        selected=feats[:k]
        for q in [0.90,0.95]:
            for hold in [10,20]:
                for breadth in [0.0,0.35]:
                    vars.append({'factor_count':k,'features':selected,'quantile':q,'hold':hold,'breadth_gate':breadth})
    return vars,direction

def score_frame(x,v,direction):
    cols=[]
    for feat in v['features']:
        p=x[feat+'_pct']; cols.append((p-0.5)*float(direction[feat]))
    if not cols: return pd.Series(np.nan,index=x.index)
    s=sum(cols)/len(cols)
    return s.rank(pct=True)

def simulate(d,v,direction,start,end):
    dd=d[(d.date>=start)&(d.date<=end)].copy(); dates=sorted(dd.date.unique()); by={dt:x.set_index('code',drop=False) for dt,x in dd.groupby('date')}
    cash=START_CAPITAL; pos={}; pb=[]; ps=set(); trades=[]; navs=[]
    for dt in dates:
        x=by[dt]
        for code,p in list(pos.items()):
            if code in x.index and bool(x.loc[code].action_flag): p['shares']=max(int(math.floor(p['shares']*float(x.loc[code].bridge))),0)
        for code in list(ps):
            if code not in pos or code not in x.index: continue
            r=x.loc[code]; p=pos.pop(code); sh=int(p['shares']); px=float(r.open)*(1-SELL_SLIP); proceeds=sh*px*(1-FEE-SELL_TAX); cash+=proceeds
            trades.append({'code':code,'entry_date':p['entry_date'],'exit_date':dt,'shares':sh,'entry_price':p['entry_price'],'exit_price':px,'pnl':proceeds-p['cost'],'return':(proceeds-p['cost'])/p['cost'],'hold_days':p['held']})
        ps=set()
        for o in pb:
            code=o['code']
            if code in pos or code not in x.index or len(pos)>=MAX_POS: continue
            r=x.loc[code]; limit=o['limit']
            if float(r.low)>limit: continue
            px=min(float(r.open)*(1+BUY_SLIP),limit) if float(r.open)<=limit else limit
            budget=min(o['budget'],cash/(1+FEE)); sh=int(math.floor(budget/px))
            if sh<=0: continue
            cost=sh*px*(1+FEE)
            if cost>cash: sh=int(math.floor(cash/(px*(1+FEE)))); cost=sh*px*(1+FEE)
            if sh<=0: continue
            cash-=cost; pos[code]={'shares':sh,'entry_date':dt,'entry_price':px,'entry_adj':px*float(r.adj_factor),'cost':cost,'held':0}
        pb=[]
        mv=0.0
        for code,p in pos.items():
            if code in x.index: mv+=p['shares']*float(x.loc[code].close); p['held']+=1
        nav=cash+mv; navs.append({'date':dt,'nav':nav})
        for code,p in pos.items():
            if code not in x.index: continue
            r=x.loc[code]
            if p['held']>=v['hold'] or float(r.adj_close)<=p['entry_adj']*0.88: ps.add(code)
        slots=MAX_POS-len(pos)
        if slots>0 and float(x.breadth_ma60.iloc[0])>=v['breadth_gate']:
            z=x[x.eligible].copy(); z['score_pct']=score_frame(z,v,direction); z=z[(z.score_pct>=v['quantile'])&~z.code.isin(pos)]
            z=z.sort_values(['score_pct','liq_pct'],ascending=False).head(slots)
            budget=min(nav*MAX_STOCK,nav*TOTAL_EXPOSURE/MAX_POS)
            for _,r in z.iterrows(): pb.append({'code':r.code,'limit':float(r.close)*1.01,'budget':budget})
    if dates:
        x=by[dates[-1]]
        for code,p in list(pos.items()):
            if code not in x.index: continue
            sh=p['shares']; px=float(x.loc[code].close)*(1-SELL_SLIP); proceeds=sh*px*(1-FEE-SELL_TAX); cash+=proceeds
            trades.append({'code':code,'entry_date':p['entry_date'],'exit_date':dates[-1],'shares':sh,'entry_price':p['entry_price'],'exit_price':px,'pnl':proceeds-p['cost'],'return':(proceeds-p['cost'])/p['cost'],'hold_days':p['held']})
        if navs: navs[-1]['nav']=cash
    n=pd.DataFrame(navs); t=pd.DataFrame(trades)
    if n.empty: return {'end_nav':START_CAPITAL,'return':0.0,'max_dd':0.0,'pf':np.nan,'win_rate':np.nan,'trades':0},t
    n['dd']=n.nav/n.nav.cummax()-1; gp=float(t.loc[t.pnl>0,'pnl'].sum()) if len(t) else 0; gl=float(-t.loc[t.pnl<0,'pnl'].sum()) if len(t) else 0
    return {'end_nav':float(n.nav.iloc[-1]),'return':float(n.nav.iloc[-1]/START_CAPITAL-1),'max_dd':float(n.dd.min()),'pf':gp/gl if gl>0 else np.nan,'win_rate':float((t.pnl>0).mean()) if len(t) else np.nan,'trades':int(len(t))},t

def main():
    d=prepare(); factor_table,robust=discover_factors(d); factor_table.to_csv(OUT/'factor_ic_by_year.csv',index=False); robust.to_csv(OUT/'robust_factors_2016_2018.csv',index=False)
    variants,direction=make_variants(robust); rows=[]; trades=[]
    for i,v in enumerate(variants):
        vid=f'FD8_{i:03d}'
        for pname,(a,b) in [('DISCOVERY_2016_2018',DISCOVERY),('OOS1_2019',OOS1),('OOS2_2020',OOS2)]:
            m,t=simulate(d,v,direction,a,b); rows.append({'variant_id':vid,'period':pname,'factor_count':v['factor_count'],'features':'|'.join(v['features']),'quantile':v['quantile'],'hold':v['hold'],'breadth_gate':v['breadth_gate'],**m})
            if len(t): trades.append(t.assign(variant_id=vid,period=pname))
    r=pd.DataFrame(rows); r.to_csv(OUT/'variant_period_results.csv',index=False)
    if trades: pd.concat(trades,ignore_index=True).to_csv(OUT/'all_trades.csv.gz',index=False,compression='gzip')
    gate=[]; frozen=[]
    if len(r):
        tr=r[r.period=='DISCOVERY_2016_2018'].set_index('variant_id'); va=r[r.period=='OOS1_2019'].set_index('variant_id')
        for vid in sorted(set(tr.index)&set(va.index)):
            a=tr.loc[vid]; b=va.loc[vid]
            rp=bool(a.trades>=20 and a['return']>0 and a.pf>=1.10 and a.max_dd>=-0.25)
            op=bool(b.trades>=4 and b['return']>0 and b.pf>=1.05 and b.max_dd>=-0.20)
            gate.append({'variant_id':vid,'research_pass':rp,'oos1_pass':op,'pre2020_pass':bool(rp and op),'research_return':a['return'],'research_pf':a.pf,'research_dd':a.max_dd,'oos1_return':b['return'],'oos1_pf':b.pf,'oos1_dd':b.max_dd})
        g=pd.DataFrame(gate); g.to_csv(OUT/'pre2020_gate.csv',index=False)
        z=g[g.pre2020_pass].copy()
        if len(z):
            z['score']=z.research_return+z.oos1_return+0.05*(z.research_pf.clip(0,3)-1)+0.05*(z.oos1_pf.clip(0,3)-1)
            frozen=z.sort_values('score',ascending=False).head(3)
    frozen=pd.DataFrame(frozen); frozen.to_csv(OUT/'frozen_before_2020.csv',index=False)
    final=[]
    if len(frozen):
        for _,s in frozen.iterrows():
            x=r[(r.variant_id==s.variant_id)&(r.period=='OOS2_2020')].iloc[0]
            final.append({'variant_id':s.variant_id,'oos2_return':x['return'],'oos2_pf':x.pf,'oos2_dd':x.max_dd,'oos2_trades':x.trades,'oos2_pass':bool(x.trades>=4 and x['return']>0 and x.pf>=1.05 and x.max_dd>=-0.20)})
    final=pd.DataFrame(final); final.to_csv(OUT/'oos2_2020_results.csv',index=False)
    manifest={'stage':'8_factor_discovery','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'oos2_2020_used_for_tuning':False,'discovery_uses_future_labels_only_within_2016_2018':True,'scale_invariant_liquidity_rank':True,'pre2020_corp_action_method':'inferred_bridge_approximate','feature_count':len(FEATURES),'robust_factor_count':int(len(robust)),'variant_count':len(variants),'pre2020_survivors':int(len(frozen)),'oos2_survivors':int(final.oos2_pass.sum()) if len(final) else 0}
    (OUT/'stage8_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('ADAPTIVE_ROUTER_STAGE8_COMPLETE'); print(json.dumps(manifest,indent=2)); print('\nROBUST FACTORS'); print(robust.to_string(index=False) if len(robust) else 'NONE'); print('\nFROZEN BEFORE 2020'); print(frozen.to_string(index=False) if len(frozen) else 'NONE'); print('\nOOS2 2020'); print(final.to_string(index=False) if len(final) else 'NONE')

if __name__=='__main__': main()
