#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
REF=ROOT/'reference_bundle'
OUT=ROOT/'adaptive_router_stage7'; OUT.mkdir(exist_ok=True)
START_CAPITAL=1_300_000.0
FEE=0.000855; SELL_TAX=0.003; BUY_SLIP=0.005; SELL_SLIP=0.005
MAX_POS=4; MAX_STOCK=0.25; TOTAL_EXPOSURE=0.90
RESEARCH=(20160101,20181231); OOS1=(20190101,20191231); OOS2=(20200101,20201231)

# Small economically interpretable grids only. No fine-grained optimizer.
VARIANTS=[]
def add(family, **kwargs):
    VARIANTS.append({'family':family, **kwargs})
for n in [40,60,120]:
    for mom in [0.06,0.12]: add('TREND_BREAKOUT', breakout=n, mom=mom, hold=25)
for pull in [0.97,1.00]:
    for mom in [0.08,0.15]: add('PULLBACK_CONTINUATION', pull=pull, mom=mom, hold=18)
for fast,slow in [(20,60),(40,80)]: add('ROTATION_MOMENTUM', fast=fast, slow=slow, hold=20)
for shock in [-0.08,-0.12]: add('MEAN_REVERSION', shock=shock, hold=8)
for width in [0.10,0.14]: add('VOL_CONTRACTION_BREAKOUT', width=width, hold=22)
for dd in [-0.15,-0.22]: add('PANIC_RECOVERY', dd=dd, hold=10)
for inst in [0.0,0.05]: add('INSTITUTIONAL_MOMENTUM', inst=inst, hold=20)

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
        p=REF/'institutional_2015_2019'/f'institutional_{y}.csv.gz'; parts.append(pd.read_csv(p,dtype={'code':str},low_memory=False))
    x=pd.read_parquet(REF/'formal_2020'/'institutional_2020_2025.parquet')
    yr=x.date.dt.year if pd.api.types.is_datetime64_any_dtype(x.date) else pd.to_datetime(x.date).dt.year
    parts.append(x[yr==2020].copy())
    d=pd.concat(parts,ignore_index=True); d.columns=[str(c).strip().lower() for c in d.columns]; d['code']=norm_code(d.code)
    d['date']=pd.to_datetime(d.date).dt.strftime('%Y%m%d').astype(int)
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
    d.loc[(d.bridge<0.10)|(d.bridge>10.0),'bridge']=1.0; d['adj_factor']=g.bridge.cumprod()
    for c in ['open','high','low','close']: d['adj_'+c]=d[c]*d.adj_factor
    d['amount']=d.close*d.volume; g=d.groupby('code',group_keys=False)
    d['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean())
    d['vma20']=g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean()); d['vol_ratio']=d.volume/d.vma20.replace(0,np.nan)
    for n in [5,10,20,40,60,80,120]:
        d[f'ma{n}']=g.adj_close.transform(lambda s,n=n:s.rolling(n,min_periods=n).mean())
        if n in [5,20,40,60,80,120]: d[f'r{n}']=g.adj_close.transform(lambda s,n=n:s.pct_change(n,fill_method=None))
    for n in [20,40,60,120]: d[f'prior{n}']=g.adj_close.transform(lambda s,n=n:s.shift(1).rolling(n,min_periods=n).max())
    d['low20']=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).min())
    d['hi20']=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
    d['range20']=(d.hi20-d.low20)/d.ma20.replace(0,np.nan)
    d['dd20']=d.adj_close/d.hi20-1
    d['inst_net']=d.foreign_net+d.trust_net; d['inst5']=g.inst_net.transform(lambda s:s.rolling(5,min_periods=3).sum())
    d['inst_ratio']=d.inst5/d.vma20.replace(0,np.nan)
    d['action_recent']=g.action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool)
    d['liquid']=(d.amt20>=30_000_000)&(d.close>=3.0)&~d.action_recent
    # Cross-sectional ranks use same-day T-close data only.
    for c in ['r20','r40','r60','r80','inst_ratio']:
        d[c+'_pct']=d.groupby('date')[c].rank(pct=True)
    return d.sort_values(['date','code']).reset_index(drop=True)

def signal(x,v):
    f=v['family']; base=x.liquid & x.ma120.notna()
    if f=='TREND_BREAKOUT':
        n=v['breakout']; m=base&(x.adj_close>x[f'prior{n}'])&(x.ma20>x.ma60)&(x.ma60>x.ma120)&(x.r20>v['mom'])&(x.vol_ratio>=1.0)
        s=1.5*x.r20+x.r60+0.04*np.log1p(x.vol_ratio.clip(lower=0))+0.03*x.inst_ratio.clip(-1,1)
    elif f=='PULLBACK_CONTINUATION':
        q=x.adj_close/x.ma20; m=base&(x.ma60>x.ma120)&(x.r60>v['mom'])&(x.r20>0)&(x.r5<0.01)&(q>=v['pull'])&(q<=1.015)
        s=x.r60+0.5*x.r20-2*abs(q-1)+0.03*x.inst_ratio.clip(-1,1)
    elif f=='ROTATION_MOMENTUM':
        a=x[f"r{v['fast']}_pct"]; b=x[f"r{v['slow']}_pct"]
        m=base&(x.adj_close>x.ma60)&(x.ma60>x.ma120)&(a>=0.90)&(b>=0.80)&(x.r5>-0.06)
        s=a+b+0.10*x.inst_ratio_pct
    elif f=='MEAN_REVERSION':
        m=base&(x.adj_close>x.ma120*0.95)&(x.r5<=v['shock'])&(x.r20>-0.25)&(x.vol_ratio>=0.8)
        s=-x.r5-0.3*abs(x.r20)+0.02*x.inst_ratio.clip(-1,1)
    elif f=='VOL_CONTRACTION_BREAKOUT':
        m=base&(x.range20<=v['width'])&(x.adj_close>x.prior20)&(x.ma60>x.ma120)&(x.vol_ratio>=1.15)
        s=-x.range20+0.7*x.r20+0.2*x.r60
    elif f=='PANIC_RECOVERY':
        m=base&(x.dd20<=v['dd'])&(x.r5>0)&(x.adj_close>x.low20*1.05)&(x.vol_ratio>=1.0)
        s=-x.dd20+x.r5+0.02*x.inst_ratio.clip(-1,1)
    else:
        m=base&(x.ma20>x.ma60)&(x.ma60>x.ma120)&(x.r20_pct>=0.85)&(x.r60_pct>=0.75)&(x.inst_ratio>=v['inst'])
        s=x.r20_pct+x.r60_pct+0.2*x.inst_ratio_pct
    return m,s

def should_exit(r,pos,v):
    f=v['family']; held=pos['held']; c=r.adj_close; e=pos['entry_adj']
    if c<=e*0.88: return True
    if held>=v['hold']: return True
    if f in ['TREND_BREAKOUT','ROTATION_MOMENTUM','INSTITUTIONAL_MOMENTUM'] and pd.notna(r.ma20) and c<r.ma20: return True
    if f=='PULLBACK_CONTINUATION' and ((pd.notna(r.ma60) and c<r.ma60) or (pd.notna(r.ma20) and c>r.ma20*1.08)): return True
    if f in ['MEAN_REVERSION','PANIC_RECOVERY'] and pd.notna(r.ma20) and c>=r.ma20: return True
    if f=='VOL_CONTRACTION_BREAKOUT' and pd.notna(r.ma20) and c<r.ma20: return True
    return False

def simulate(d,v,start,end):
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
            if code in x.index and should_exit(x.loc[code],p,v): ps.add(code)
        slots=MAX_POS-len(pos)
        if slots>0:
            m,s=signal(x,v); c=x.loc[m].copy(); c['score']=s.loc[m]; c=c[~c.code.isin(pos)].sort_values(['score','amount'],ascending=False).head(slots)
            budget=min(nav*MAX_STOCK,nav*TOTAL_EXPOSURE/MAX_POS)
            for _,r in c.iterrows(): pb.append({'code':r.code,'limit':float(r.close)*1.01,'budget':budget})
    if dates:
        x=by[dates[-1]]
        for code,p in list(pos.items()):
            if code not in x.index: continue
            sh=p['shares']; px=float(x.loc[code].close)*(1-SELL_SLIP); proceeds=sh*px*(1-FEE-SELL_TAX); cash+=proceeds
            trades.append({'code':code,'entry_date':p['entry_date'],'exit_date':dates[-1],'shares':sh,'entry_price':p['entry_price'],'exit_price':px,'pnl':proceeds-p['cost'],'return':(proceeds-p['cost'])/p['cost'],'hold_days':p['held']})
        if navs: navs[-1]['nav']=cash
    n=pd.DataFrame(navs); t=pd.DataFrame(trades)
    if n.empty: return {'end_nav':START_CAPITAL,'return':0,'max_dd':0,'pf':np.nan,'win_rate':np.nan,'trades':0},t
    n['dd']=n.nav/n.nav.cummax()-1; ret=float(n.nav.iloc[-1]/START_CAPITAL-1)
    gp=float(t.loc[t.pnl>0,'pnl'].sum()) if len(t) else 0; gl=float(-t.loc[t.pnl<0,'pnl'].sum()) if len(t) else 0
    return {'end_nav':float(n.nav.iloc[-1]),'return':ret,'max_dd':float(n.dd.min()),'pf':gp/gl if gl>0 else np.nan,'win_rate':float((t.pnl>0).mean()) if len(t) else np.nan,'trades':int(len(t))},t

def vid(i,v): return f"{v['family']}_{i:02d}"

def main():
    d=prepare(); rows=[]; trade_parts=[]
    periods=[('RESEARCH_2016_2018',*RESEARCH),('OOS1_2019',*OOS1),('OOS2_2020',*OOS2)]
    for i,v in enumerate(VARIANTS):
        for pname,a,b in periods:
            m,t=simulate(d,v,a,b); row={'variant_id':vid(i,v),'family':v['family'],'period':pname,**v,**m}; rows.append(row)
            if len(t): t=t.assign(variant_id=vid(i,v),family=v['family'],period=pname); trade_parts.append(t)
    r=pd.DataFrame(rows)
    # Gate is frozen before OOS2: research + OOS1 only. OOS2 never selects/tunes.
    tr=r[r.period=='RESEARCH_2016_2018'].set_index('variant_id'); va=r[r.period=='OOS1_2019'].set_index('variant_id')
    ids=sorted(set(tr.index)&set(va.index)); gate=[]
    for x in ids:
        a=tr.loc[x]; b=va.loc[x]
        passed=(a.trades>=12 and a['return']>0 and a.pf>=1.10 and a.max_dd>=-0.25 and b.trades>=3 and b['return']>0 and b.pf>=1.0)
        gate.append({'variant_id':x,'family':a.family,'research_return':a['return'],'research_pf':a.pf,'research_dd':a.max_dd,'research_trades':a.trades,'oos1_return':b['return'],'oos1_pf':b.pf,'oos1_dd':b.max_dd,'oos1_trades':b.trades,'passed_pre2020_gate':bool(passed)})
    g=pd.DataFrame(gate)
    # One survivor per independent family, ranked without OOS2.
    survivors=[]
    if len(g):
        z=g[g.passed_pre2020_gate].copy(); z['pre2020_score']=z.research_return+z.oos1_return+0.05*(z.research_pf.clip(0,3)-1)+0.05*(z.oos1_pf.clip(0,3)-1)
        survivors=z.sort_values('pre2020_score',ascending=False).groupby('family',as_index=False).head(1)
    r.to_csv(OUT/'all_variant_period_results.csv',index=False); g.to_csv(OUT/'pre2020_gate.csv',index=False); survivors.to_csv(OUT/'family_survivors_frozen_before_2020.csv',index=False)
    if trade_parts: pd.concat(trade_parts,ignore_index=True).to_csv(OUT/'all_trades.csv.gz',index=False,compression='gzip')
    final=[]
    for _,s in survivors.iterrows():
        x=r[(r.variant_id==s.variant_id)&(r.period=='OOS2_2020')].iloc[0]
        final.append({'variant_id':s.variant_id,'family':s.family,'oos2_2020_return':x['return'],'oos2_2020_pf':x.pf,'oos2_2020_dd':x.max_dd,'oos2_2020_trades':x.trades,'oos2_pass':bool(x.trades>=3 and x['return']>0 and x.pf>=1.0)})
    final=pd.DataFrame(final); final.to_csv(OUT/'oos2_2020_frozen_results.csv',index=False)
    manifest={'stage':'7_multi_strategy_factory','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'oos2_2020_used_for_tuning':False,'fine_grid_optimization':False,'research_period':'2016-2018','oos1':'2019','oos2':'2020','variant_count':len(VARIANTS),'pre2020_survivors':int(len(survivors)),'oos2_survivors':int(final.oos2_pass.sum()) if len(final) else 0}
    (OUT/'stage7_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('ADAPTIVE_ROUTER_STAGE7_COMPLETE'); print(json.dumps(manifest,indent=2))
    print('\nPRE2020 SURVIVORS'); print(survivors.to_string(index=False) if len(survivors) else 'NONE')
    print('\nFROZEN OOS2 2020'); print(final.to_string(index=False) if len(final) else 'NONE')

if __name__=='__main__': main()
