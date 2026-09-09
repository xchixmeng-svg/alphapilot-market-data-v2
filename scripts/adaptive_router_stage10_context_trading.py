#!/usr/bin/env python3
from pathlib import Path
import json, math, zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent; REF=ROOT/'reference_bundle'; S9=ROOT/'stage9_input'; OUT=ROOT/'adaptive_router_stage10'; OUT.mkdir(exist_ok=True)
CAP=1_300_000.; FEE=.000855; TAX=.003; BUY_SLIP=.005; SELL_SLIP=.005; MAX_POS=4; MAX_STOCK=.25; EXP=.90

def norm(s): return s.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)
def load():
 p=[]
 for y in range(2015,2020):
  with zipfile.ZipFile(REF/'ohlcv_raw'/f'yearly_{y}.zip') as z:
   n=next(x for x in z.namelist() if x.lower().endswith('.csv')); p.append(pd.read_csv(z.open(n),dtype={'code':str},low_memory=False))
 p.append(pd.read_parquet(REF/'formal_2020'/'ohlcv_2020.parquet')); d=pd.concat(p,ignore_index=True); d.columns=[str(c).lower().strip() for c in d.columns]; d['code']=norm(d.code)
 if pd.api.types.is_datetime64_any_dtype(d.date): d['date']=d.date.dt.strftime('%Y%m%d').astype(int)
 else: d['date']=pd.to_numeric(d.date.astype(str).str.replace('-','',regex=False),errors='coerce').astype('Int64')
 for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=['date','code','open','high','low','close','volume']); d['date']=d.date.astype(int); d=d[d.code.str.fullmatch(r'[1-9]\d{3}')].sort_values(['code','date']).drop_duplicates(['code','date'],keep='last')
 g=d.groupby('code',group_keys=False); d['prev']=g.close.shift(1); d['bridge']=d.prev/d.open; d['action_flag']=d.prev.notna()&((d.bridge<.82)|(d.bridge>1.18)); d.loc[~d.action_flag,'bridge']=1.; d.loc[(d.bridge<.1)|(d.bridge>10),'bridge']=1.; d['af']=g.bridge.cumprod()
 for c in ['open','high','low','close']: d['adj_'+c]=d[c]*d.af
 g=d.groupby('code',group_keys=False); d['amount']=d.close*d.volume; d['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean()); d['liq_pct']=d.groupby('date').amt20.rank(pct=True)
 d['r20']=g.adj_close.transform(lambda s:s.pct_change(20,fill_method=None)); d['r60']=g.adj_close.transform(lambda s:s.pct_change(60,fill_method=None)); d['ma60']=g.adj_close.transform(lambda s:s.rolling(60,min_periods=60).mean()); d['dist_ma60']=d.adj_close/d.ma60-1
 d['prior60']=g.adj_close.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max()); d['break60']=d.adj_close/d.prior60-1; ret1=g.adj_close.transform(lambda s:s.pct_change(fill_method=None)); d['vol20']=ret1.groupby(d.code).transform(lambda s:s.rolling(20,min_periods=20).std())
 hi=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).max()); lo=g.adj_close.transform(lambda s:s.shift(1).rolling(20,min_periods=20).min()); d['range20']=(hi-lo)/d.ma60.replace(0,np.nan)
 d['action_recent']=g.action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool); d['eligible']=(d.liq_pct>=.35)&~d.action_recent
 cs=d[d.eligible].groupby('date').agg(breadth=('dist_ma60',lambda s:float((s>0).mean())),market_mom=('r20','median'),dispersion=('r20','std')).sort_index()
 for c in ['breadth','market_mom','dispersion']:
  cs[c+'_q70']=cs[c].shift(1).rolling(252,min_periods=120).quantile(.70); cs[c+'_q30']=cs[c].shift(1).rolling(252,min_periods=120).quantile(.30)
 ctx=pd.DataFrame(index=cs.index); ctx['BREADTH_HIGH']=cs.breadth>=cs.breadth_q70; ctx['BREADTH_LOW']=cs.breadth<=cs.breadth_q30; ctx['DISPERSION_HIGH']=cs.dispersion>=cs.dispersion_q70; ctx['DISPERSION_LOW']=cs.dispersion<=cs.dispersion_q30; ctx['MARKET_MOM_HIGH']=cs.market_mom>=cs.market_mom_q70; ctx['MARKET_MOM_LOW']=cs.market_mom<=cs.market_mom_q30
 return d.sort_values(['date','code']),ctx

def simulate(d,ctx,e,start,end):
 dd=d[(d.date>=start)&(d.date<=end)].copy(); dates=sorted(dd.date.unique()); by={dt:x.set_index('code',drop=False) for dt,x in dd.groupby('date')}; cash=CAP; pos={}; pb=[]; ps=set(); tr=[]; nav=[]
 feat=e.feature; cname=e.context; direction=int(e.direction)
 for dt in dates:
  x=by[dt]
  for code,p in list(pos.items()):
   if code in x.index and bool(x.loc[code].action_flag): p['shares']=max(int(math.floor(p['shares']*float(x.loc[code].bridge))),0)
  for code in list(ps):
   if code not in pos or code not in x.index: continue
   r=x.loc[code]; p=pos.pop(code); sh=p['shares']; px=float(r.open)*(1-SELL_SLIP); proceeds=sh*px*(1-FEE-TAX); cash+=proceeds; tr.append({'code':code,'entry_date':p['entry_date'],'exit_date':dt,'pnl':proceeds-p['cost'],'return':(proceeds-p['cost'])/p['cost'],'shares':sh})
  ps=set()
  for o in pb:
   code=o['code']
   if code in pos or code not in x.index or len(pos)>=MAX_POS: continue
   r=x.loc[code]; limit=o['limit']
   if float(r.low)>limit: continue
   px=min(float(r.open)*(1+BUY_SLIP),limit) if float(r.open)<=limit else limit; budget=min(o['budget'],cash/(1+FEE)); sh=int(math.floor(budget/px)); cost=sh*px*(1+FEE)
   if sh>0 and cost<=cash: cash-=cost; pos[code]={'shares':sh,'entry_date':dt,'cost':cost,'held':0}
  pb=[]; mv=0
  for code,p in pos.items():
   if code in x.index: mv+=p['shares']*float(x.loc[code].close); p['held']+=1
  n=cash+mv; nav.append((dt,n))
  for code,p in pos.items():
   if p['held']>=20: ps.add(code)
  on=bool(ctx.loc[dt,cname]) if dt in ctx.index and pd.notna(ctx.loc[dt,cname]) else False
  if on and len(pos)<MAX_POS:
   q=x[x.eligible & ~x.code.isin(pos)].dropna(subset=[feat]).copy()
   if len(q)>=40:
    q['rank']=q[feat].rank(pct=True); q=q[q['rank']<=.20] if direction<0 else q[q['rank']>=.80]; q=q.sort_values(['rank','amount'],ascending=[direction<0,False]).head(MAX_POS-len(pos)); budget=min(n*MAX_STOCK,n*EXP/MAX_POS)
    for _,r in q.iterrows(): pb.append({'code':r.code,'limit':float(r.close)*1.01,'budget':budget})
 if dates and pos:
  x=by[dates[-1]]
  for code,p in list(pos.items()):
   if code in x.index:
    sh=p['shares']; px=float(x.loc[code].close)*(1-SELL_SLIP); proceeds=sh*px*(1-FEE-TAX); cash+=proceeds; tr.append({'code':code,'entry_date':p['entry_date'],'exit_date':dates[-1],'pnl':proceeds-p['cost'],'return':(proceeds-p['cost'])/p['cost'],'shares':sh})
  nav[-1]=(dates[-1],cash)
 t=pd.DataFrame(tr); n=pd.DataFrame(nav,columns=['date','nav']); n['dd']=n.nav/n.nav.cummax()-1 if len(n) else 0; gp=t.loc[t.pnl>0,'pnl'].sum() if len(t) else 0; gl=-t.loc[t.pnl<0,'pnl'].sum() if len(t) else 0
 return {'end_nav':float(n.nav.iloc[-1]) if len(n) else CAP,'return':float(n.nav.iloc[-1]/CAP-1) if len(n) else 0,'max_dd':float(n.dd.min()) if len(n) else 0,'trades':len(t),'win_rate':float((t.pnl>0).mean()) if len(t) else np.nan,'pf':float(gp/gl) if gl>0 else np.nan},t

def main():
 d,ctx=load(); edges=pd.read_csv(S9/'robust_context_edges_frozen_pre2019.csv'); rows=[]; trades=[]
 for i,e in edges.iterrows():
  eid=f"E{i+1}_{e.feature}_{e.context}"
  for name,a,b in [('OOS1_2019',20190101,20191231),('OOS2_2020',20200101,20201231)]:
   m,t=simulate(d,ctx,e,a,b); rows.append({'edge_id':eid,'feature':e.feature,'context':e.context,'direction':int(e.direction),'period':name,**m})
   if len(t): t=t.assign(edge_id=eid,period=name); trades.append(t)
 r=pd.DataFrame(rows); r.to_csv(OUT/'transaction_results.csv',index=False); pd.concat(trades,ignore_index=True).to_csv(OUT/'trades.csv.gz',index=False) if trades else None
 o1=r[r.period=='OOS1_2019'].copy(); o1['gate_oos1']=(o1.trades>=3)&(o1['return']>0)&(o1.pf>=1.0)&(o1.max_dd>=-.25); frozen=o1[o1.gate_oos1].copy(); frozen.to_csv(OUT/'frozen_after_2019.csv',index=False)
 o2=r[(r.period=='OOS2_2020')&r.edge_id.isin(frozen.edge_id)].copy(); o2['gate_oos2']=(o2.trades>=3)&(o2['return']>0)&(o2.pf>=1.0)&(o2.max_dd>=-.25); o2.to_csv(OUT/'oos2_2020_frozen_results.csv',index=False)
 m={'stage':'10_context_edge_transaction_validation','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'stage9_oos2_used_for_selection':False,'oos1_2019_used_only_as_transaction_gate':True,'oos2_2020_used_for_tuning':False,'t_plus_1':True,'integer_shares':True,'shared_cash':True,'scale_invariant_liquidity_rank':True,'edge_count':int(len(edges)),'oos1_transaction_pass':int(len(frozen)),'oos2_pass_after_freeze':int(o2.gate_oos2.sum()) if len(o2) else 0,'pre2020_corp_action_method':'inferred_bridge_approximate'}; json.dump(m,open(OUT/'stage10_manifest.json','w'),indent=2); print('STAGE10_COMPLETE'); print(json.dumps(m,indent=2)); print(r.to_string(index=False))
if __name__=='__main__': main()
