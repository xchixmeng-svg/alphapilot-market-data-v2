#!/usr/bin/env python3
"""Full-Market AI V3: causal dual absolute/relative objective with abstention."""
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v2m=load('fmv2',ROOT/'scripts'/'backtest_full_market_ai_v2.py'); v1=v2m.v1; v2=v2m.v2; base=v2m.base
RUN_ROOT=ROOT/'full_market_ai_v3_results'; BASELINE_DIR=RUN_ROOT/'baseline'; VARIANT_DIR=RUN_ROOT/'FULL_MARKET_AI_V3'
for m in (v2m,v1,v2,base):
 if hasattr(m,'RUN_ROOT'): m.RUN_ROOT=RUN_ROOT
 if hasattr(m,'BASELINE_DIR'): m.BASELINE_DIR=BASELINE_DIR
 if hasattr(m,'VARIANT_DIR'): m.VARIANT_DIR=VARIANT_DIR
v2.VARIANTS={'FULL_MARKET_AI_V3':VARIANT_DIR}
PURGE=60; TOP_DAILY=5; CF=v1.CONTEXT_FEATURES; MF=v1.MODEL_FEATURES

def fit(train):
 med=train[MF].replace([np.inf,-np.inf],np.nan).median(numeric_only=True); X=train[MF].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.)
 kw=dict(learning_rate=.05,max_iter=120,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=2.)
 ra=HistGradientBoostingRegressor(**kw,random_state=630).fit(X,train.y_ret20.astype(float))
 rr=HistGradientBoostingRegressor(**kw,random_state=631).fit(X,train.y_alpha20.astype(float))
 y=train.y_fail20.astype(int); clf=None if y.nunique()<2 else HistGradientBoostingClassifier(**kw,random_state=632).fit(X,y)
 return ra,rr,clf,float(y.mean()),med

def pred(b,f):
 ra,rr,c,p0,med=b; X=f[MF].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.); return ra.predict(X),rr.predict(X),np.full(len(f),p0) if c is None else c.predict_proba(X)[:,1]

def expanding(ev,px):
 ev=ev.copy(); medret=ev.loc[ev.historical_fill&ev.y_ret20.notna()].groupby('date').y_ret20.median(); ev['y_alpha20']=ev.y_ret20-ev.date.map(medret)
 dates=np.array(sorted(map(int,px.date.unique())),dtype=np.int64); didx={d:i for i,d in enumerate(dates)}; ctx=px[['date']+CF].drop_duplicates('date').sort_values('date')
 outs=[]; ds=[]; audits=[]
 for year in range(2021,2026):
  te=ev[(ev.date>=year*10000+101)&(ev.date<=year*10000+1231)].copy().reset_index(drop=True); cutoff=int(dates[didx[int(te.date.min())]-PURGE])
  tr=ev[(ev.date<cutoff)&ev.historical_fill&ev.y_alpha20.notna()].copy(); tr=tr[tr.market_i%5==0].copy()
  pc=ctx[ctx.date<cutoff]; cm=pc[CF].replace([np.inf,-np.inf],np.nan).median(numeric_only=True); C=pc[CF].replace([np.inf,-np.inf],np.nan).fillna(cm).fillna(0.)
  sc=StandardScaler().fit(C); km=KMeans(n_clusters=3,n_init=20,random_state=42).fit(sc.transform(C)); need=pd.concat([tr[['date']],te[['date']]]).drop_duplicates().merge(ctx,on='date',how='left'); CN=need[CF].replace([np.inf,-np.inf],np.nan).fillna(cm).fillna(0.); need['cluster']=km.predict(sc.transform(CN)); mp=dict(zip(need.date.astype(int),need.cluster.astype(int))); tr['cluster']=tr.date.map(mp); te['cluster']=te.date.map(mp)
  gb=fit(tr); bs={c:(fit(tr[tr.cluster.eq(c)]) if len(tr[tr.cluster.eq(c)])>=5000 else gb) for c in range(3)}; ar=np.empty(len(te)); al=np.empty(len(te)); fp=np.empty(len(te))
  for c in range(3):
   mask=te.cluster.eq(c).to_numpy()
   if mask.any(): ar[mask],al[mask],fp[mask]=pred(bs[c],te.loc[mask])
  te['pred_return20']=ar; te['pred_alpha20']=al; te['pred_fail_prob']=fp
  te['abs_rank']=te.groupby('date').pred_return20.rank(pct=True); te['alpha_rank']=te.groupby('date').pred_alpha20.rank(pct=True); te['safety_rank']=te.groupby('date').pred_fail_prob.rank(pct=True,ascending=False)
  te['ai_score']=.45*te.abs_rank+.35*te.alpha_rank+.20*te.safety_rank
  eligible=(te.pred_return20>0)&(te.pred_alpha20>0)&(te.pred_fail_prob<.50); te['eligible']=eligible.astype(int); te['daily_rank']=te.ai_score.where(eligible).groupby(te.date).rank(method='first',ascending=False); te['ai_accept']=(eligible&(te.daily_rank<=TOP_DAILY)).astype(int); te['context_cluster']=te.cluster; te['fold_train_cutoff']=cutoff
  lab=te[te.historical_fill&te.y_ret20.notna()]; sel=lab[lab.ai_accept.eq(1)]; ds.append({'test_year':year,'train_cutoff':cutoff,'train_rows_sampled':len(tr),'test_universe_rows':len(te),'spearman_abs20':lab.pred_return20.corr(lab.y_ret20,method='spearman'),'spearman_alpha20':lab.pred_alpha20.corr(lab.y_alpha20,method='spearman'),'failure_auc':roc_auc_score(lab.y_fail20.astype(int),lab.pred_fail_prob) if lab.y_fail20.nunique()==2 else np.nan,'selected_n':len(sel),'selected_actual20_mean':sel.y_ret20.mean(),'selected_actual20_win':(sel.y_ret20>0).mean() if len(sel) else np.nan,'selected_actual_alpha20_mean':sel.y_alpha20.mean()})
  a=te[te.ai_accept.eq(1)].sort_values(['date','ai_score'],ascending=[True,False]); outs.append(a)
  for d,g in te.groupby('date'):
   z=a[a.date.eq(d)]; audits.append({'date':int(d),'year':year,'context_cluster':int(g.cluster.iloc[0]),'full_executable_universe_n':len(g),'eligible_n':int(g.eligible.sum()),'accepted_n':len(z),'accepted_outside_r10_n':int(z.outside_r10.sum()) if len(z) else 0})
  print(f'FULL_MARKET_V3 fold={year} cutoff={cutoff} train={len(tr)} test={len(te)} selected={len(a)}',flush=True)
 return pd.concat(outs,ignore_index=True),pd.DataFrame(ds),pd.DataFrame(audits)

def main():
 base.prepare_baseline(); px=pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl'); px['code']=px.code.astype(str).str.zfill(4); px=base.add_features(px); ev=v1.build_full_market_events(px); pr,di,da=expanding(ev,px)
 keep=['date','code','name','context_cluster','pred_return20','pred_alpha20','pred_fail_prob','abs_rank','alpha_rank','safety_rank','ai_score','daily_rank','ai_accept','outside_r10','fold_train_cutoff']; pr[keep].to_csv(RUN_ROOT/'FULL_MARKET_AI_OOS_TOP50.csv',index=False); di.to_csv(RUN_ROOT/'FULL_MARKET_AI_V3_MODEL_DIAGNOSTICS.csv',index=False); da.to_csv(RUN_ROOT/'FULL_MARKET_AI_V3_DAILY_AUDIT.csv',index=False)
 cov={'selected_rows':len(pr),'selected_outside_r10_rows':int(pr.outside_r10.sum()),'selected_outside_r10_share':float(pr.outside_r10.mean()) if len(pr) else None,'mean_daily_executable_universe':float(da.full_executable_universe_n.mean()),'mean_daily_eligible':float(da.eligible_n.mean()),'mean_daily_selected':float(da.accepted_n.mean())}; (RUN_ROOT/'FULL_MARKET_AI_V3_COVERAGE.json').write_text(json.dumps(cov,indent=2))
 locked=base.LOCKED.read_text(); VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR); runner=VARIANT_DIR/'runner.py'; runner.write_text(v1.build_runner(locked).replace('FULL_MARKET_AI_V1','FULL_MARKET_AI_V3')); v2.run_py_verbose(runner,VARIANT_DIR,'execution.log')
 comp=pd.DataFrame([base.summarize(BASELINE_DIR,'BASELINE_R10'),base.summarize(VARIANT_DIR,'FULL_MARKET_AI_V3')]); b,r=comp.iloc[0],comp.iloc[1]; comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100; comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100; comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor; comp.to_csv(RUN_ROOT/'FULL_MARKET_AI_V3_COMPARISON.csv',index=False); gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-.01)}; gate['pass']=all(gate.values()); (RUN_ROOT/'FULL_MARKET_AI_V3_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2)); print('=== V3 COVERAGE ===\n'+json.dumps(cov,indent=2)); print(di.to_string(index=False)); print(comp.to_string(index=False)); print(json.dumps(gate,indent=2))
if __name__=='__main__': main()
