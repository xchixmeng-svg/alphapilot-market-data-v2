#!/usr/bin/env python3
"""AlphaPilot AI V12: adaptive causal router, not one model for every setup.

Principle:
- Keep locked R10 execution/eligibility/exits/sizing/caps/T+1/common cash immutable.
- Keep V10's proven strategy-slot frontier intervention: only cutoff vs adjacent N+1.
- Use V11 strategy-path shadow outcome as the target, fixing the V11 baseline-path bug.
- Before predicting, automatically route each pair by information known at signal close:
  1) strategy + R10 market exposure state,
  2) strategy only,
  3) global fallback.
- Within the selected route, choose model family from prior OOS evidence only:
  LINEAR = LogisticRegression + Ridge;
  TREE   = shallow RandomForest classifier/regressor.
- Model-family choice uses expanding prior-year replay; no current/future test-year labels.
- If no route/model has enough historical support, abstain and retain R10.
- Success gate unchanged: CAGR > R10, PF >= R10, Max DD no more than 1pp worse.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'scripts'/'backtest_ai_causal_reranker_v11.py'
spec=importlib.util.spec_from_file_location('ai_v11',src)
v11=importlib.util.module_from_spec(spec); spec.loader.exec_module(v11)
v10=v11.v10; v2=v11.v2

RUN_ROOT=ROOT/'ai_causal_router_v12_results'
BASELINE_DIR=RUN_ROOT/'baseline'
VARIANT_DIR=RUN_ROOT/'AI_ADAPTIVE_ROUTER'
HORIZONS=[20,40,60]
v2.RUN_ROOT=RUN_ROOT; v2.BASELINE_DIR=BASELINE_DIR; v2.VARIANTS={'AI_ADAPTIVE_ROUTER':VARIANT_DIR}
v2.mod.RUN_ROOT=RUN_ROOT; v2.mod.BASELINE_DIR=BASELINE_DIR; v2.mod.VARIANT_DIR=VARIANT_DIR; v2.mod.HORIZONS=HORIZONS

FEATURES=['ret_margin','fail_margin']
MIN_ROUTE_N=20
MIN_CV_DECISIONS=5


def mark_baseline_frontier(pairs:pd.DataFrame)->pd.DataFrame:
    """Local baseline path: do not inherit V10 module's old RUN_ROOT."""
    orders=pd.read_csv(BASELINE_DIR/'r10max_formal_orders.csv',dtype={'code':str})
    orders['code']=orders.code.astype(str).str.zfill(4)
    entry=orders[(orders.side.astype(str)=='BUY')&(orders.reason.astype(str)=='ENTRY')]
    selected={}
    for r in entry.itertuples(index=False):
        selected.setdefault((int(r.signal_date),str(r.strategy)),set()).add(str(r.code))
    out=pairs.copy()
    out['baseline_upper_selected']=[str(c) in selected.get((int(d),str(s)),set()) for d,s,c in zip(out.date,out.strategy,out.upper_code)]
    out['baseline_lower_selected']=[str(c) in selected.get((int(d),str(s)),set()) for d,s,c in zip(out.date,out.strategy,out.lower_code)]
    out['execution_frontier']=out.baseline_upper_selected & ~out.baseline_lower_selected
    return out


def exposure_state(x):
    if not np.isfinite(x): return 'UNKNOWN'
    if x<=0.60: return 'DEFENSIVE'
    if x<=0.80: return 'NORMAL'
    return 'STRONG'


def fit_family(name,train):
    X=train[FEATURES]; y=train.path_hit.astype(int); z=train.path_uplift.astype(float)
    if name=='LINEAR':
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=42))
        reg=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    elif name=='TREE':
        clf=RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=5,random_state=42,class_weight='balanced')
        reg=RandomForestRegressor(n_estimators=200,max_depth=3,min_samples_leaf=5,random_state=42)
    else: raise ValueError(name)
    clf.fit(X,y); reg.fit(X,z)
    return clf,reg


def family_replay_score(route_train,family):
    """Expanding prior-year replay score. Returns (-inf,0,0) without evidence."""
    decisions=[]
    years=sorted(int(x) for x in route_train.year.unique())
    for y in years:
        tr=route_train[route_train.year<y]
        te=route_train[route_train.year==y]
        if len(tr)<MIN_ROUTE_N or tr.path_hit.nunique()<2 or te.empty: continue
        clf,reg=fit_family(family,tr)
        p=clf.predict_proba(te[FEATURES])[:,1]; u=reg.predict(te[FEATURES])
        allow=(te.ret_margin.to_numpy()>0)&(te.fail_margin.to_numpy()>0)&(p>=0.60)&(u>0)
        if allow.any():
            decisions.extend(te.loc[allow,'path_uplift'].astype(float).tolist())
    if len(decisions)<MIN_CV_DECISIONS: return (-np.inf,len(decisions),np.nan)
    return (float(np.mean(decisions)),len(decisions),float(np.mean(np.array(decisions)>0)))


def choose_route(prior,row):
    specs=[
        ('STRATEGY_STATE', (prior.strategy==row.strategy)&(prior.market_state==row.market_state)),
        ('STRATEGY', prior.strategy==row.strategy),
        ('GLOBAL', pd.Series(True,index=prior.index)),
    ]
    for name,mask in specs:
        sub=prior[mask]
        if len(sub)>=MIN_ROUTE_N and sub.path_hit.nunique()>=2:
            return name,sub
    return 'ABSTAIN',prior.iloc[0:0]


def build_runner(locked_source:str)->str:
    return v10.build_runner(locked_source).replace('V10 strategy-slot frontier AI tie-break','V12 adaptive-router strategy-slot AI tie-break')


def main():
    v2.mod.prepare_baseline()
    px=pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl'); px['code']=px.code.astype(str).str.zfill(4)
    px=v2.mod.add_features(px)
    ev=v2.mod.candidate_events(px); ev=v2.mod.attach_labels(ev,px)
    pred,diag=v2.mod.fit_oos(ev,px)
    pred['ai_pair_return']=pred[[f'pred_ret_{h}' for h in HORIZONS]].mean(axis=1)
    pred['ai_pair_fail']=pred[[f'pred_fail_{h}' for h in HORIZONS]].mean(axis=1)

    pairs=mark_baseline_frontier(v10.strategy_adjacent_pairs(pred))
    if 'r7_exposure' in px.columns:
        expmap=px.groupby('date')['r7_exposure'].first().to_dict()
    else:
        expmap={}
    pairs['market_exposure']=[float(expmap.get(int(d),np.nan)) for d in pairs.date]
    pairs['market_state']=[exposure_state(x) for x in pairs.market_exposure]

    dates=sorted(int(x) for x in px.date.unique()); didx={d:i for i,d in enumerate(dates)}
    cache={}
    def shadow(d,c,s):
        k=(int(d),str(c),str(s))
        if k not in cache: cache[k]=v11.shadow_return(px,didx,dates,*k)
        return cache[k]
    pairs['upper_shadow']=[shadow(d,c,s) for d,c,s in zip(pairs.date,pairs.upper_code,pairs.strategy)]
    pairs['lower_shadow']=[shadow(d,c,s) for d,c,s in zip(pairs.date,pairs.lower_code,pairs.strategy)]
    pairs['path_uplift']=pairs.lower_shadow-pairs.upper_shadow
    pairs['path_hit']=np.where(pairs.path_uplift.notna(),(pairs.path_uplift>0).astype(float),np.nan)
    train=pairs[pairs.execution_frontier & pairs.path_uplift.notna() & pairs.path_hit.notna()].copy()

    scored=[]; route_audit=[]
    for yr in [2022,2023,2024,2025]:
        prior=train[train.year<yr].copy(); cur=pairs[pairs.year==yr].copy()
        cur['meta_p_hit']=np.nan; cur['meta_pred_path_uplift']=np.nan; cur['meta_allow']=False
        cur['route']='ABSTAIN'; cur['model_family']='NONE'; cur['route_cv_uplift']=np.nan
        model_cache={}
        for idx,row in cur.iterrows():
            route,sub=choose_route(prior,row)
            if route=='ABSTAIN': continue
            route_key=(route,str(row.strategy) if route!='GLOBAL' else '*',str(row.market_state) if route=='STRATEGY_STATE' else '*')
            if route_key not in model_cache:
                scores={fam:family_replay_score(sub,fam) for fam in ['LINEAR','TREE']}
                valid=[(sc[0],fam,sc) for fam,sc in scores.items() if np.isfinite(sc[0]) and sc[0]>0]
                if not valid:
                    model_cache[route_key]=None
                else:
                    valid.sort(reverse=True)
                    _,fam,best=valid[0]
                    clf,reg=fit_family(fam,sub)
                    model_cache[route_key]=(fam,clf,reg,best)
                route_audit.append({'test_year':yr,'route_key':'|'.join(route_key),'n_train':len(sub),
                                    'linear_cv_uplift':scores['LINEAR'][0],'linear_cv_n':scores['LINEAR'][1],
                                    'tree_cv_uplift':scores['TREE'][0],'tree_cv_n':scores['TREE'][1],
                                    'chosen':'NONE' if model_cache[route_key] is None else model_cache[route_key][0]})
            chosen=model_cache[route_key]
            if chosen is None: continue
            fam,clf,reg,best=chosen
            X=row[FEATURES].to_frame().T.astype(float)
            p=float(clf.predict_proba(X)[:,1][0]); u=float(reg.predict(X)[0])
            allow=(row.ret_margin>0) and (row.fail_margin>0) and (p>=0.60) and (u>0)
            cur.at[idx,'meta_p_hit']=p; cur.at[idx,'meta_pred_path_uplift']=u; cur.at[idx,'meta_allow']=bool(allow)
            cur.at[idx,'route']=route; cur.at[idx,'model_family']=fam; cur.at[idx,'route_cv_uplift']=best[0]
        scored.append(cur)
    scored=pd.concat(scored,ignore_index=True)
    scored.to_csv(RUN_ROOT/'AI_SLOT_PERMISSIONS.csv',index=False)
    pairs.to_csv(RUN_ROOT/'AI_V12_ROUTE_PAIRS.csv',index=False); train.to_csv(RUN_ROOT/'AI_V12_PATH_TRAINING.csv',index=False)
    pd.DataFrame(route_audit).to_csv(RUN_ROOT/'AI_V12_ROUTE_AUDIT.csv',index=False)
    pred.to_csv(RUN_ROOT/'AI_OOS_PREDICTIONS.csv',index=False); diag.to_csv(RUN_ROOT/'AI_OOS_MODEL_DIAGNOSTICS.csv',index=False)

    locked=v2.mod.LOCKED.read_text(encoding='utf-8'); VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR/'runner.py'; runner.write_text(build_runner(locked),encoding='utf-8'); v2.run_py_verbose(runner,VARIANT_DIR,'execution.log')
    comp=pd.DataFrame([v2.mod.summarize(BASELINE_DIR,'BASELINE_R10'),v2.mod.summarize(VARIANT_DIR,'AI_ADAPTIVE_ROUTER')])
    b=comp.iloc[0]; r=comp.iloc[1]
    comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100; comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100
    comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor; comp['win_delta_pp']=(comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT/'AI_CAUSAL_V12_COMPARISON.csv',index=False)
    gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-0.01)}; gate['pass']=all(gate.values())
    (RUN_ROOT/'AI_V12_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    print('=== V12 ROUTE AUDIT ==='); print(pd.DataFrame(route_audit).to_string(index=False) if route_audit else 'NO ELIGIBLE ROUTES')
    print('\n=== V12 INTERVENTIONS ==='); print(scored.groupby(['year','route','model_family']).meta_allow.agg(['count','sum']).reset_index().to_string(index=False))
    print('\n=== V12 PORTFOLIO ==='); print(comp.to_string(index=False)); print('\n=== SUCCESS GATE ==='); print(json.dumps(gate,indent=2))
if __name__=='__main__': main()
