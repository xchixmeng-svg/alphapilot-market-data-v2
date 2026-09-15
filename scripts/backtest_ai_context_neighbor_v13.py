#!/usr/bin/env python3
"""AlphaPilot AI V13: contextual nearest-evidence router.

Research change vs V12:
- Keep locked R10 execution, exits, sizing, T+1, common cash, caps unchanged.
- Keep strategy-slot frontier intervention (rank N vs N+1 only).
- Keep strategy-path shadow labels and the indexed fast implementation.
- Replace the parametric LINEAR/TREE family gate that abstained everywhere with
  a non-parametric prior-OOS evidence router.
- Each decision first auto-routes by strategy+market_state, then strategy, then global.
- Within that route, compare the current pair only with nearest historical frontier
  decisions using robustly scaled ret_margin/fail_margin. No current-year labels.
- Intervention requires lower candidate to dominate on both margins, nearest prior
  evidence mean path uplift > 0 and empirical hit rate >= 60%.
- Neighbor count is a fixed data-size rule, not tuned on 2025.
- Final research success remains portfolio CAGR/PF/DD gate, not model statistics.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
# Load V12 research scaffolding and fast shadow implementation without executing them.
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v12=load('ai_v12',ROOT/'scripts'/'backtest_ai_causal_router_v12.py')
fast=load('ai_v12_fast',ROOT/'scripts'/'backtest_ai_causal_router_v12_fast.py')
v12.v11.shadow_return=fast.shadow_return_fast
v10=v12.v10; v2=v12.v2

RUN_ROOT=ROOT/'ai_context_neighbor_v13_results'
BASELINE_DIR=RUN_ROOT/'baseline'
VARIANT_DIR=RUN_ROOT/'AI_CONTEXT_NEIGHBOR'
HORIZONS=[20,40,60]
v2.RUN_ROOT=RUN_ROOT; v2.BASELINE_DIR=BASELINE_DIR; v2.VARIANTS={'AI_CONTEXT_NEIGHBOR':VARIANT_DIR}
v2.mod.RUN_ROOT=RUN_ROOT; v2.mod.BASELINE_DIR=BASELINE_DIR; v2.mod.VARIANT_DIR=VARIANT_DIR; v2.mod.HORIZONS=HORIZONS

FEATURES=['ret_margin','fail_margin']
ROUTE_MIN={'STRATEGY_STATE':10,'STRATEGY':12,'GLOBAL':16}


def choose_route(prior,row):
    specs=[
        ('STRATEGY_STATE',(prior.strategy==row.strategy)&(prior.market_state==row.market_state)),
        ('STRATEGY',prior.strategy==row.strategy),
        ('GLOBAL',pd.Series(True,index=prior.index)),
    ]
    for name,mask in specs:
        sub=prior[mask].copy()
        if len(sub)>=ROUTE_MIN[name]: return name,sub
    return 'ABSTAIN',prior.iloc[0:0].copy()


def neighbor_evidence(sub,row):
    # Robust scale from prior data only. Avoid divide-by-zero with deterministic fallback.
    X=sub[FEATURES].astype(float)
    med=X.median(); q1=X.quantile(.25); q3=X.quantile(.75); scale=(q3-q1).replace(0,np.nan).fillna(X.std()).replace(0,1.0).fillna(1.0)
    z=(X-med)/scale
    zr=(pd.Series({f:float(row[f]) for f in FEATURES})-med)/scale
    dist=np.sqrt(((z-zr)**2).sum(axis=1))
    n=len(sub)
    k=min(20,max(8,int(round(np.sqrt(n)*3))))
    nn=sub.loc[dist.nsmallest(k).index].copy()
    mean=float(nn.path_uplift.mean()); hit=float((nn.path_uplift>0).mean())
    med_u=float(nn.path_uplift.median())
    return nn,mean,hit,med_u,k


def build_runner(locked_source):
    return v10.build_runner(locked_source).replace('V10 strategy-slot frontier AI tie-break','V13 contextual-neighbor strategy-slot AI tie-break')


def main():
    fast.self_test(); print('FAST_SHADOW_PATCH ACTIVE',flush=True)
    v2.mod.prepare_baseline()
    px=pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl'); px['code']=px.code.astype(str).str.zfill(4)
    px=v2.mod.add_features(px)
    ev=v2.mod.candidate_events(px); ev=v2.mod.attach_labels(ev,px)
    pred,diag=v2.mod.fit_oos(ev,px)
    pred['ai_pair_return']=pred[[f'pred_ret_{h}' for h in HORIZONS]].mean(axis=1)
    pred['ai_pair_fail']=pred[[f'pred_fail_{h}' for h in HORIZONS]].mean(axis=1)

    pairs=v12.mark_baseline_frontier(v10.strategy_adjacent_pairs(pred))
    expmap=px.groupby('date')['r7_exposure'].first().to_dict() if 'r7_exposure' in px.columns else {}
    pairs['market_exposure']=[float(expmap.get(int(d),np.nan)) for d in pairs.date]
    pairs['market_state']=[v12.exposure_state(x) for x in pairs.market_exposure]

    dates=sorted(int(x) for x in px.date.unique()); didx={d:i for i,d in enumerate(dates)}; cache={}
    def shadow(d,c,s):
        k=(int(d),str(c),str(s))
        if k not in cache: cache[k]=fast.shadow_return_fast(px,didx,dates,*k)
        return cache[k]
    print('V13 stage=shadow_labels start',flush=True)
    pairs['upper_shadow']=[shadow(d,c,s) for d,c,s in zip(pairs.date,pairs.upper_code,pairs.strategy)]
    pairs['lower_shadow']=[shadow(d,c,s) for d,c,s in zip(pairs.date,pairs.lower_code,pairs.strategy)]
    pairs['path_uplift']=pairs.lower_shadow-pairs.upper_shadow
    pairs['path_hit']=np.where(pairs.path_uplift.notna(),(pairs.path_uplift>0).astype(float),np.nan)
    train=pairs[pairs.execution_frontier & pairs.path_uplift.notna()].copy()
    print(f'V13 stage=shadow_labels done train={len(train)}',flush=True)

    scored=[]; audit=[]
    for yr in [2022,2023,2024,2025]:
        prior=train[train.year<yr].copy(); cur=pairs[pairs.year==yr].copy()
        cur['meta_allow']=False; cur['route']='ABSTAIN'; cur['neighbor_n']=0; cur['neighbor_mean_uplift']=np.nan; cur['neighbor_hit_rate']=np.nan; cur['neighbor_median_uplift']=np.nan
        for idx,row in cur.iterrows():
            route,sub=choose_route(prior,row)
            if route=='ABSTAIN': continue
            nn,mean,hit,med_u,k=neighbor_evidence(sub,row)
            allow=bool((row.ret_margin>0) and (row.fail_margin>0) and (mean>0) and (hit>=0.60))
            cur.at[idx,'route']=route; cur.at[idx,'neighbor_n']=k; cur.at[idx,'neighbor_mean_uplift']=mean; cur.at[idx,'neighbor_hit_rate']=hit; cur.at[idx,'neighbor_median_uplift']=med_u; cur.at[idx,'meta_allow']=allow
            if allow:
                audit.append({'year':yr,'date':int(row.date),'strategy':str(row.strategy),'market_state':str(row.market_state),'route':route,'upper_code':str(row.upper_code),'lower_code':str(row.lower_code),'neighbor_n':k,'neighbor_mean_uplift':mean,'neighbor_hit_rate':hit,'neighbor_median_uplift':med_u})
        scored.append(cur)
    scored=pd.concat(scored,ignore_index=True)
    scored.to_csv(RUN_ROOT/'AI_SLOT_PERMISSIONS.csv',index=False)
    pairs.to_csv(RUN_ROOT/'AI_V13_ROUTE_PAIRS.csv',index=False); train.to_csv(RUN_ROOT/'AI_V13_PATH_TRAINING.csv',index=False)
    pd.DataFrame(audit).to_csv(RUN_ROOT/'AI_V13_DECISION_AUDIT.csv',index=False)
    pred.to_csv(RUN_ROOT/'AI_OOS_PREDICTIONS.csv',index=False); diag.to_csv(RUN_ROOT/'AI_OOS_MODEL_DIAGNOSTICS.csv',index=False)

    locked=v2.mod.LOCKED.read_text(encoding='utf-8'); VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR/'runner.py'; runner.write_text(build_runner(locked),encoding='utf-8'); v2.run_py_verbose(runner,VARIANT_DIR,'execution.log')
    comp=pd.DataFrame([v2.mod.summarize(BASELINE_DIR,'BASELINE_R10'),v2.mod.summarize(VARIANT_DIR,'AI_CONTEXT_NEIGHBOR')])
    b=comp.iloc[0]; r=comp.iloc[1]
    comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100; comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100; comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor; comp['win_delta_pp']=(comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT/'AI_V13_COMPARISON.csv',index=False)
    gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-0.01)}; gate['pass']=all(gate.values())
    (RUN_ROOT/'AI_V13_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    print('=== V13 INTERVENTIONS ==='); print(scored.groupby(['year','route']).meta_allow.agg(['count','sum']).reset_index().to_string(index=False))
    print('\n=== V13 PORTFOLIO ==='); print(comp.to_string(index=False)); print('\n=== SUCCESS GATE ==='); print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
