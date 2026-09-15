#!/usr/bin/env python3
"""AlphaPilot AI V15: exact runtime-frontier OOS policy.

Purpose:
- Keep locked R10 execution, sizing, exits, T+1, fees, caps and common cash unchanged.
- Fix V14's structural mismatch: an allowed research pair was not necessarily the
  actual cutoff pair after removing held names and resolving remaining strategy slots.
- First replay locked R10 with an instrumented no-AI probe and record the exact
  runtime frontier (date, strategy, upper, lower) seen by stage 3.
- Train/evaluate the causal neighbor policy only on those exact historical runtime
  frontiers, using prior-year OOS strategy-path shadow outcomes only.
- One frozen evidence regime may authorize at most one intervention per OOS year.
- Final success is still portfolio CAGR > R10, PF >= R10, and Max DD no more than
  1 percentage point worse. Model/neighbor statistics alone never count as success.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v13=load('ai_v13',ROOT/'scripts'/'backtest_ai_context_neighbor_v13.py')
v10=v13.v10; v2=v13.v2; fast=v13.fast; v12=v13.v12

RUN_ROOT=ROOT/'ai_runtime_frontier_v15_results'
BASELINE_DIR=RUN_ROOT/'baseline'
PROBE_DIR=RUN_ROOT/'RUNTIME_PROBE'
VARIANT_DIR=RUN_ROOT/'AI_RUNTIME_FRONTIER'
HORIZONS=[20,40,60]
v2.RUN_ROOT=RUN_ROOT; v2.BASELINE_DIR=BASELINE_DIR; v2.VARIANTS={'AI_RUNTIME_FRONTIER':VARIANT_DIR}
v2.mod.RUN_ROOT=RUN_ROOT; v2.mod.BASELINE_DIR=BASELINE_DIR; v2.mod.VARIANT_DIR=VARIANT_DIR; v2.mod.HORIZONS=HORIZONS
v12.BASELINE_DIR=BASELINE_DIR


def instrument_probe(locked_source:str)->str:
    src=v10.build_runner(locked_source)
    anchor="SLOT_ALLOW = {(int(r.date), str(r.strategy), str(r.upper_code), str(r.lower_code)): bool(r.meta_allow) for r in _sp.itertuples(index=False)}"
    assert anchor in src
    src=src.replace(anchor,anchor+"\nFRONTIER_LOG = []",1)
    r7="_a, _b = str(_lst[r7_free-1]), str(_lst[r7_free])\n                if SLOT_ALLOW.get((int(di), 'R7', _a, _b), False):"
    assert r7 in src
    src=src.replace(r7,"_a, _b = str(_lst[r7_free-1]), str(_lst[r7_free])\n                FRONTIER_LOG.append({'date':int(di),'strategy':'R7','upper_code':_a,'lower_code':_b})\n                if SLOT_ALLOW.get((int(di), 'R7', _a, _b), False):",1)
    r05="_a, _b = str(_lst[r05_free-1]), str(_lst[r05_free])\n                if SLOT_ALLOW.get((int(di), 'R05', _a, _b), False):"
    assert r05 in src
    src=src.replace(r05,"_a, _b = str(_lst[r05_free-1]), str(_lst[r05_free])\n                FRONTIER_LOG.append({'date':int(di),'strategy':'R05','upper_code':_a,'lower_code':_b})\n                if SLOT_ALLOW.get((int(di), 'R05', _a, _b), False):",1)
    src += "\npd.DataFrame(FRONTIER_LOG).drop_duplicates().to_csv('runtime_frontiers.csv', index=False)\n"
    return src


def main():
    fast.self_test(); v13.v12.v11.shadow_return=fast.shadow_return_fast
    print('V15 stage=prepare_baseline',flush=True)
    v2.mod.prepare_baseline()
    locked=v2.mod.LOCKED.read_text(encoding='utf-8')

    # Pass A: exact locked-R10 runtime frontier probe with AI disabled.
    pd.DataFrame(columns=['date','strategy','upper_code','lower_code','meta_allow']).to_csv(RUN_ROOT/'AI_SLOT_PERMISSIONS.csv',index=False)
    PROBE_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(PROBE_DIR)
    probe=PROBE_DIR/'runner.py'; probe.write_text(instrument_probe(locked),encoding='utf-8')
    print('V15 stage=runtime_probe start',flush=True)
    v2.run_py_verbose(probe,PROBE_DIR,'execution.log')
    front=pd.read_csv(PROBE_DIR/'runtime_frontiers.csv',dtype={'upper_code':str,'lower_code':str})
    front['upper_code']=front.upper_code.astype(str).str.zfill(4); front['lower_code']=front.lower_code.astype(str).str.zfill(4)
    front=front.drop_duplicates(['date','strategy','upper_code','lower_code'])
    print(f'V15 stage=runtime_probe done n={len(front)}',flush=True)

    px=pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl'); px['code']=px.code.astype(str).str.zfill(4); px=v2.mod.add_features(px)
    ev=v2.mod.candidate_events(px); ev=v2.mod.attach_labels(ev,px); pred,diag=v2.mod.fit_oos(ev,px)
    pred['ai_pair_return']=pred[[f'pred_ret_{h}' for h in HORIZONS]].mean(axis=1); pred['ai_pair_fail']=pred[[f'pred_fail_{h}' for h in HORIZONS]].mean(axis=1)
    pairs=v12.mark_baseline_frontier(v10.strategy_adjacent_pairs(pred))
    pairs=pairs.merge(front.assign(runtime_frontier=True),on=['date','strategy','upper_code','lower_code'],how='inner')
    expmap=px.groupby('date')['r7_exposure'].first().to_dict() if 'r7_exposure' in px.columns else {}
    pairs['market_exposure']=[float(expmap.get(int(d),np.nan)) for d in pairs.date]; pairs['market_state']=[v12.exposure_state(x) for x in pairs.market_exposure]

    dates=sorted(int(x) for x in px.date.unique()); didx={d:i for i,d in enumerate(dates)}; cache={}
    def shadow(d,c,s):
        k=(int(d),str(c),str(s))
        if k not in cache: cache[k]=fast.shadow_return_fast(px,didx,dates,*k)
        return cache[k]
    print('V15 stage=shadow_labels start',flush=True)
    pairs['upper_shadow']=[shadow(d,c,s) for d,c,s in zip(pairs.date,pairs.upper_code,pairs.strategy)]
    pairs['lower_shadow']=[shadow(d,c,s) for d,c,s in zip(pairs.date,pairs.lower_code,pairs.strategy)]
    pairs['path_uplift']=pairs.lower_shadow-pairs.upper_shadow
    pairs['path_hit']=np.where(pairs.path_uplift.notna(),(pairs.path_uplift>0).astype(float),np.nan)
    train=pairs[pairs.execution_frontier & pairs.path_uplift.notna()].copy()
    print(f'V15 stage=shadow_labels done runtime_pairs={len(pairs)} train={len(train)}',flush=True)

    scored=[]; audit=[]; used=set()
    for yr in [2022,2023,2024,2025]:
        prior=train[train.year<yr].copy(); cur=pairs[pairs.year==yr].copy()
        cur['meta_allow']=False; cur['route']='ABSTAIN'; cur['neighbor_n']=0; cur['neighbor_mean_uplift']=np.nan; cur['neighbor_hit_rate']=np.nan; cur['neighbor_median_uplift']=np.nan
        for idx,row in cur.iterrows():
            route,sub=v13.choose_route(prior,row)
            if route=='ABSTAIN': continue
            nn,mean,hit,med_u,k=v13.neighbor_evidence(sub,row)
            evidence_key=(yr,route,str(row.strategy),str(row.market_state) if route=='STRATEGY_STATE' else '*')
            raw=bool((row.ret_margin>0) and (row.fail_margin>0) and (mean>0) and (hit>=0.60))
            allow=bool(raw and evidence_key not in used)
            if allow: used.add(evidence_key)
            cur.at[idx,'route']=route; cur.at[idx,'neighbor_n']=k; cur.at[idx,'neighbor_mean_uplift']=mean; cur.at[idx,'neighbor_hit_rate']=hit; cur.at[idx,'neighbor_median_uplift']=med_u; cur.at[idx,'meta_allow']=allow
            if allow: audit.append({'year':yr,'date':int(row.date),'strategy':str(row.strategy),'market_state':str(row.market_state),'route':route,'upper_code':str(row.upper_code),'lower_code':str(row.lower_code),'neighbor_n':k,'neighbor_mean_uplift':mean,'neighbor_hit_rate':hit,'neighbor_median_uplift':med_u})
        scored.append(cur)
    scored=pd.concat(scored,ignore_index=True)
    scored.to_csv(RUN_ROOT/'AI_SLOT_PERMISSIONS.csv',index=False); pairs.to_csv(RUN_ROOT/'AI_V15_RUNTIME_PAIRS.csv',index=False); train.to_csv(RUN_ROOT/'AI_V15_PATH_TRAINING.csv',index=False); pd.DataFrame(audit).to_csv(RUN_ROOT/'AI_V15_DECISION_AUDIT.csv',index=False); pred.to_csv(RUN_ROOT/'AI_OOS_PREDICTIONS.csv',index=False); diag.to_csv(RUN_ROOT/'AI_OOS_MODEL_DIAGNOSTICS.csv',index=False)

    # Pass B: same locked R10 stage-3, now with permissions guaranteed to be baseline runtime-frontier keys.
    VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR/'runner.py'; runner.write_text(v10.build_runner(locked).replace('V10 strategy-slot frontier AI tie-break','V15 exact runtime-frontier AI policy'),encoding='utf-8')
    v2.run_py_verbose(runner,VARIANT_DIR,'execution.log')
    comp=pd.DataFrame([v2.mod.summarize(BASELINE_DIR,'BASELINE_R10'),v2.mod.summarize(VARIANT_DIR,'AI_RUNTIME_FRONTIER')]); b=comp.iloc[0]; r=comp.iloc[1]
    comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100; comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100; comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor; comp['win_delta_pp']=(comp.win_rate-b.win_rate)*100; comp.to_csv(RUN_ROOT/'AI_V15_COMPARISON.csv',index=False)
    gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-0.01)}; gate['pass']=all(gate.values()); (RUN_ROOT/'AI_V15_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    print('=== V15 INTERVENTIONS ==='); print(scored.groupby(['year','route']).meta_allow.agg(['count','sum']).reset_index().to_string(index=False) if len(scored) else 'NONE')
    print('\n=== V15 DECISION AUDIT ==='); print(pd.DataFrame(audit).to_string(index=False) if audit else 'NONE')
    print('\n=== V15 PORTFOLIO ==='); print(comp.to_string(index=False)); print('\n=== SUCCESS GATE ==='); print(json.dumps(gate,indent=2))
if __name__=='__main__': main()
