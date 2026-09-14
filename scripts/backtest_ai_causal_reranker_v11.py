#!/usr/bin/env python3
# trigger: v11 workflow after workflow registration
"""AlphaPilot AI causal reranker V11: strategy-path counterfactual frontier labels.

V10 proved the slot-frontier intervention reaches real orders, but the fixed 20-session
label was misaligned with R10's actual strategy exits. V11 keeps the V10 execution
frontier and replaces the learning target with a causal shadow trade under the same
strategy-specific exit logic.

Contract:
- Locked R10 eligibility/execution/exits/sizing/caps/common cash remain immutable.
- Expanding-year OOS + 60-session purge remain unchanged.
- AI can only swap the strategy cutoff candidate with the immediately adjacent N+1 name.
- Training uses prior-year OOS frontier pairs only.
- Pair outcome is lower shadow trade net return minus upper shadow trade net return.
- Shadow entry uses the locked T+1 limit/fill convention; exit trigger follows locked
  R7/R05 rules and executes T+1 with locked 0.5% adverse sell slippage.
- Fixed LogisticRegression + Ridge; no tuning on 2025 outcomes.
- Require >=20 prior valid shadow-labeled frontier pairs and both hit classes.
- Promotion needs Pareto model agreement, P(hit)>=0.60, and predicted uplift>0.
- Success gate unchanged: CAGR > R10, PF >= R10, Max DD no more than 1pp worse.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'scripts'/'backtest_ai_causal_reranker_v10.py'
spec=importlib.util.spec_from_file_location('ai_v10',src)
v10=importlib.util.module_from_spec(spec); spec.loader.exec_module(v10)
v2=v10.v2

RUN_ROOT=ROOT/'ai_causal_reranker_v11_results'
BASELINE_DIR=RUN_ROOT/'baseline'
VARIANT_DIR=RUN_ROOT/'AI_PATH_FRONTIER'
HORIZONS=[20,40,60]
# Keep every imported helper on the V11 run-local baseline.  V10's helper functions
# resolve their own module globals, so updating only v2/mod leaves mark_baseline_frontier
# pointing at ai_causal_reranker_v10_results/baseline on a clean CI runner.
v10.RUN_ROOT=RUN_ROOT; v10.BASELINE_DIR=BASELINE_DIR; v10.VARIANT_DIR=VARIANT_DIR
v2.RUN_ROOT=RUN_ROOT; v2.BASELINE_DIR=BASELINE_DIR; v2.VARIANTS={'AI_PATH_FRONTIER':VARIANT_DIR}
v2.mod.RUN_ROOT=RUN_ROOT; v2.mod.BASELINE_DIR=BASELINE_DIR; v2.mod.VARIANT_DIR=VARIANT_DIR; v2.mod.HORIZONS=HORIZONS
BUY_FEE=0.000855; SELL_FEE=0.000855; SELL_TAX=0.003; BUY_ADVERSE=0.005; SELL_ADVERSE=0.005; MIN_HOLD=3

def tick(p): return 0.01 if p<10 else 0.05 if p<50 else 0.1 if p<100 else 0.5 if p<500 else 1.0 if p<1000 else 5.0
def floor_tick(p):
    t=tick(p); return round(np.floor((p+1e-10)/t)*t,4)
def ceil_tick(p):
    t=tick(p); return round(np.ceil((p-1e-10)/t)*t,4)
def buy_fill(op,lo,lim):
    if op<=lim: return min(ceil_tick(op*(1+BUY_ADVERSE)),lim)
    return lim if lo<=lim else None

def shadow_return(px, date_to_i, dates, signal_date, code, strategy):
    """Independent one-share causal shadow trade; returns net return or NaN."""
    i=date_to_i.get(int(signal_date));
    if i is None or i+1>=len(dates): return np.nan
    sd=int(signal_date); ed=dates[i+1]
    s=px[(px.date==sd)&(px.code==code)]
    e=px[(px.date==ed)&(px.code==code)]
    if len(s)!=1 or len(e)!=1: return np.nan
    sr=s.iloc[0]; er=e.iloc[0]
    lim=floor_tick(float(sr.close)*(0.98 if strategy=='R7' else 0.995))
    fill=buy_fill(float(er.open),float(er.low),lim)
    if fill is None or fill<=0: return np.nan
    adj_factor=float(er.aclose)/float(er.close) if float(er.close)>0 else np.nan
    if not np.isfinite(adj_factor): return np.nan
    entry_idx=fill*adj_factor
    peak=float(er.aclose); state=None; hold=0
    exit_idx=np.nan
    for j in range(i+1,len(dates)-1):
        d=dates[j]
        rr=px[(px.date==d)&(px.code==code)]
        if len(rr)!=1: continue
        r=rr.iloc[0]; hold+=1; idx=float(r.aclose); peak=max(peak,idx)
        if hold<MIN_HOLD: continue
        ret=idx/entry_idx-1.0; reason=False
        if strategy=='R7':
            reason=(ret<=-0.12) or (float(r.get('r7_exposure',1.0))<=0.0) or (not bool(r.get('r7_hard',False)))
        else:
            ar=r.get('amount_ratio',np.nan)
            if state is None and ret>=0.40 and np.isfinite(ar) and ar>=2.0: state='runner'
            if state=='runner' and ret>=0.80: state='mega' if np.isfinite(ar) and ar>=1.2 else 'target'
            if ret<=-0.10: reason=True
            elif state=='target' and (ret>=2.00 or idx<=peak*0.80): reason=True
            elif state=='mega' and idx<=peak*0.84: reason=True
            elif state=='runner' and idx<=peak*0.86: reason=True
            elif state is None and ret>=0.50 and idx<=peak*0.88: reason=True
            elif hold>=(120 if state is not None else 60): reason=True
        if reason:
            xd=dates[j+1]; xr=px[(px.date==xd)&(px.code==code)]
            if len(xr)!=1: return np.nan
            xr=xr.iloc[0]
            raw_exit=floor_tick(float(xr.open)*(1-SELL_ADVERSE))
            af=float(xr.aclose)/float(xr.close) if float(xr.close)>0 else np.nan
            if not np.isfinite(af): return np.nan
            exit_idx=raw_exit*af; break
    if not np.isfinite(exit_idx): return np.nan
    gross_ratio=exit_idx/entry_idx
    return gross_ratio*(1-SELL_FEE-SELL_TAX)/(1+BUY_FEE)-1.0

def build_runner(locked_source:str)->str:
    return v10.build_runner(locked_source).replace('V10 strategy-slot frontier AI tie-break','V11 strategy-path-labeled frontier AI tie-break')

def main():
    v2.mod.prepare_baseline()
    px=pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl'); px['code']=px.code.astype(str).str.zfill(4)
    px=v2.mod.add_features(px)
    ev=v2.mod.candidate_events(px); ev=v2.mod.attach_labels(ev,px)
    pred,diag=v2.mod.fit_oos(ev,px)
    pred['ai_pair_return']=pred[[f'pred_ret_{h}' for h in HORIZONS]].mean(axis=1)
    pred['ai_pair_fail']=pred[[f'pred_fail_{h}' for h in HORIZONS]].mean(axis=1)
    pairs=v10.mark_baseline_frontier(v10.strategy_adjacent_pairs(pred))

    dates=sorted(int(x) for x in px.date.unique()); date_to_i={d:i for i,d in enumerate(dates)}
    cache={}
    def sr(d,c,s):
        k=(int(d),str(c),str(s))
        if k not in cache: cache[k]=shadow_return(px,date_to_i,dates,*k)
        return cache[k]
    pairs['upper_shadow']=[sr(d,c,s) for d,c,s in zip(pairs.date,pairs.upper_code,pairs.strategy)]
    pairs['lower_shadow']=[sr(d,c,s) for d,c,s in zip(pairs.date,pairs.lower_code,pairs.strategy)]
    pairs['path_uplift']=pairs.lower_shadow-pairs.upper_shadow
    pairs['path_hit']=np.where(pairs.path_uplift.notna(),(pairs.path_uplift>0).astype(float),np.nan)
    train=pairs[pairs.execution_frontier & pairs.path_uplift.notna() & pairs.path_hit.notna()].copy()

    fcols=['ret_margin','fail_margin']; scored=[]; gates=[]
    for yr in [2022,2023,2024,2025]:
        prior=train[train.year<yr].copy(); cur=pairs[pairs.year==yr].copy()
        eligible=len(prior)>=20 and prior.path_hit.nunique()>=2
        if eligible and len(cur):
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=42))
            reg=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
            clf.fit(prior[fcols],prior.path_hit.astype(int)); reg.fit(prior[fcols],prior.path_uplift.astype(float))
            cur['meta_p_hit']=clf.predict_proba(cur[fcols])[:,1]; cur['meta_pred_path_uplift']=reg.predict(cur[fcols])
            cur['base_pareto']=(cur.ret_margin>0)&(cur.fail_margin>0)
            cur['meta_allow']=cur.base_pareto&(cur.meta_p_hit>=0.60)&(cur.meta_pred_path_uplift>0)
        else:
            cur['meta_p_hit']=np.nan; cur['meta_pred_path_uplift']=np.nan; cur['base_pareto']=(cur.ret_margin>0)&(cur.fail_margin>0); cur['meta_allow']=False
        scored.append(cur); gates.append({'test_year':yr,'prior_path_frontier_n':int(len(prior)),'eligible':bool(eligible),'allowed_pairs':int(cur.meta_allow.sum())})
    scored=pd.concat(scored,ignore_index=True)
    scored.to_csv(RUN_ROOT/'AI_SLOT_PERMISSIONS.csv',index=False)
    pairs.to_csv(RUN_ROOT/'AI_PATH_FRONTIER_PAIRS.csv',index=False); train.to_csv(RUN_ROOT/'AI_PATH_FRONTIER_TRAINING.csv',index=False)
    pred.to_csv(RUN_ROOT/'AI_OOS_PREDICTIONS.csv',index=False); diag.to_csv(RUN_ROOT/'AI_OOS_MODEL_DIAGNOSTICS.csv',index=False)
    pd.DataFrame(gates).to_csv(RUN_ROOT/'AI_V11_META_GATE.csv',index=False)

    locked=v2.mod.LOCKED.read_text(encoding='utf-8'); VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR/'runner.py'; runner.write_text(build_runner(locked),encoding='utf-8'); v2.run_py_verbose(runner,VARIANT_DIR,'execution.log')
    comp=pd.DataFrame([v2.mod.summarize(BASELINE_DIR,'BASELINE_R10'),v2.mod.summarize(VARIANT_DIR,'AI_PATH_FRONTIER')])
    b=comp.iloc[0]; r=comp.iloc[1]
    comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100; comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100
    comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor; comp['win_delta_pp']=(comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT/'AI_CAUSAL_V11_COMPARISON.csv',index=False)
    gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-0.01)}; gate['pass']=all(gate.values())
    (RUN_ROOT/'AI_V11_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    print('=== V11 PATH-LABEL GATE ==='); print(pd.DataFrame(gates).to_string(index=False))
    print('\n=== V11 TRAINING ==='); print(train.groupby(['year','strategy']).agg(n=('path_hit','size'),hit_rate=('path_hit','mean'),mean_path_uplift=('path_uplift','mean')).reset_index().to_string(index=False) if len(train) else 'NO TRAINING PAIRS')
    print('\n=== V11 PORTFOLIO ==='); print(comp.to_string(index=False)); print('\n=== SUCCESS GATE ==='); print(json.dumps(gate,indent=2))
if __name__=='__main__': main()
