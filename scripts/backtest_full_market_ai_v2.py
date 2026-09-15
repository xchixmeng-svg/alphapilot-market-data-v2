#!/usr/bin/env python3
"""AlphaPilot Full-Market AI V2: sparse cross-sectional relative-alpha selector."""
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

v1 = load('full_market_v1', ROOT/'scripts'/'backtest_full_market_ai_v1.py')
v2 = v1.v2
base = v1.base
RUN_ROOT = ROOT/'full_market_ai_v2_results'
BASELINE_DIR = RUN_ROOT/'baseline'
VARIANT_DIR = RUN_ROOT/'FULL_MARKET_AI_V2'
v2.RUN_ROOT = RUN_ROOT; v2.BASELINE_DIR = BASELINE_DIR; v2.VARIANTS = {'FULL_MARKET_AI_V2': VARIANT_DIR}
base.RUN_ROOT = RUN_ROOT; base.BASELINE_DIR = BASELINE_DIR; base.VARIANT_DIR = VARIANT_DIR
v1.RUN_ROOT = RUN_ROOT; v1.BASELINE_DIR = BASELINE_DIR; v1.VARIANT_DIR = VARIANT_DIR

PURGE = 60
TOP_DAILY = 10
CONTEXT_FEATURES = v1.CONTEXT_FEATURES
MODEL_FEATURES = v1.MODEL_FEATURES


def fit_models(train: pd.DataFrame):
    med = train[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).median(numeric_only=True)
    X = train[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0)
    reg = HistGradientBoostingRegressor(learning_rate=0.05,max_iter=120,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=2.0,random_state=520)
    reg.fit(X, train['y_alpha20'].astype(float))
    yc = train['y_fail20'].astype(int)
    if yc.nunique() < 2:
        return reg, None, float(yc.mean()), med
    clf = HistGradientBoostingClassifier(learning_rate=0.05,max_iter=110,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=2.0,random_state=521)
    clf.fit(X,yc)
    return reg, clf, None, med


def predict_pair(bundle, frame):
    reg, clf, const, med = bundle
    X = frame[MODEL_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0)
    pa = reg.predict(X)
    pf = np.full(len(frame),const,dtype=float) if clf is None else clf.predict_proba(X)[:,1]
    return pa,pf


def expanding_oos(ev: pd.DataFrame, px: pd.DataFrame):
    ev = ev.copy()
    med_by_date = ev.loc[ev.historical_fill & ev.y_ret20.notna()].groupby('date')['y_ret20'].median()
    ev['same_date_median_ret20'] = ev['date'].map(med_by_date)
    ev['y_alpha20'] = ev['y_ret20'] - ev['same_date_median_ret20']

    dates=np.array(sorted(int(d) for d in px.date.unique()),dtype=np.int64); didx={int(d):i for i,d in enumerate(dates)}
    ctx=px[['date']+CONTEXT_FEATURES].drop_duplicates('date').sort_values('date').copy()
    outputs=[]; diagnostics=[]; daily_rows=[]
    for year in [2021,2022,2023,2024,2025]:
        test=ev[(ev.date>=year*10000+101)&(ev.date<=year*10000+1231)].copy().reset_index(drop=True)
        first=int(test.date.min()); ci=didx[first]-PURGE
        if ci<=0: raise RuntimeError(f'insufficient purge history {year}')
        cutoff=int(dates[ci])
        tr=ev[(ev.date<cutoff)&ev.historical_fill&ev.y_alpha20.notna()].copy()
        tr=tr[(tr.market_i%5)==0].copy()
        if len(tr)<2000: raise RuntimeError(f'insufficient train rows {year}: {len(tr)}')

        prior_ctx=ctx[ctx.date<cutoff].copy(); cmed=prior_ctx[CONTEXT_FEATURES].replace([np.inf,-np.inf],np.nan).median(numeric_only=True)
        Ctr=prior_ctx[CONTEXT_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(cmed).fillna(0.0)
        scaler=StandardScaler().fit(Ctr); km=KMeans(n_clusters=3,n_init=20,random_state=42).fit(scaler.transform(Ctr))
        needed=pd.concat([tr[['date']],test[['date']]]).drop_duplicates().merge(ctx,on='date',how='left')
        Cneed=needed[CONTEXT_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(cmed).fillna(0.0)
        needed['context_cluster']=km.predict(scaler.transform(Cneed)); cmap=dict(zip(needed.date.astype(int),needed.context_cluster.astype(int)))
        tr['context_cluster']=tr.date.map(cmap).astype(int); test['context_cluster']=test.date.map(cmap).astype(int)

        global_bundle=fit_models(tr); bundles={}
        for c in range(3):
            tc=tr[tr.context_cluster.eq(c)]; bundles[c]=fit_models(tc) if len(tc)>=5000 else global_bundle
        pa=np.empty(len(test)); pf=np.empty(len(test))
        for c in range(3):
            mask=test.context_cluster.eq(c).to_numpy()
            if mask.any(): pa[mask],pf[mask]=predict_pair(bundles[c],test.loc[mask])
        test['pred_alpha20']=pa; test['pred_fail_prob']=pf
        test['alpha_rank']=test.groupby('date')['pred_alpha20'].rank(method='average',pct=True)
        test['safety_rank']=test.groupby('date')['pred_fail_prob'].rank(method='average',pct=True,ascending=False)
        test['ai_score']=0.65*test.alpha_rank+0.35*test.safety_rank
        test['daily_rank']=test.groupby('date')['ai_score'].rank(method='first',ascending=False)
        test['ai_accept']=((test.daily_rank<=TOP_DAILY)&(test.pred_alpha20>0)).astype(int)
        test['fold_train_cutoff']=cutoff

        labeled=test[test.y_alpha20.notna()&test.historical_fill].copy()
        spear=labeled.pred_alpha20.corr(labeled.y_alpha20,method='spearman') if len(labeled) else np.nan
        auc=np.nan
        if len(labeled) and labeled.y_fail20.nunique()==2: auc=roc_auc_score(labeled.y_fail20.astype(int),labeled.pred_fail_prob)
        top=labeled[labeled.ai_accept.eq(1)]
        diagnostics.append({'test_year':year,'train_cutoff':cutoff,'train_rows_sampled':len(tr),'test_universe_rows':len(test),
                            'spearman_alpha20':spear,'failure_auc':auc,'selected_n':len(top),
                            'selected_actual_alpha20_mean':top.y_alpha20.mean(),'selected_actual20_mean':top.y_ret20.mean(),
                            'selected_actual20_win':(top.y_ret20>0).mean() if len(top) else np.nan})
        accepted=test[test.ai_accept.eq(1)].copy().sort_values(['date','ai_score'],ascending=[True,False])
        outputs.append(accepted)
        for d,gd in test.groupby('date'):
            ga=accepted[accepted.date.eq(d)]
            daily_rows.append({'date':int(d),'year':year,'context_cluster':int(gd.context_cluster.iloc[0]),'full_executable_universe_n':len(gd),
                               'accepted_n':len(ga),'accepted_outside_r10_n':int(ga.outside_r10.sum()) if len(ga) else 0,
                               'top_score':float(ga.ai_score.max()) if len(ga) else np.nan})
        print(f'FULL_MARKET_V2 fold={year} cutoff={cutoff} train={len(tr)} test={len(test)} accepted={len(accepted)}',flush=True)
    return pd.concat(outputs,ignore_index=True),pd.DataFrame(diagnostics),pd.DataFrame(daily_rows)


def main():
    base.prepare_baseline()
    px=pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl'); px['code']=px.code.astype(str).str.zfill(4); px=base.add_features(px)
    events=v1.build_full_market_events(px)
    pred,diag,daily=expanding_oos(events,px)
    keep=['date','code','name','context_cluster','pred_alpha20','pred_fail_prob','alpha_rank','safety_rank','ai_score','daily_rank','ai_accept','outside_r10','fold_train_cutoff']
    pred[keep].to_csv(RUN_ROOT/'FULL_MARKET_AI_OOS_TOP50.csv',index=False)
    diag.to_csv(RUN_ROOT/'FULL_MARKET_AI_V2_MODEL_DIAGNOSTICS.csv',index=False); daily.to_csv(RUN_ROOT/'FULL_MARKET_AI_V2_DAILY_AUDIT.csv',index=False)
    coverage={'selected_rows':int(len(pred)),'selected_outside_r10_rows':int(pred.outside_r10.sum()),'selected_outside_r10_share':float(pred.outside_r10.mean()) if len(pred) else np.nan,
              'mean_daily_executable_universe':float(daily.full_executable_universe_n.mean()),'mean_daily_selected':float(daily.accepted_n.mean())}
    (RUN_ROOT/'FULL_MARKET_AI_V2_COVERAGE.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')

    locked=base.LOCKED.read_text(encoding='utf-8'); VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR/'runner.py'; runner.write_text(v1.build_runner(locked).replace('FULL_MARKET_AI_V1','FULL_MARKET_AI_V2'),encoding='utf-8')
    v2.run_py_verbose(runner,VARIANT_DIR,'execution.log')
    comp=pd.DataFrame([base.summarize(BASELINE_DIR,'BASELINE_R10'),base.summarize(VARIANT_DIR,'FULL_MARKET_AI_V2')]); b,r=comp.iloc[0],comp.iloc[1]
    comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100; comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100; comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor; comp['win_delta_pp']=(comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT/'FULL_MARKET_AI_V2_COMPARISON.csv',index=False)
    gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-0.01)}; gate['pass']=all(gate.values())
    (RUN_ROOT/'FULL_MARKET_AI_V2_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    print('=== V2 COVERAGE ==='); print(json.dumps(coverage,indent=2)); print('\n=== V2 OOS DIAGNOSTICS ==='); print(diag.to_string(index=False)); print('\n=== V2 PORTFOLIO ==='); print(comp.to_string(index=False)); print('\n=== SUCCESS GATE ==='); print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
