#!/usr/bin/env python3
"""Full-Market AI V4: decision-level opportunity-cost gate.

Full-market universe remains independent of R10 entry eligibility. V4 adds a
prior-only conservative conditional-return model and permits a trade only when
even the predicted 25th-percentile net return clears the cash hurdle.
"""
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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v3 = load('fmv3', ROOT/'scripts'/'backtest_full_market_ai_v3.py')
v2m, v1, v2, base = v3.v2m, v3.v1, v3.v2, v3.base
RUN_ROOT = ROOT/'full_market_ai_v4_results'
BASELINE_DIR = RUN_ROOT/'baseline'
VARIANT_DIR = RUN_ROOT/'FULL_MARKET_AI_V4'
for m in (v3, v2m, v1, v2, base):
    if hasattr(m, 'RUN_ROOT'): m.RUN_ROOT = RUN_ROOT
    if hasattr(m, 'BASELINE_DIR'): m.BASELINE_DIR = BASELINE_DIR
    if hasattr(m, 'VARIANT_DIR'): m.VARIANT_DIR = VARIANT_DIR
v2.VARIANTS = {'FULL_MARKET_AI_V4': VARIANT_DIR}

PURGE = 60
TOP_DAILY = 5
CF = v1.CONTEXT_FEATURES
MF = v1.MODEL_FEATURES


def fit4(train: pd.DataFrame):
    med = train[MF].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    X = train[MF].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    common = dict(learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
                  min_samples_leaf=40, l2_regularization=2.0)
    mean_ret = HistGradientBoostingRegressor(**common, random_state=730).fit(X, train.y_ret20.astype(float))
    q25_ret = HistGradientBoostingRegressor(loss='quantile', quantile=0.25, **common, random_state=731).fit(X, train.y_ret20.astype(float))
    alpha = HistGradientBoostingRegressor(**common, random_state=732).fit(X, train.y_alpha20.astype(float))
    y = train.y_fail20.astype(int)
    clf = None if y.nunique() < 2 else HistGradientBoostingClassifier(**common, random_state=733).fit(X, y)
    return mean_ret, q25_ret, alpha, clf, float(y.mean()), med


def predict4(bundle, frame: pd.DataFrame):
    mean_ret, q25_ret, alpha, clf, p0, med = bundle
    X = frame[MF].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    pm = mean_ret.predict(X)
    pq = q25_ret.predict(X)
    pa = alpha.predict(X)
    pf = np.full(len(frame), p0, dtype=float) if clf is None else clf.predict_proba(X)[:, 1]
    return pm, pq, pa, pf


def expanding_v4(ev: pd.DataFrame, px: pd.DataFrame):
    ev = ev.copy()
    same_day_median = ev.loc[ev.historical_fill & ev.y_ret20.notna()].groupby('date').y_ret20.median()
    ev['y_alpha20'] = ev.y_ret20 - ev.date.map(same_day_median)
    dates = np.array(sorted(map(int, px.date.unique())), dtype=np.int64)
    didx = {d:i for i,d in enumerate(dates)}
    ctx = px[['date'] + CF].drop_duplicates('date').sort_values('date')
    outs, diagnostics, audits = [], [], []

    for year in range(2021, 2026):
        te = ev[(ev.date >= year*10000+101) & (ev.date <= year*10000+1231)].copy().reset_index(drop=True)
        if te.empty:
            raise RuntimeError(f'no test rows for {year}')
        cutoff = int(dates[didx[int(te.date.min())] - PURGE])
        tr = ev[(ev.date < cutoff) & ev.historical_fill & ev.y_alpha20.notna()].copy()
        tr = tr[tr.market_i % 5 == 0].copy()
        if len(tr) < 2000:
            raise RuntimeError(f'insufficient training rows for {year}: {len(tr)}')

        pc = ctx[ctx.date < cutoff].copy()
        cm = pc[CF].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        C = pc[CF].replace([np.inf, -np.inf], np.nan).fillna(cm).fillna(0.0)
        scaler = StandardScaler().fit(C)
        km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(scaler.transform(C))
        need = pd.concat([tr[['date']], te[['date']]]).drop_duplicates().merge(ctx, on='date', how='left')
        CN = need[CF].replace([np.inf, -np.inf], np.nan).fillna(cm).fillna(0.0)
        need['cluster'] = km.predict(scaler.transform(CN))
        cmap = dict(zip(need.date.astype(int), need.cluster.astype(int)))
        tr['cluster'] = tr.date.map(cmap).astype(int)
        te['cluster'] = te.date.map(cmap).astype(int)

        global_bundle = fit4(tr)
        bundles = {}
        for c in range(3):
            tc = tr[tr.cluster.eq(c)]
            bundles[c] = fit4(tc) if len(tc) >= 5000 else global_bundle

        pm = np.empty(len(te)); pq = np.empty(len(te)); pa = np.empty(len(te)); pf = np.empty(len(te))
        for c in range(3):
            mask = te.cluster.eq(c).to_numpy()
            if mask.any():
                pm[mask], pq[mask], pa[mask], pf[mask] = predict4(bundles[c], te.loc[mask])
        te['pred_return20'] = pm
        te['pred_q25_return20'] = pq
        te['pred_alpha20'] = pa
        te['pred_fail_prob'] = pf

        eligible = (
            (te.pred_return20 > 0) &
            (te.pred_q25_return20 > 0) &
            (te.pred_alpha20 > 0) &
            (te.pred_fail_prob < 0.50)
        )
        te['eligible'] = eligible.astype(int)
        te['ai_score'] = te.pred_q25_return20
        te['daily_rank'] = te.ai_score.where(eligible).groupby(te.date).rank(method='first', ascending=False)
        te['ai_accept'] = (eligible & (te.daily_rank <= TOP_DAILY)).astype(int)
        te['context_cluster'] = te.cluster
        te['fold_train_cutoff'] = cutoff

        lab = te[te.historical_fill & te.y_ret20.notna()].copy()
        sel = lab[lab.ai_accept.eq(1)].copy()
        auc = np.nan
        if len(lab) and lab.y_fail20.nunique() == 2:
            auc = roc_auc_score(lab.y_fail20.astype(int), lab.pred_fail_prob)
        diagnostics.append({
            'test_year': year,
            'train_cutoff': cutoff,
            'train_rows_sampled': len(tr),
            'test_universe_rows': len(te),
            'spearman_mean20': lab.pred_return20.corr(lab.y_ret20, method='spearman') if len(lab) else np.nan,
            'spearman_q25_20': lab.pred_q25_return20.corr(lab.y_ret20, method='spearman') if len(lab) else np.nan,
            'spearman_alpha20': lab.pred_alpha20.corr(lab.y_alpha20, method='spearman') if len(lab) else np.nan,
            'failure_auc': auc,
            'selected_n': len(sel),
            'selected_actual20_mean': sel.y_ret20.mean(),
            'selected_actual20_q25': sel.y_ret20.quantile(0.25) if len(sel) else np.nan,
            'selected_actual20_win': (sel.y_ret20 > 0).mean() if len(sel) else np.nan,
            'selected_actual_alpha20_mean': sel.y_alpha20.mean(),
        })

        chosen = te[te.ai_accept.eq(1)].sort_values(['date','ai_score'], ascending=[True,False]).copy()
        outs.append(chosen)
        for d, gd in te.groupby('date'):
            z = chosen[chosen.date.eq(d)]
            audits.append({
                'date': int(d), 'year': year, 'context_cluster': int(gd.cluster.iloc[0]),
                'full_executable_universe_n': int(len(gd)),
                'eligible_n': int(gd.eligible.sum()),
                'accepted_n': int(len(z)),
                'accepted_outside_r10_n': int(z.outside_r10.sum()) if len(z) else 0,
                'best_pred_q25': float(z.pred_q25_return20.max()) if len(z) else np.nan,
            })
        print(f'FULL_MARKET_V4 fold={year} cutoff={cutoff} train={len(tr)} test={len(te)} selected={len(chosen)}', flush=True)

    pred = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    return pred, pd.DataFrame(diagnostics), pd.DataFrame(audits)


def main():
    base.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl')
    px['code'] = px.code.astype(str).str.zfill(4)
    px = base.add_features(px)
    ev = v1.build_full_market_events(px)
    pred, diag, daily = expanding_v4(ev, px)

    keep = ['date','code','name','context_cluster','pred_return20','pred_q25_return20','pred_alpha20',
            'pred_fail_prob','ai_score','daily_rank','ai_accept','outside_r10','fold_train_cutoff']
    pred[keep].to_csv(RUN_ROOT/'FULL_MARKET_AI_OOS_TOP50.csv', index=False)
    diag.to_csv(RUN_ROOT/'FULL_MARKET_AI_V4_MODEL_DIAGNOSTICS.csv', index=False)
    daily.to_csv(RUN_ROOT/'FULL_MARKET_AI_V4_DAILY_AUDIT.csv', index=False)

    coverage = {
        'selected_rows': int(len(pred)),
        'selected_outside_r10_rows': int(pred.outside_r10.sum()) if len(pred) else 0,
        'selected_outside_r10_share': float(pred.outside_r10.mean()) if len(pred) else None,
        'mean_daily_executable_universe': float(daily.full_executable_universe_n.mean()),
        'mean_daily_eligible': float(daily.eligible_n.mean()),
        'mean_daily_selected': float(daily.accepted_n.mean()),
        'zero_selection_days': int((daily.accepted_n == 0).sum()),
    }
    (RUN_ROOT/'FULL_MARKET_AI_V4_COVERAGE.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')

    locked = base.LOCKED.read_text(encoding='utf-8')
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR/'runner.py'
    runner.write_text(v1.build_runner(locked).replace('FULL_MARKET_AI_V1','FULL_MARKET_AI_V4'), encoding='utf-8')
    v2.run_py_verbose(runner, VARIANT_DIR, 'execution.log')

    comp = pd.DataFrame([
        base.summarize(BASELINE_DIR, 'BASELINE_R10'),
        base.summarize(VARIANT_DIR, 'FULL_MARKET_AI_V4'),
    ])
    b, r = comp.iloc[0], comp.iloc[1]
    comp['cagr_delta_pp'] = (comp.cagr - b.cagr) * 100
    comp['dd_improvement_pp'] = (comp.max_drawdown - b.max_drawdown) * 100
    comp['pf_delta'] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp['win_delta_pp'] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT/'FULL_MARKET_AI_V4_COMPARISON.csv', index=False)

    gate = {
        'cagr_better': bool(r.cagr > b.cagr),
        'pf_not_worse': bool(r.pnl_profit_factor >= b.pnl_profit_factor),
        'dd_not_worse_over_1pp': bool(r.max_drawdown >= b.max_drawdown - 0.01),
    }
    gate['pass'] = all(gate.values())
    (RUN_ROOT/'FULL_MARKET_AI_V4_SUCCESS_GATE.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')

    print('=== V4 COVERAGE ===')
    print(json.dumps(coverage, indent=2))
    print('\n=== V4 OOS DIAGNOSTICS ===')
    print(diag.to_string(index=False))
    print('\n=== V4 PORTFOLIO ===')
    print(comp.to_string(index=False))
    print('\n=== V4 SUCCESS GATE ===')
    print(json.dumps(gate, indent=2))

if __name__ == '__main__':
    main()
