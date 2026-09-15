#!/usr/bin/env python3
"""Full-Market AI V5: prior-only held-out calibration gate."""
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v4 = load('fmv4', ROOT/'scripts'/'backtest_full_market_ai_v4.py')
v3, v2m, v1, v2, base = v4.v3, v4.v2m, v4.v1, v4.v2, v4.base
RUN_ROOT = ROOT/'full_market_ai_v5_results'
BASELINE_DIR = RUN_ROOT/'baseline'
VARIANT_DIR = RUN_ROOT/'FULL_MARKET_AI_V5'
for m in (v4, v3, v2m, v1, v2, base):
    if hasattr(m, 'RUN_ROOT'): m.RUN_ROOT = RUN_ROOT
    if hasattr(m, 'BASELINE_DIR'): m.BASELINE_DIR = BASELINE_DIR
    if hasattr(m, 'VARIANT_DIR'): m.VARIANT_DIR = VARIANT_DIR
v2.VARIANTS = {'FULL_MARKET_AI_V5': VARIANT_DIR}

PURGE = 60
CAL_SESSIONS = 60
INNER_GAP = 20
TOP_DAILY = 5
MIN_CAL_N = 50
TOP_PCT = 0.99
CF = v1.CONTEXT_FEATURES
MF = v1.MODEL_FEATURES


def predict_scored(frame: pd.DataFrame, bundles: dict[int, tuple]) -> pd.DataFrame:
    z = frame.copy().reset_index(drop=True)
    pr = np.empty(len(z)); pa = np.empty(len(z)); pf = np.empty(len(z))
    for c in range(3):
        mask = z.cluster.eq(c).to_numpy()
        if mask.any():
            pr[mask], pa[mask], pf[mask] = v3.pred(bundles[c], z.loc[mask])
    z['pred_return20'] = pr
    z['pred_alpha20'] = pa
    z['pred_fail_prob'] = pf
    z['abs_rank'] = z.groupby('date').pred_return20.rank(method='average', pct=True)
    z['alpha_rank'] = z.groupby('date').pred_alpha20.rank(method='average', pct=True)
    z['safety_rank'] = z.groupby('date').pred_fail_prob.rank(method='average', pct=True, ascending=False)
    z['ai_score'] = 0.45*z.abs_rank + 0.35*z.alpha_rank + 0.20*z.safety_rank
    z['decision_pct'] = z.groupby('date').ai_score.rank(method='average', pct=True)
    return z


def calibration_stats(cal: pd.DataFrame) -> tuple[dict[int, bool], pd.DataFrame]:
    allow = {}
    rows = []
    for c in range(3):
        z = cal[(cal.cluster.eq(c)) & cal.historical_fill & cal.y_ret20.notna() & (cal.decision_pct >= TOP_PCT)].copy()
        n = len(z)
        mean = float(z.y_ret20.mean()) if n else np.nan
        std = float(z.y_ret20.std(ddof=1)) if n > 1 else np.nan
        se = std / np.sqrt(n) if n > 1 and np.isfinite(std) else np.nan
        lcb = mean - se if np.isfinite(mean) and np.isfinite(se) else np.nan
        pos = float(z.loc[z.y_ret20 > 0, 'y_ret20'].sum()) if n else 0.0
        neg = float(-z.loc[z.y_ret20 < 0, 'y_ret20'].sum()) if n else 0.0
        pf = pos / neg if neg > 0 else (np.inf if pos > 0 else np.nan)
        win = float((z.y_ret20 > 0).mean()) if n else np.nan
        ok = bool(n >= MIN_CAL_N and np.isfinite(lcb) and lcb > 0 and np.isfinite(pf) and pf >= 1.0)
        allow[c] = ok
        rows.append({'context_cluster': c, 'cal_n': n, 'cal_mean20': mean, 'cal_std20': std,
                     'cal_lcb_1se20': lcb, 'cal_profit_factor': pf, 'cal_win_rate': win,
                     'authorized': ok})
    return allow, pd.DataFrame(rows)


def expanding_v5(ev: pd.DataFrame, px: pd.DataFrame):
    ev = ev.copy()
    same_day_median = ev.loc[ev.historical_fill & ev.y_ret20.notna()].groupby('date').y_ret20.median()
    ev['y_alpha20'] = ev.y_ret20 - ev.date.map(same_day_median)
    dates = np.array(sorted(map(int, px.date.unique())), dtype=np.int64)
    didx = {d:i for i,d in enumerate(dates)}
    ctx = px[['date'] + CF].drop_duplicates('date').sort_values('date')
    outs, diagnostics, audits, cal_rows = [], [], [], []

    for year in range(2021, 2026):
        te = ev[(ev.date >= year*10000+101) & (ev.date <= year*10000+1231)].copy().reset_index(drop=True)
        if te.empty:
            raise RuntimeError(f'no test rows for {year}')
        first_i = didx[int(te.date.min())]
        outer_i = first_i - PURGE
        cal_start_i = outer_i - CAL_SESSIONS
        model_cut_i = cal_start_i - INNER_GAP
        if model_cut_i <= 0:
            raise RuntimeError(f'insufficient inner history for {year}')
        outer_cutoff = int(dates[outer_i])
        cal_start = int(dates[cal_start_i])
        model_cutoff = int(dates[model_cut_i])

        tr = ev[(ev.date < model_cutoff) & ev.historical_fill & ev.y_alpha20.notna()].copy()
        tr = tr[tr.market_i % 5 == 0].copy()
        cal = ev[(ev.date >= cal_start) & (ev.date < outer_cutoff)].copy().reset_index(drop=True)
        if len(tr) < 2000:
            raise RuntimeError(f'insufficient model-training rows for {year}: {len(tr)}')
        if cal.empty:
            raise RuntimeError(f'empty calibration window for {year}')

        prior_ctx = ctx[ctx.date < model_cutoff].copy()
        med = prior_ctx[CF].replace([np.inf,-np.inf], np.nan).median(numeric_only=True)
        C = prior_ctx[CF].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
        scaler = StandardScaler().fit(C)
        km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(scaler.transform(C))
        needed = pd.concat([tr[['date']], cal[['date']], te[['date']]]).drop_duplicates().merge(ctx, on='date', how='left')
        CN = needed[CF].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
        needed['cluster'] = km.predict(scaler.transform(CN))
        cmap = dict(zip(needed.date.astype(int), needed.cluster.astype(int)))
        tr['cluster'] = tr.date.map(cmap).astype(int)
        cal['cluster'] = cal.date.map(cmap).astype(int)
        te['cluster'] = te.date.map(cmap).astype(int)

        global_bundle = v3.fit(tr)
        bundles = {}
        for c in range(3):
            tc = tr[tr.cluster.eq(c)]
            bundles[c] = v3.fit(tc) if len(tc) >= 5000 else global_bundle

        calp = predict_scored(cal, bundles)
        tep = predict_scored(te, bundles)
        allow, cstats = calibration_stats(calp)
        cstats.insert(0, 'test_year', year)
        cstats.insert(1, 'model_cutoff', model_cutoff)
        cstats.insert(2, 'cal_start', cal_start)
        cstats.insert(3, 'outer_cutoff', outer_cutoff)
        cal_rows.append(cstats)

        tep['context_authorized'] = tep.cluster.map(allow).fillna(False).astype(int)
        base_ok = (
            tep.context_authorized.eq(1) &
            (tep.decision_pct >= TOP_PCT) &
            (tep.pred_return20 > 0) &
            (tep.pred_alpha20 > 0) &
            (tep.pred_fail_prob < 0.50)
        )
        tep['eligible'] = base_ok.astype(int)
        tep['daily_rank'] = tep.ai_score.where(base_ok).groupby(tep.date).rank(method='first', ascending=False)
        tep['ai_accept'] = (base_ok & (tep.daily_rank <= TOP_DAILY)).astype(int)
        tep['context_cluster'] = tep.cluster
        tep['fold_train_cutoff'] = model_cutoff

        lab = tep[tep.historical_fill & tep.y_ret20.notna()].copy()
        sel = lab[lab.ai_accept.eq(1)].copy()
        diagnostics.append({
            'test_year': year, 'model_cutoff': model_cutoff, 'cal_start': cal_start,
            'outer_cutoff': outer_cutoff, 'train_rows_sampled': len(tr), 'cal_rows': len(cal),
            'test_universe_rows': len(tep), 'authorized_contexts': int(sum(allow.values())),
            'selected_n': len(sel), 'selected_actual20_mean': sel.y_ret20.mean(),
            'selected_actual20_q25': sel.y_ret20.quantile(0.25) if len(sel) else np.nan,
            'selected_actual20_win': (sel.y_ret20 > 0).mean() if len(sel) else np.nan,
            'selected_actual_alpha20_mean': sel.y_alpha20.mean(),
        })

        chosen = tep[tep.ai_accept.eq(1)].sort_values(['date','ai_score'], ascending=[True,False]).copy()
        outs.append(chosen)
        for d, gd in tep.groupby('date'):
            z = chosen[chosen.date.eq(d)]
            audits.append({'date': int(d), 'year': year, 'context_cluster': int(gd.cluster.iloc[0]),
                           'context_authorized': int(gd.context_authorized.iloc[0]),
                           'full_executable_universe_n': int(len(gd)), 'eligible_n': int(gd.eligible.sum()),
                           'accepted_n': int(len(z)),
                           'accepted_outside_r10_n': int(z.outside_r10.sum()) if len(z) else 0})
        print(f'FULL_MARKET_V5 fold={year} model_cut={model_cutoff} cal={cal_start}:{outer_cutoff} '
              f'authorized={sum(allow.values())} selected={len(chosen)}', flush=True)

    pred = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    return pred, pd.DataFrame(diagnostics), pd.DataFrame(audits), pd.concat(cal_rows, ignore_index=True)


def main():
    base.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl')
    px['code'] = px.code.astype(str).str.zfill(4)
    px = base.add_features(px)
    ev = v1.build_full_market_events(px)
    pred, diag, daily, cal = expanding_v5(ev, px)

    keep = ['date','code','name','context_cluster','context_authorized','pred_return20','pred_alpha20',
            'pred_fail_prob','abs_rank','alpha_rank','safety_rank','ai_score','decision_pct','daily_rank',
            'ai_accept','outside_r10','fold_train_cutoff']
    pred[keep].to_csv(RUN_ROOT/'FULL_MARKET_AI_OOS_TOP50.csv', index=False)
    diag.to_csv(RUN_ROOT/'FULL_MARKET_AI_V5_MODEL_DIAGNOSTICS.csv', index=False)
    daily.to_csv(RUN_ROOT/'FULL_MARKET_AI_V5_DAILY_AUDIT.csv', index=False)
    cal.to_csv(RUN_ROOT/'FULL_MARKET_AI_V5_CALIBRATION.csv', index=False)

    coverage = {
        'selected_rows': int(len(pred)),
        'selected_outside_r10_rows': int(pred.outside_r10.sum()) if len(pred) else 0,
        'selected_outside_r10_share': float(pred.outside_r10.mean()) if len(pred) else None,
        'mean_daily_executable_universe': float(daily.full_executable_universe_n.mean()),
        'mean_daily_selected': float(daily.accepted_n.mean()),
        'zero_selection_days': int((daily.accepted_n == 0).sum()),
        'authorized_day_share': float(daily.context_authorized.mean()),
    }
    (RUN_ROOT/'FULL_MARKET_AI_V5_COVERAGE.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')

    locked = base.LOCKED.read_text(encoding='utf-8')
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR/'runner.py'
    runner.write_text(v1.build_runner(locked).replace('FULL_MARKET_AI_V1','FULL_MARKET_AI_V5'), encoding='utf-8')
    v2.run_py_verbose(runner, VARIANT_DIR, 'execution.log')

    comp = pd.DataFrame([base.summarize(BASELINE_DIR,'BASELINE_R10'), base.summarize(VARIANT_DIR,'FULL_MARKET_AI_V5')])
    b, r = comp.iloc[0], comp.iloc[1]
    comp['cagr_delta_pp'] = (comp.cagr-b.cagr)*100
    comp['dd_improvement_pp'] = (comp.max_drawdown-b.max_drawdown)*100
    comp['pf_delta'] = comp.pnl_profit_factor-b.pnl_profit_factor
    comp['win_delta_pp'] = (comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT/'FULL_MARKET_AI_V5_COMPARISON.csv', index=False)
    gate = {'cagr_better': bool(r.cagr>b.cagr), 'pf_not_worse': bool(r.pnl_profit_factor>=b.pnl_profit_factor),
            'dd_not_worse_over_1pp': bool(r.max_drawdown>=b.max_drawdown-0.01)}
    gate['pass'] = all(gate.values())
    (RUN_ROOT/'FULL_MARKET_AI_V5_SUCCESS_GATE.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')

    print('=== V5 COVERAGE ==='); print(json.dumps(coverage, indent=2))
    print('\n=== V5 CALIBRATION ==='); print(cal.to_string(index=False))
    print('\n=== V5 OOS DIAGNOSTICS ==='); print(diag.to_string(index=False))
    print('\n=== V5 PORTFOLIO ==='); print(comp.to_string(index=False))
    print('\n=== V5 SUCCESS GATE ==='); print(json.dumps(gate, indent=2))

if __name__ == '__main__':
    main()
