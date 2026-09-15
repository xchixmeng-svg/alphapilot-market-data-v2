#!/usr/bin/env python3
"""AlphaPilot Portfolio Decision AI V1.

Framework change: learn executable slot-action value in real R10 allocation contexts,
then choose KEEP R10 / REPLACE from full market / CASH at the runtime entry frontier.
See research/PORTFOLIO_DECISION_AI_V1_PREREGISTRATION.md.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

v3 = load('fmv3', ROOT/'scripts'/'backtest_full_market_ai_v3.py')
v1, v2, base = v3.v1, v3.v2, v3.base
RUN_ROOT = ROOT/'portfolio_decision_ai_v1_results'
BASELINE_DIR = RUN_ROOT/'baseline'
VARIANT_DIR = RUN_ROOT/'PORTFOLIO_DECISION_AI_V1'
for m in (v3, v1, v2, base):
    if hasattr(m, 'RUN_ROOT'): m.RUN_ROOT = RUN_ROOT
    if hasattr(m, 'BASELINE_DIR'): m.BASELINE_DIR = BASELINE_DIR
    if hasattr(m, 'VARIANT_DIR'): m.VARIANT_DIR = VARIANT_DIR
v2.VARIANTS = {'PORTFOLIO_DECISION_AI_V1': VARIANT_DIR}

MF = v1.MODEL_FEATURES
BUY_FEE, SELL_FEE, SELL_TAX = 0.000855, 0.000855, 0.003
SELL_ADVERSE = 0.005
PURGE = 60
AI_HARD_STOP = -0.10
AI_MAX_HOLD = 20
TOP_CACHE = 50  # storage/computation cache only; not an acceptance threshold


def make_price_cache(px: pd.DataFrame):
    out = {}
    q = px[['date','code','open','close','aclose']].copy()
    q['code'] = q.code.astype(str).str.zfill(4)
    for code, g in q.sort_values(['code','date']).groupby('code', sort=False):
        out[str(code)] = {
            'dates': g.date.to_numpy(np.int64),
            'open': g.open.to_numpy(float),
            'close': g.close.to_numpy(float),
            'aclose': g.aclose.to_numpy(float),
            'pos': {int(d): i for i, d in enumerate(g.date.to_numpy(np.int64))},
        }
    return out


def attach_policy_labels(frame: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Realized net return under the preregistered AI replacement policy."""
    z = frame.copy()
    vals, exits, reasons = [], [], []
    for r in z.itertuples(index=False):
        c = cache.get(str(r.code))
        if c is None or not bool(r.historical_fill) or not np.isfinite(r.entry_fill):
            vals.append(np.nan); exits.append(np.nan); reasons.append('NO_FILL'); continue
        p = c['pos'].get(int(r.date))
        if p is None or p + 21 >= len(c['dates']):
            vals.append(np.nan); exits.append(np.nan); reasons.append('NO_PATH'); continue
        ep = p + 1
        eclose = c['close'][ep]; eadj = c['aclose'][ep]
        if not np.isfinite(eclose) or eclose <= 0 or not np.isfinite(eadj):
            vals.append(np.nan); exits.append(np.nan); reasons.append('BAD_ENTRY'); continue
        entry_idx = eadj * float(r.entry_fill) / eclose
        cost = entry_idx * (1 + BUY_FEE)
        exitp = None; why = 'AI_MAXHOLD'
        # entry day hold_days=1; stop first becomes actionable when hold_days=3.
        for j in range(ep + 2, ep + AI_MAX_HOLD):
            if j + 1 >= len(c['dates']): break
            ret_sig = c['aclose'][j] / eadj - 1.0
            if np.isfinite(ret_sig) and ret_sig <= AI_HARD_STOP:
                exitp = j + 1; why = 'AI_HARD'; break
        if exitp is None:
            exitp = ep + AI_MAX_HOLD
        if exitp >= len(c['dates']) or c['close'][exitp] <= 0:
            vals.append(np.nan); exits.append(np.nan); reasons.append('NO_EXIT'); continue
        aopen = c['aclose'][exitp] * c['open'][exitp] / c['close'][exitp]
        proceeds = aopen * (1 - SELL_ADVERSE) * (1 - SELL_FEE - SELL_TAX)
        vals.append(proceeds / cost - 1.0)
        exits.append(int(c['dates'][exitp])); reasons.append(why)
    z['y_policy_return'] = vals
    z['y_policy_loss'] = np.where(z.y_policy_return.notna(), (z.y_policy_return <= 0).astype(float), np.nan)
    z['policy_exit_date'] = exits
    z['policy_exit_reason'] = reasons
    return z


def fit_bundle(train: pd.DataFrame):
    med = train[MF].replace([np.inf,-np.inf], np.nan).median(numeric_only=True)
    X = train[MF].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
    y = train.y_policy_return.astype(float)
    kw = dict(learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
              min_samples_leaf=40, l2_regularization=2.0)
    reg = HistGradientBoostingRegressor(**kw, random_state=9101).fit(X, y)
    yl = train.y_policy_loss.astype(int)
    clf = None if yl.nunique() < 2 else HistGradientBoostingClassifier(**kw, random_state=9102).fit(X, yl)
    p0 = float(yl.mean())
    return reg, clf, p0, med


def predict(bundle, f: pd.DataFrame):
    reg, clf, p0, med = bundle
    X = f[MF].replace([np.inf,-np.inf], np.nan).fillna(med).fillna(0.0)
    pr = reg.predict(X)
    pf = np.full(len(f), p0, dtype=float) if clf is None else clf.predict_proba(X)[:,1]
    return pr, pf


def expanding_oos(ev: pd.DataFrame, px: pd.DataFrame, baseline_orders: pd.DataFrame):
    dates = np.array(sorted(map(int, px.date.unique())), dtype=np.int64)
    didx = {int(d): i for i,d in enumerate(dates)}
    buy_orders = baseline_orders[baseline_orders.side.astype(str).eq('BUY')].copy()
    buy_orders['signal_date'] = pd.to_numeric(buy_orders.signal_date, errors='coerce').astype('Int64')
    decision_dates = set(int(x) for x in buy_orders.signal_date.dropna().astype(int).tolist())

    # Label only the framework's training contexts once: 2020 bootstrap + historical R10 BUY signal dates.
    train_context = ev[(ev.date < 20250101) & ((ev.date < 20210101) | ev.date.isin(decision_dates)) & ev.historical_fill].copy()
    cache = make_price_cache(px)
    train_context = attach_policy_labels(train_context, cache)
    train_context = train_context[train_context.y_policy_return.notna()].copy()

    stored, diagnostics, daily = [], [], []
    for year in range(2021, 2026):
        te = ev[(ev.date >= year*10000+101) & (ev.date <= year*10000+1231)].copy().reset_index(drop=True)
        first = int(te.date.min()); ci = didx[first] - PURGE
        if ci <= 0: raise RuntimeError(f'insufficient purge for {year}')
        cutoff = int(dates[ci])
        tr = train_context[train_context.date < cutoff].copy()
        if len(tr) < 2000: raise RuntimeError(f'insufficient decision-context training rows {year}: {len(tr)}')
        bundle = fit_bundle(tr)
        pr, pf = predict(bundle, te)
        te['pred_policy_return'] = pr
        te['pred_policy_loss_prob'] = pf
        te['fold_train_cutoff'] = cutoff
        te['is_top_candidate'] = 0
        top = te.sort_values(['date','pred_policy_return'], ascending=[True,False]).groupby('date', group_keys=False).head(TOP_CACHE).copy()
        top['is_top_candidate'] = 1
        r10rows = te[te.r7_hard.fillna(False) | te.r05_hard.fillna(False)].copy()
        store = pd.concat([top, r10rows], ignore_index=True).sort_values(['date','is_top_candidate','pred_policy_return'], ascending=[True,False,False]).drop_duplicates(['date','code'], keep='first')
        stored.append(store)

        # Outcome diagnostics are reporting only and are never fed back into V1.
        top_lab = attach_policy_labels(top[top.historical_fill].copy(), cache)
        top_lab = top_lab[top_lab.y_policy_return.notna()]
        auc = np.nan
        if len(top_lab) and top_lab.y_policy_loss.nunique() == 2:
            auc = roc_auc_score(top_lab.y_policy_loss.astype(int), top_lab.pred_policy_loss_prob)
        diagnostics.append({
            'test_year':year, 'train_cutoff':cutoff, 'train_rows':len(tr), 'test_universe_rows':len(te),
            'top_cached_rows':len(top), 'top_actual_policy_mean':top_lab.y_policy_return.mean(),
            'top_actual_policy_win':(top_lab.y_policy_return>0).mean() if len(top_lab) else np.nan,
            'policy_loss_auc':auc,
        })
        for d,g in te.groupby('date'):
            gt = top[top.date.eq(d)]
            daily.append({'date':int(d),'year':year,'universe_n':len(g),'top_cached_n':len(gt),
                          'top_outside_r10_n':int(gt.outside_r10.sum()) if len(gt) else 0,
                          'top_pred_return':float(gt.pred_policy_return.max()) if len(gt) else np.nan})
        print(f'PORTFOLIO_DECISION fold={year} cutoff={cutoff} train={len(tr)} test={len(te)} cache={len(store)}', flush=True)
    return pd.concat(stored, ignore_index=True), pd.DataFrame(diagnostics), pd.DataFrame(daily)


def build_runner(locked_source: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source: raise RuntimeError('stage3 marker missing')
    stage3 = locked_source.split(marker,1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "_ai = pd.read_csv('../PORTFOLIO_DECISION_AI_OOS.csv', dtype={'code': str})\n"
        "_ai['code'] = _ai['code'].astype(str).str.zfill(4)\n"
        "AI_BY_DATE = {int(d): g.set_index('code').sort_values(['is_top_candidate','pred_policy_return'], ascending=[False,False]) for d,g in _ai.groupby('date')}\n"
        "ai_decision_rows = []"
    )
    if old_load not in stage3: raise RuntimeError('load anchor missing')
    stage3 = stage3.replace(old_load, new_load, 1)

    # AI positions use the same causal policy used to create training labels.
    anchor = "        if p['strategy'] == 'R7':\n"
    repl = (
        "        if str(p['strategy']).startswith('AI_'):\n"
        "            if ret <= -0.10:\n"
        "                reason = 'AI_HARD'\n"
        "            elif p['hold_days'] >= 20:\n"
        "                reason = 'AI_MAXHOLD'\n"
        "        elif p['strategy'] == 'R7':\n"
    )
    if anchor not in stage3: raise RuntimeError('exit anchor missing')
    stage3 = stage3.replace(anchor, repl, 1)
    stage3 = stage3.replace("n_r7 = sum(p['strategy'] == 'R7' for p in positions.values())",
                            "n_r7 = sum(p['strategy'] in ('R7','AI_R7') for p in positions.values())", 1)
    stage3 = stage3.replace("n_r05 = sum(p['strategy'] == 'R05' for p in positions.values())",
                            "n_r05 = sum(p['strategy'] in ('R05','AI_R05') for p in positions.values())", 1)

    start = "            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n"
    end = "                used.add(code); slots_free -= 1\n"
    si = stage3.find(start); ei0 = stage3.find(end, si)
    if si < 0 or ei0 < 0: raise RuntimeError('entry anchors missing')
    ei = ei0 + len(end)
    block = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]
            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))
            created_mv, reserved_cost, used = 0.0, 0.0, set()
            _ai_day = AI_BY_DATE.get(int(di))
            for source_strat, source_code, source_r in combined:
                if slots_free <= 0: break
                if source_code in used: continue
                strat, code, r = source_strat, source_code, source_r
                action = 'KEEP_R10'; baseline_pred = np.nan; chosen_pred = np.nan
                baseline_fail = np.nan; chosen_fail = np.nan
                if _ai_day is not None and str(source_code) in _ai_day.index:
                    _br = _ai_day.loc[str(source_code)]
                    if isinstance(_br, pd.DataFrame): _br = _br.iloc[0]
                    baseline_pred = float(_br['pred_policy_return']); baseline_fail = float(_br['pred_policy_loss_prob'])
                    _replacement = None
                    for _acode, _ar in _ai_day.iterrows():
                        _acode = str(_acode)
                        if int(_ar.get('is_top_candidate',0)) != 1: continue
                        if _acode == str(source_code) or _acode in held or _acode in used or _acode not in sub.index: continue
                        _aret = float(_ar['pred_policy_return']); _afail = float(_ar['pred_policy_loss_prob'])
                        if _aret > 0.0 and _aret > baseline_pred and _afail <= baseline_fail:
                            _replacement = (_acode, _aret, _afail); break
                    if _replacement is not None:
                        code, chosen_pred, chosen_fail = _replacement
                        r = sub.loc[code]
                        strat = 'AI_' + source_strat
                        action = 'REPLACE'
                    elif baseline_pred <= 0.0:
                        action = 'CASH'
                        ai_decision_rows.append({'date':di,'source_strategy':source_strat,'source_code':source_code,
                                                 'action':action,'chosen_code':'','baseline_pred':baseline_pred,
                                                 'chosen_pred':np.nan,'baseline_fail':baseline_fail,'chosen_fail':np.nan})
                        continue
                ai_decision_rows.append({'date':di,'source_strategy':source_strat,'source_code':source_code,
                                         'action':action,'chosen_code':code,'baseline_pred':baseline_pred,
                                         'chosen_pred':chosen_pred,'baseline_fail':baseline_fail,'chosen_fail':chosen_fail})
                if code in used: continue
                limit = floor_tick(float(r['close']) * (0.995 if str(strat).startswith('AI_') else (0.98 if source_strat == 'R7' else 0.995)))
                base_cash = (R7_BASE if source_strat == 'R7' else R05_BASE) * nav * mult
                if source_strat == 'R7': base_cash *= r7_exposure
                target = min(base_cash, nav * SINGLE_CAP, nav * TOTAL_CAP - mv - created_mv,
                             max(0.0, cash - reserved_cost - 1000.0) / (1 + BUY_FEE))
                if target <= 0 or limit <= 0: continue
                shares = int(target / limit) if limit * 1000 > target else int(target / limit / 1000) * 1000
                vol20 = r.get('vol20', np.nan)
                if np.isfinite(vol20) and vol20 > 0:
                    cap = int(vol20 * 0.02 / 1000) * 1000
                    if cap < 1000: cap = int(vol20 * 0.02)
                    shares = min(shares, cap) if cap > 0 else 0
                if shares <= 0: continue
                gross = shares * limit
                post_total, post_code = (mv + created_mv + gross) / nav, gross / nav
                if post_total > TOTAL_CAP + 1e-12 or post_code > SINGLE_CAP + 1e-12: continue
                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,
                           nav, post_total, post_code)
                pending_buys.setdefault(exdate, []).append(o)
                created_mv += gross; reserved_cost += gross * (1 + BUY_FEE)
                used.add(code); slots_free -= 1
"""
    stage3 = stage3[:si] + block + stage3[ei:]
    nanchor = "nav_df, trades_df, orders_df = pd.DataFrame(nav_rows), pd.DataFrame(trade_rows), pd.DataFrame(order_rows)"
    stage3 = stage3.replace(nanchor, nanchor + "\npd.DataFrame(ai_decision_rows).to_csv('portfolio_ai_decisions.csv', index=False)", 1)
    return "# Generated from immutable R10 stage-3 for Portfolio Decision AI V1.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


def main():
    base.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR/'r10max_signals_final.pkl')
    px['code'] = px.code.astype(str).str.zfill(4)
    px = base.add_features(px)
    ev = v1.build_full_market_events(px)
    orders = pd.read_csv(BASELINE_DIR/'r10max_formal_orders.csv', dtype={'code':str})
    pred, diag, daily = expanding_oos(ev, px, orders)
    keep = ['date','code','name','pred_policy_return','pred_policy_loss_prob','is_top_candidate',
            'r7_hard','r05_hard','outside_r10','fold_train_cutoff']
    pred[keep].to_csv(RUN_ROOT/'PORTFOLIO_DECISION_AI_OOS.csv', index=False)
    diag.to_csv(RUN_ROOT/'PORTFOLIO_DECISION_AI_MODEL_DIAGNOSTICS.csv', index=False)
    daily.to_csv(RUN_ROOT/'PORTFOLIO_DECISION_AI_DAILY_AUDIT.csv', index=False)

    locked = base.LOCKED.read_text(encoding='utf-8')
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR/'runner.py'
    runner.write_text(build_runner(locked), encoding='utf-8')
    v2.run_py_verbose(runner, VARIANT_DIR, 'execution.log')

    comp = pd.DataFrame([base.summarize(BASELINE_DIR,'BASELINE_R10'), base.summarize(VARIANT_DIR,'PORTFOLIO_DECISION_AI_V1')])
    b,r = comp.iloc[0], comp.iloc[1]
    comp['cagr_delta_pp']=(comp.cagr-b.cagr)*100
    comp['dd_improvement_pp']=(comp.max_drawdown-b.max_drawdown)*100
    comp['pf_delta']=comp.pnl_profit_factor-b.pnl_profit_factor
    comp.to_csv(RUN_ROOT/'PORTFOLIO_DECISION_AI_V1_COMPARISON.csv', index=False)
    gate={'cagr_better':bool(r.cagr>b.cagr),'pf_not_worse':bool(r.pnl_profit_factor>=b.pnl_profit_factor),
          'dd_not_worse_over_1pp':bool(r.max_drawdown>=b.max_drawdown-.01)}
    gate['pass']=all(gate.values())
    (RUN_ROOT/'PORTFOLIO_DECISION_AI_V1_SUCCESS_GATE.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    dec = pd.read_csv(VARIANT_DIR/'portfolio_ai_decisions.csv') if (VARIANT_DIR/'portfolio_ai_decisions.csv').exists() else pd.DataFrame()
    audit={'decision_rows':int(len(dec)), 'keep':int((dec.action=='KEEP_R10').sum()) if len(dec) else 0,
           'replace':int((dec.action=='REPLACE').sum()) if len(dec) else 0,
           'cash':int((dec.action=='CASH').sum()) if len(dec) else 0}
    (RUN_ROOT/'PORTFOLIO_DECISION_AI_ACTION_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('=== DECISION MODEL OOS ==='); print(diag.to_string(index=False))
    print('=== ACTION AUDIT ==='); print(json.dumps(audit,indent=2))
    print('=== PORTFOLIO ==='); print(comp.to_string(index=False))
    print('=== SUCCESS GATE ==='); print(json.dumps(gate,indent=2))

if __name__ == '__main__': main()
