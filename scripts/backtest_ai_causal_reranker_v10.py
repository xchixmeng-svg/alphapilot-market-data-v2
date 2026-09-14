#!/usr/bin/env python3
"""AlphaPilot AI causal reranker V10: strategy-slot execution frontier.

V9 diagnosis:
- V9 learned execution-boundary permissions, but R10 truncates each strategy with
  head(r7_free)/head(r05_free) before the final combined ranking.
- Therefore the N+1 candidate was often removed before the AI boundary swap could act.

V10 preregistered contract:
- Locked R10 eligibility/exits/sizing/caps/common cash/T+1 execution remain immutable.
- Expanding-year OOS with the existing 60-session purge remains unchanged.
- AI never deletes eligibility and can move a name by at most one rank.
- Intervention is only at a real strategy slot frontier: R7 rank r7_free vs r7_free+1,
  or R05 rank r05_free vs r05_free+1, before the locked head() truncation.
- Training labels use only prior-year OOS frontier pairs where baseline submitted the
  upper candidate and did not submit the adjacent lower candidate; both need comparable
  historical T+1 fill opportunity and realized 20-session labels.
- Fixed meta models: LogisticRegression P(uplift>0) and Ridge expected uplift.
- Require >=20 prior labeled frontier pairs and both classes.
- Promotion requires base Pareto agreement (higher predicted return AND lower predicted
  failure risk), meta P(hit)>=0.60, and predicted uplift>0.
- Success gate unchanged: CAGR > R10, PF >= R10, Max DD no more than 1pp worse.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v9.py"
spec = importlib.util.spec_from_file_location("ai_v9", src)
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)
v2 = v9.v2

RUN_ROOT = ROOT / "ai_causal_reranker_v10_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_SLOT_FRONTIER"
HORIZONS = [20, 40, 60]

v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"AI_SLOT_FRONTIER": VARIANT_DIR}
v2.mod.RUN_ROOT = RUN_ROOT
v2.mod.BASELINE_DIR = BASELINE_DIR
v2.mod.VARIANT_DIR = VARIANT_DIR
v2.mod.HORIZONS = HORIZONS


def strategy_adjacent_pairs(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, strat), gd in pred.groupby(["date", "strategy"], sort=True):
        score_col = "r7_score" if strat == "R7" else "r05_score"
        gd = gd.sort_values(score_col, ascending=False).reset_index(drop=True)
        for k in range(len(gd)-1):
            a, b = gd.iloc[k], gd.iloc[k+1]
            vals = [a.ai_pair_return, b.ai_pair_return, a.ai_pair_fail, b.ai_pair_fail]
            if not all(np.isfinite(v) for v in vals):
                continue
            uplift = np.nan; hit = np.nan
            if bool(a.historical_fill) and bool(b.historical_fill) and np.isfinite(a.y_ret_20) and np.isfinite(b.y_ret_20):
                uplift = float(b.y_ret_20 - a.y_ret_20)
                hit = float(uplift > 0)
            rows.append({
                "date": int(d), "year": int(d)//10000, "strategy": str(strat),
                "upper_rank": int(k+1), "upper_code": str(a.code), "lower_code": str(b.code),
                "ret_margin": float(b.ai_pair_return-a.ai_pair_return),
                "fail_margin": float(a.ai_pair_fail-b.ai_pair_fail),
                "uplift20": uplift, "hit": hit,
                "upper_fill": bool(a.historical_fill), "lower_fill": bool(b.historical_fill),
            })
    return pd.DataFrame(rows)


def mark_baseline_frontier(pairs: pd.DataFrame) -> pd.DataFrame:
    orders = pd.read_csv(BASELINE_DIR / "r10max_formal_orders.csv", dtype={"code": str})
    orders["code"] = orders["code"].astype(str).str.zfill(4)
    entry = orders[(orders.side.astype(str)=="BUY") & (orders.reason.astype(str)=="ENTRY")].copy()
    selected = {}
    for r in entry.itertuples(index=False):
        selected.setdefault((int(r.signal_date), str(r.strategy)), set()).add(str(r.code))
    out = pairs.copy()
    out["baseline_upper_selected"] = [str(c) in selected.get((int(d),str(s)),set()) for d,s,c in zip(out.date,out.strategy,out.upper_code)]
    out["baseline_lower_selected"] = [str(c) in selected.get((int(d),str(s)),set()) for d,s,c in zip(out.date,out.strategy,out.lower_code)]
    out["execution_frontier"] = out.baseline_upper_selected & ~out.baseline_lower_selected
    return out


def build_runner(locked_source: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source:
        raise RuntimeError("stage3 marker missing")
    stage3 = locked_source.split(marker,1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "_sp = pd.read_csv('../AI_SLOT_PERMISSIONS.csv', dtype={'upper_code': str, 'lower_code': str})\n"
        "_sp['upper_code'] = _sp['upper_code'].astype(str).str.zfill(4)\n"
        "_sp['lower_code'] = _sp['lower_code'].astype(str).str.zfill(4)\n"
        "SLOT_ALLOW = {(int(r.date), str(r.strategy), str(r.upper_code), str(r.lower_code)): bool(r.meta_allow) for r in _sp.itertuples(index=False)}"
    )
    if old_load not in stage3: raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load,new_load,1)

    old = """            held = set(positions)\n            r7 = sub[sub['r7_hard'] == True].drop(index=list(held), errors='ignore').sort_values('r7_score', ascending=False).head(r7_free) if r7_exposure > 0 else sub.iloc[0:0]\n            r05 = sub[sub['r05_hard'] == True].drop(index=list(held), errors='ignore').sort_values('r05_score', ascending=False).head(r05_free)\n            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    new = """            held = set(positions)\n            _r7all = sub[sub['r7_hard'] == True].drop(index=list(held), errors='ignore').sort_values('r7_score', ascending=False) if r7_exposure > 0 else sub.iloc[0:0]\n            _r05all = sub[sub['r05_hard'] == True].drop(index=list(held), errors='ignore').sort_values('r05_score', ascending=False)\n            # V10 keeps exactly one candidate beyond each locked strategy cutoff,\n            # permits only a one-rank swap at that cutoff, then restores locked head().\n            if r7_free > 0 and len(_r7all) > r7_free:\n                _lst = list(_r7all.index)\n                _a, _b = str(_lst[r7_free-1]), str(_lst[r7_free])\n                if SLOT_ALLOW.get((int(di), 'R7', _a, _b), False):\n                    _lst[r7_free-1], _lst[r7_free] = _lst[r7_free], _lst[r7_free-1]\n                    _r7all = _r7all.loc[_lst]\n            if r05_free > 0 and len(_r05all) > r05_free:\n                _lst = list(_r05all.index)\n                _a, _b = str(_lst[r05_free-1]), str(_lst[r05_free])\n                if SLOT_ALLOW.get((int(di), 'R05', _a, _b), False):\n                    _lst[r05_free-1], _lst[r05_free] = _lst[r05_free], _lst[r05_free-1]\n                    _r05all = _r05all.loc[_lst]\n            r7 = _r7all.head(r7_free)\n            r05 = _r05all.head(r05_free)\n            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3: raise RuntimeError("candidate frontier anchor missing")
    stage3 = stage3.replace(old,new,1)
    return "# Generated from immutable R10 stage-3; V10 strategy-slot frontier AI tie-break.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


def main() -> None:
    v2.mod.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR/"r10max_signals_final.pkl")
    px["code"] = px["code"].astype(str).str.zfill(4)
    px = v2.mod.add_features(px)
    ev = v2.mod.candidate_events(px)
    ev = v2.mod.attach_labels(ev,px)
    pred,diag = v2.mod.fit_oos(ev,px)
    pred["ai_pair_return"] = pred[[f"pred_ret_{h}" for h in HORIZONS]].mean(axis=1)
    pred["ai_pair_fail"] = pred[[f"pred_fail_{h}" for h in HORIZONS]].mean(axis=1)

    pairs = mark_baseline_frontier(strategy_adjacent_pairs(pred))
    train = pairs[pairs.execution_frontier & pairs.upper_fill & pairs.lower_fill & pairs.uplift20.notna() & pairs.hit.notna()].copy()
    feature_cols=["ret_margin","fail_margin"]
    scored_rows=[]; gate_rows=[]
    for yr in [2022,2023,2024,2025]:
        prior=train[train.year<yr].copy(); cur=pairs[pairs.year==yr].copy()
        eligible=len(prior)>=20 and prior.hit.nunique()>=2
        if eligible and len(cur):
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=42))
            reg=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
            clf.fit(prior[feature_cols],prior.hit.astype(int)); reg.fit(prior[feature_cols],prior.uplift20.astype(float))
            cur["meta_p_hit"]=clf.predict_proba(cur[feature_cols])[:,1]
            cur["meta_pred_uplift20"]=reg.predict(cur[feature_cols])
            cur["base_pareto"]=(cur.ret_margin>0)&(cur.fail_margin>0)
            cur["meta_allow"]=cur.base_pareto&(cur.meta_p_hit>=0.60)&(cur.meta_pred_uplift20>0)
        else:
            cur["meta_p_hit"]=np.nan; cur["meta_pred_uplift20"]=np.nan
            cur["base_pareto"]=(cur.ret_margin>0)&(cur.fail_margin>0); cur["meta_allow"]=False
        scored_rows.append(cur)
        gate_rows.append({"test_year":yr,"prior_slot_frontier_n":int(len(prior)),"eligible":bool(eligible),"allowed_pairs":int(cur.meta_allow.sum())})
    scored=pd.concat(scored_rows,ignore_index=True)
    scored.to_csv(RUN_ROOT/"AI_SLOT_PERMISSIONS.csv",index=False)
    pairs.to_csv(RUN_ROOT/"AI_SLOT_FRONTIER_PAIRS.csv",index=False)
    train.to_csv(RUN_ROOT/"AI_SLOT_FRONTIER_TRAINING.csv",index=False)
    pred.to_csv(RUN_ROOT/"AI_OOS_PREDICTIONS.csv",index=False); diag.to_csv(RUN_ROOT/"AI_OOS_MODEL_DIAGNOSTICS.csv",index=False)
    pd.DataFrame(gate_rows).to_csv(RUN_ROOT/"AI_V10_META_GATE.csv",index=False)

    locked=v2.mod.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True,exist_ok=True); v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR/"runner.py"; runner.write_text(build_runner(locked),encoding="utf-8")
    v2.run_py_verbose(runner,VARIANT_DIR,"execution.log")

    comp=pd.DataFrame([v2.mod.summarize(BASELINE_DIR,"BASELINE_R10"),v2.mod.summarize(VARIANT_DIR,"AI_SLOT_FRONTIER")])
    b=comp.iloc[0]; r=comp.iloc[1]
    comp["cagr_delta_pp"]=(comp.cagr-b.cagr)*100
    comp["dd_improvement_pp"]=(comp.max_drawdown-b.max_drawdown)*100
    comp["pf_delta"]=comp.pnl_profit_factor-b.pnl_profit_factor
    comp["win_delta_pp"]=(comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT/"AI_CAUSAL_V10_COMPARISON.csv",index=False)
    (RUN_ROOT/"AI_CAUSAL_V10_COMPARISON.json").write_text(comp.to_json(orient="records",indent=2),encoding="utf-8")
    gate={"cagr_better":bool(r.cagr>b.cagr),"pf_not_worse":bool(r.pnl_profit_factor>=b.pnl_profit_factor),"dd_not_worse_over_1pp":bool(r.max_drawdown>=b.max_drawdown-0.01)}
    gate["pass"]=all(gate.values())
    (RUN_ROOT/"AI_V10_SUCCESS_GATE.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    print("=== V10 STRATEGY-SLOT FRONTIER GATE ==="); print(pd.DataFrame(gate_rows).to_string(index=False))
    print("\n=== V10 TRAINING ===")
    print(train.groupby(["year","strategy"]).agg(n=("hit","size"),hit_rate=("hit","mean"),mean_uplift20=("uplift20","mean")).reset_index().to_string(index=False) if len(train) else "NO TRAINING PAIRS")
    print("\n=== V10 PORTFOLIO ==="); print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ==="); print(json.dumps(gate,indent=2))

if __name__=="__main__":
    main()
