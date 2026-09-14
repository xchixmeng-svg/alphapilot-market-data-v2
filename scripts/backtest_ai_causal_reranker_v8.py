#!/usr/bin/env python3
"""AlphaPilot AI causal reranker V8: pairwise meta-calibration.

Preregistered after V7 abstained everywhere because hard high-confidence sample gates
were too sparse. This version does not tune a percentile on 2022-2025 outcomes.
Instead it learns, from prior OOS pair decisions only, whether a proposed adjacent
promotion is likely to beat the original R10 ordering.

Contract:
- Locked R10 eligibility/execution/exits/sizing/caps/common cash immutable.
- Expanding-year OOS with 60-session purge unchanged.
- AI never deletes candidates and can move a name by at most one R10 rank.
- Base AI must already prefer lower-ranked name on BOTH expected return and failure risk.
- Meta models are trained only on prior-year OOS pair decisions with realized 20D uplift.
- Fixed models: LogisticRegression for P(uplift>0), Ridge for expected uplift.
- No intervention unless prior labeled pairs >=50 and both hit classes are present.
- Promotion only when meta P(hit)>=0.60 AND predicted uplift>0; otherwise abstain.
- Success gate: CAGR > R10, PF >= R10, Max DD no more than 1pp worse.
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
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v7.py"
spec = importlib.util.spec_from_file_location("ai_v7", src)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
v6 = v7.v6
v2 = v7.v2

RUN_ROOT = ROOT / "ai_causal_reranker_v8_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_META_PAIR"
HORIZONS = [20, 40, 60]

v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"AI_META_PAIR": VARIANT_DIR}
v2.mod.RUN_ROOT = RUN_ROOT
v2.mod.BASELINE_DIR = BASELINE_DIR
v2.mod.VARIANT_DIR = VARIANT_DIR
v2.mod.HORIZONS = HORIZONS


def pair_rows(pred: pd.DataFrame) -> pd.DataFrame:
    return v7.candidate_pair_rows(pred)


def build_runner(locked_source: str) -> str:
    return v6.build_runner(locked_source).replace(
        "# Generated from immutable R10 stage-3; V6 pairwise-skill-gated adjacent AI tie-break.",
        "# Generated from immutable R10 stage-3; V8 pairwise meta-calibrated adjacent AI tie-break."
    )


def main() -> None:
    v2.mod.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR / "r10max_signals_final.pkl")
    px["code"] = px["code"].astype(str).str.zfill(4)
    px = v2.mod.add_features(px)
    ev = v2.mod.candidate_events(px)
    ev = v2.mod.attach_labels(ev, px)
    pred, diag = v2.mod.fit_oos(ev, px)
    pred["ai_pair_return"] = pred[[f"pred_ret_{h}" for h in HORIZONS]].mean(axis=1)
    pred["ai_pair_fail"] = pred[[f"pred_fail_{h}" for h in HORIZONS]].mean(axis=1)

    pairs = pair_rows(pred)
    feature_cols = ["ret_margin", "fail_margin"]
    pair_permissions = {}
    gate_rows = []
    scored_rows = []

    for yr in [2022, 2023, 2024, 2025]:
        prior = pairs[(pairs.year < yr) & pairs.uplift20.notna() & pairs.hit.notna()].copy()
        cur = pairs[pairs.year == yr].copy()
        eligible = len(prior) >= 50 and prior.hit.nunique() >= 2
        auc_proxy = np.nan
        enabled_n = 0
        if eligible and len(cur):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            clf.fit(prior[feature_cols], prior.hit.astype(int))
            reg.fit(prior[feature_cols], prior.uplift20.astype(float))
            p_hit = clf.predict_proba(cur[feature_cols])[:, 1]
            p_uplift = reg.predict(cur[feature_cols])
            cur["meta_p_hit"] = p_hit
            cur["meta_pred_uplift20"] = p_uplift
            cur["meta_allow"] = (cur.meta_p_hit >= 0.60) & (cur.meta_pred_uplift20 > 0)
            enabled_n = int(cur.meta_allow.sum())
            for r in cur[cur.meta_allow].itertuples(index=False):
                pair_permissions[(int(r.date), str(r.upper_code), str(r.lower_code))] = True
            scored_rows.append(cur)
        gate_rows.append({"test_year":yr,"prior_pair_n":int(len(prior)),"eligible":bool(eligible),"enabled_pairs":enabled_n})

    pred["ai_enabled"] = 0
    x = pred.copy()
    x["orig_score"] = np.where(x.strategy.eq("R7"), x.r7_score, x.r05_score)
    enable_keys = set()
    for d, gd in x.groupby("date", sort=True):
        gd = gd.sort_values("orig_score", ascending=False).reset_index(drop=True)
        for k in range(0, len(gd), 2):
            if k + 1 >= len(gd):
                continue
            a, b = gd.iloc[k], gd.iloc[k+1]
            if pair_permissions.get((int(d), str(a.code), str(b.code)), False):
                enable_keys.add((int(d), str(a.code), str(a.strategy)))
                enable_keys.add((int(d), str(b.code), str(b.strategy)))
    pred["ai_enabled"] = [int((int(d),str(c),str(s)) in enable_keys) for d,c,s in zip(pred.date,pred.code,pred.strategy)]

    keep=["date","code","strategy","r7_score","r05_score","historical_fill","y_ret_20","ai_enabled","ai_pair_return","ai_pair_fail","fold_train_cutoff"]
    keep += [f"pred_ret_{h}" for h in HORIZONS] + [f"pred_fail_{h}" for h in HORIZONS]
    pred[keep].to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)
    pairs.to_csv(RUN_ROOT / "AI_PAIRWISE_DECISIONS.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(RUN_ROOT / "AI_META_GATE.csv", index=False)
    if scored_rows:
        pd.concat(scored_rows, ignore_index=True).to_csv(RUN_ROOT / "AI_META_SCORED_PAIRS.csv", index=False)
    else:
        pd.DataFrame().to_csv(RUN_ROOT / "AI_META_SCORED_PAIRS.csv", index=False)

    locked=v2.mod.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner=VARIANT_DIR / "runner.py"
    runner.write_text(build_runner(locked), encoding="utf-8")
    v2.run_py_verbose(runner, VARIANT_DIR, "execution.log")

    comp=pd.DataFrame([v2.mod.summarize(BASELINE_DIR,"BASELINE_R10"),v2.mod.summarize(VARIANT_DIR,"AI_META_PAIR")])
    b=comp.iloc[0]; r=comp.iloc[1]
    comp["cagr_delta_pp"]=(comp.cagr-b.cagr)*100
    comp["dd_improvement_pp"]=(comp.max_drawdown-b.max_drawdown)*100
    comp["pf_delta"]=comp.pnl_profit_factor-b.pnl_profit_factor
    comp["win_delta_pp"]=(comp.win_rate-b.win_rate)*100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V8_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V8_COMPARISON.json").write_text(comp.to_json(orient="records",indent=2),encoding="utf-8")
    gate={"cagr_better":bool(r.cagr>b.cagr),"pf_not_worse":bool(r.pnl_profit_factor>=b.pnl_profit_factor),"dd_not_worse_over_1pp":bool(r.max_drawdown>=b.max_drawdown-0.01)}
    gate["pass"]=all(gate.values())
    (RUN_ROOT / "AI_V8_SUCCESS_GATE.json").write_text(json.dumps(gate,indent=2),encoding="utf-8")
    print("=== V8 META GATE ===")
    print(pd.DataFrame(gate_rows).to_string(index=False))
    print("\n=== V8 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate,indent=2))

if __name__ == "__main__":
    main()
