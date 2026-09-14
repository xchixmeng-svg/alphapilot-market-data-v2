#!/usr/bin/env python3
"""AlphaPilot AI causal reranker V7: high-confidence pairwise abstention.

Preregistered after V6 produced zero interventions:
- Locked R10 eligibility/execution/exits/sizing/caps/common cash remain immutable.
- Expanding-year OOS with 60-session purge remains unchanged.
- AI never deletes candidates and can move a name by at most one R10 rank.
- No annual blanket enable/disable. Permission is event-level and based only on prior OOS
  prediction-margin distributions plus prior realized pairwise skill.
- For each test year, thresholds are the 90th percentile of prior OOS Pareto promotion
  margins: return advantage and failure-risk reduction. The prior high-confidence subset
  must have >=20 decisions, hit-rate >=55%, and mean realized 20-session uplift >0.
- Current-year promotion occurs only when BOTH margins clear those frozen prior-year
  thresholds; otherwise abstain to original R10 order.
- Success gate: CAGR > R10, PF >= R10, Max DD no more than 1pp worse.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v6.py"
spec = importlib.util.spec_from_file_location("ai_v6", src)
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)
v2 = v6.v2

RUN_ROOT = ROOT / "ai_causal_reranker_v7_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_HIGHCONF_PAIR"
HORIZONS = [20, 40, 60]

v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"AI_HIGHCONF_PAIR": VARIANT_DIR}
v2.mod.RUN_ROOT = RUN_ROOT
v2.mod.BASELINE_DIR = BASELINE_DIR
v2.mod.VARIANT_DIR = VARIANT_DIR
v2.mod.HORIZONS = HORIZONS


def candidate_pair_rows(pred: pd.DataFrame) -> pd.DataFrame:
    x = pred.copy()
    x["orig_score"] = np.where(x["strategy"].eq("R7"), x["r7_score"], x["r05_score"])
    rows = []
    for d, gd in x.groupby("date", sort=True):
        gd = gd.sort_values("orig_score", ascending=False).reset_index(drop=True)
        for k in range(0, len(gd), 2):
            if k + 1 >= len(gd):
                continue
            a, b = gd.iloc[k], gd.iloc[k+1]
            vals = [a.ai_pair_return, b.ai_pair_return, a.ai_pair_fail, b.ai_pair_fail]
            if not all(np.isfinite(v) for v in vals):
                continue
            ret_margin = float(b.ai_pair_return - a.ai_pair_return)
            fail_margin = float(a.ai_pair_fail - b.ai_pair_fail)
            if ret_margin <= 0 or fail_margin <= 0:
                continue
            uplift = np.nan
            hit = np.nan
            if bool(a.historical_fill) and bool(b.historical_fill) and np.isfinite(a.y_ret_20) and np.isfinite(b.y_ret_20):
                uplift = float(b.y_ret_20 - a.y_ret_20)
                hit = float(uplift > 0)
            rows.append({"date": int(d), "year": int(d)//10000,
                         "upper_code": str(a.code), "lower_code": str(b.code),
                         "ret_margin": ret_margin, "fail_margin": fail_margin,
                         "uplift20": uplift, "hit": hit})
    return pd.DataFrame(rows)


def build_runner(locked_source: str) -> str:
    return v6.build_runner(locked_source).replace(
        "# Generated from immutable R10 stage-3; V6 pairwise-skill-gated adjacent AI tie-break.",
        "# Generated from immutable R10 stage-3; V7 high-confidence adjacent AI tie-break."
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

    pairs = candidate_pair_rows(pred)
    gate_rows = []
    pair_permissions = {}
    for yr in [2022, 2023, 2024, 2025]:
        prior = pairs[(pairs["year"] < yr) & pairs["uplift20"].notna()].copy() if not pairs.empty else pairs.copy()
        if len(prior):
            ret_thr = float(prior["ret_margin"].quantile(0.90))
            fail_thr = float(prior["fail_margin"].quantile(0.90))
            hc = prior[(prior["ret_margin"] >= ret_thr) & (prior["fail_margin"] >= fail_thr)].copy()
        else:
            ret_thr = np.nan; fail_thr = np.nan; hc = prior
        n = int(len(hc))
        hit = float(hc["hit"].mean()) if n else np.nan
        uplift = float(hc["uplift20"].mean()) if n else np.nan
        skilled = bool(n >= 20 and np.isfinite(hit) and np.isfinite(uplift) and hit >= 0.55 and uplift > 0)
        gate_rows.append({"test_year": yr, "prior_highconf_n": n, "prior_highconf_hit_rate": hit,
                          "prior_highconf_mean_uplift20": uplift, "ret_margin_p90": ret_thr,
                          "fail_margin_p90": fail_thr, "skilled": skilled})
        cur = pairs[pairs["year"] == yr]
        if skilled:
            allowed = cur[(cur["ret_margin"] >= ret_thr) & (cur["fail_margin"] >= fail_thr)]
            for r in allowed.itertuples(index=False):
                pair_permissions[(int(r.date), str(r.upper_code), str(r.lower_code))] = True

    pred["ai_enabled"] = 0
    # Enable only the two members of explicitly permitted adjacent pairs.
    x = pred.copy()
    x["orig_score"] = np.where(x["strategy"].eq("R7"), x["r7_score"], x["r05_score"])
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
    pred["ai_enabled"] = [int((int(d), str(c), str(s)) in enable_keys) for d,c,s in zip(pred.date,pred.code,pred.strategy)]

    keep = ["date","code","strategy","r7_score","r05_score","historical_fill","y_ret_20",
            "ai_enabled","ai_pair_return","ai_pair_fail","fold_train_cutoff"]
    keep += [f"pred_ret_{h}" for h in HORIZONS] + [f"pred_fail_{h}" for h in HORIZONS]
    pred[keep].to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)
    pairs.to_csv(RUN_ROOT / "AI_PAIRWISE_DECISIONS.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(RUN_ROOT / "AI_HIGHCONF_GATE.csv", index=False)

    locked = v2.mod.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR / "runner.py"
    runner.write_text(build_runner(locked), encoding="utf-8")
    v2.run_py_verbose(runner, VARIANT_DIR, "execution.log")

    comp = pd.DataFrame([v2.mod.summarize(BASELINE_DIR, "BASELINE_R10"), v2.mod.summarize(VARIANT_DIR, "AI_HIGHCONF_PAIR")])
    b = comp.iloc[0]; r = comp.iloc[1]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V7_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V7_COMPARISON.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")
    gate = {"cagr_better": bool(r.cagr > b.cagr), "pf_not_worse": bool(r.pnl_profit_factor >= b.pnl_profit_factor),
            "dd_not_worse_over_1pp": bool(r.max_drawdown >= b.max_drawdown - 0.01)}
    gate["pass"] = all(gate.values())
    (RUN_ROOT / "AI_V7_SUCCESS_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("=== V7 HIGH-CONFIDENCE GATE ===")
    print(pd.DataFrame(gate_rows).to_string(index=False))
    print("\n=== V7 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
