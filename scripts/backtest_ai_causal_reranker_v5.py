#!/usr/bin/env python3
"""AlphaPilot AI causal reranker V5: reliability-gated Pareto adjacent tie-break.

Preregistered after V4 failure:
- Locked R10 eligibility/execution/exits/sizing/caps/common cash remain immutable.
- Expanding-year OOS with 60-session purge remains unchanged.
- AI never deletes candidates and can move a name by at most one R10 rank.
- AI is disabled unless its horizon has at least TWO prior OOS test years and,
  using only those prior OOS folds, mean Spearman > 0 and mean failure AUC >= 0.56.
- Eligible horizons are selected causally per test year; no current/future-year
  diagnostic is used to decide whether AI is enabled.
- Within an adjacent R10 pair, the lower-ranked candidate is promoted only when
  it Pareto-dominates the higher-ranked candidate on BOTH predicted return
  (higher) and predicted <=-5% failure probability (lower). Otherwise abstain.
- Success gate: CAGR > R10, PF >= R10, and Max DD no more than 1pp worse.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v2.py"
spec = importlib.util.spec_from_file_location("ai_v2", src)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

RUN_ROOT = ROOT / "ai_causal_reranker_v5_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_RELIABILITY_PARETO"
HORIZONS = [20, 40, 60]

# Redirect the imported causal pipeline to V5 outputs.
v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"AI_RELIABILITY_PARETO": VARIANT_DIR}
v2.mod.RUN_ROOT = RUN_ROOT
v2.mod.BASELINE_DIR = BASELINE_DIR
v2.mod.VARIANT_DIR = VARIANT_DIR
v2.mod.HORIZONS = HORIZONS


def build_runner(locked_source: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source:
        raise RuntimeError("stage3 marker missing")
    stage3 = locked_source.split(marker, 1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "_ai = pd.read_csv('../AI_OOS_PREDICTIONS.csv', dtype={'code': str})\n"
        "_ai['code'] = _ai['code'].astype(str).str.zfill(4)\n"
        "AI_MAP = {(int(r.date), str(r.code), str(r.strategy)): (int(r.ai_enabled), float(r.ai_rel_return), float(r.ai_rel_fail)) for r in _ai.itertuples(index=False)}"
    )
    if old_load not in stage3:
        raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load, new_load, 1)

    old = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            if len(combined) > 1:\n                _bounded = []\n                for _k in range(0, len(combined), 2):\n                    _pair = combined[_k:_k+2]\n                    if len(_pair) == 2:\n                        _m0 = AI_MAP.get((int(di), str(_pair[0][1]), str(_pair[0][0])), (0, np.nan, np.nan))\n                        _m1 = AI_MAP.get((int(di), str(_pair[1][1]), str(_pair[1][0])), (0, np.nan, np.nan))\n                        # Causal abstention + Pareto dominance: promote lower R10 rank only\n                        # if the validated AI horizon(s) agree on better return AND lower failure risk.\n                        if _m0[0] == 1 and _m1[0] == 1 and np.isfinite(_m0[1]) and np.isfinite(_m1[1]) and np.isfinite(_m0[2]) and np.isfinite(_m1[2]):\n                            if (_m1[1] > _m0[1]) and (_m1[2] < _m0[2]):\n                                _pair = [_pair[1], _pair[0]]\n                    _bounded.extend(_pair)\n                combined = _bounded\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated from immutable R10 stage-3; V5 reliability-gated Pareto adjacent tie-break.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


def main() -> None:
    v2.mod.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR / "r10max_signals_final.pkl")
    px["code"] = px["code"].astype(str).str.zfill(4)
    px = v2.mod.add_features(px)
    ev = v2.mod.candidate_events(px)
    ev = v2.mod.attach_labels(ev, px)
    pred, diag = v2.mod.fit_oos(ev, px)

    # Reliability decisions are made from PREVIOUS OOS folds only.
    reliability_rows = []
    year_horizons = {}
    for yr in [2022, 2023, 2024, 2025]:
        prev = diag[diag["test_year"] < yr].copy()
        eligible = []
        for h in HORIZONS:
            gh = prev[prev["horizon"] == h]
            n = len(gh)
            mean_s = float(gh["spearman_return"].mean()) if n else np.nan
            mean_a = float(gh["failure_auc"].mean()) if n else np.nan
            ok = bool(n >= 2 and np.isfinite(mean_s) and np.isfinite(mean_a) and mean_s > 0.0 and mean_a >= 0.56)
            reliability_rows.append({"test_year": yr, "horizon": h, "prior_oos_folds": n,
                                     "prior_mean_spearman": mean_s, "prior_mean_failure_auc": mean_a,
                                     "eligible": ok})
            if ok:
                eligible.append(h)
        year_horizons[yr] = eligible

    pred["ai_enabled"] = 0
    pred["ai_rel_return"] = np.nan
    pred["ai_rel_fail"] = np.nan
    for yr, hs in year_horizons.items():
        if not hs:
            continue
        m = (pred["date"] // 10000) == yr
        pred.loc[m, "ai_enabled"] = 1
        pred.loc[m, "ai_rel_return"] = pred.loc[m, [f"pred_ret_{h}" for h in hs]].mean(axis=1)
        pred.loc[m, "ai_rel_fail"] = pred.loc[m, [f"pred_fail_{h}" for h in hs]].mean(axis=1)

    keep = ["date","code","strategy","ai_enabled","ai_rel_return","ai_rel_fail","fold_train_cutoff"]
    keep += [f"pred_ret_{h}" for h in HORIZONS] + [f"pred_fail_{h}" for h in HORIZONS]
    pred[keep].to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(RUN_ROOT / "AI_RELIABILITY_GATE.csv", index=False)

    locked = v2.mod.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR / "runner.py"
    runner.write_text(build_runner(locked), encoding="utf-8")
    v2.run_py_verbose(runner, VARIANT_DIR, "execution.log")

    rows = [v2.mod.summarize(BASELINE_DIR, "BASELINE_R10"), v2.mod.summarize(VARIANT_DIR, "AI_RELIABILITY_PARETO")]
    comp = pd.DataFrame(rows)
    b = comp.iloc[0]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V5_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V5_COMPARISON.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")

    r = comp.iloc[1]
    gate = {
        "cagr_better": bool(r.cagr > b.cagr),
        "pf_not_worse": bool(r.pnl_profit_factor >= b.pnl_profit_factor),
        "dd_not_worse_over_1pp": bool(r.max_drawdown >= b.max_drawdown - 0.01),
    }
    gate["pass"] = all(gate.values())
    (RUN_ROOT / "AI_V5_SUCCESS_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("=== V5 RELIABILITY GATE ===")
    print(pd.DataFrame(reliability_rows).to_string(index=False))
    print("\n=== V5 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
