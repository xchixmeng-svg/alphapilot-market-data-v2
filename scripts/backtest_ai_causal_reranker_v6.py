#!/usr/bin/env python3
"""AlphaPilot AI causal reranker V6: prior-OOS pairwise-skill-gated adjacent tie-break.

Preregistered after V5 failure:
- Locked R10 eligibility/execution/exits/sizing/caps/common cash remain immutable.
- Expanding-year OOS with 60-session purge remains unchanged.
- AI never deletes candidates and can move a name by at most one R10 rank.
- AI permission is based on the exact intervention it will make: prior OOS adjacent-pair
  decisions, not generic Spearman/AUC. For each future test year, AI is enabled only if
  prior OOS years contain >=30 Pareto promotion decisions, pairwise hit-rate >=55%, and
  mean realized 20-session uplift > 0. No current/future-year outcomes enter the gate.
- When enabled, lower R10 rank is promoted only if its mean 20/40/60 predicted return is
  higher AND its mean predicted <=-5% failure probability is lower. Otherwise abstain.
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

RUN_ROOT = ROOT / "ai_causal_reranker_v6_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_PAIRWISE_SKILL"
HORIZONS = [20, 40, 60]

v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"AI_PAIRWISE_SKILL": VARIANT_DIR}
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
        "AI_MAP = {(int(r.date), str(r.code), str(r.strategy)): (int(r.ai_enabled), float(r.ai_pair_return), float(r.ai_pair_fail)) for r in _ai.itertuples(index=False)}"
    )
    if old_load not in stage3:
        raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load, new_load, 1)
    old = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            if len(combined) > 1:\n                _bounded = []\n                for _k in range(0, len(combined), 2):\n                    _pair = combined[_k:_k+2]\n                    if len(_pair) == 2:\n                        _m0 = AI_MAP.get((int(di), str(_pair[0][1]), str(_pair[0][0])), (0, np.nan, np.nan))\n                        _m1 = AI_MAP.get((int(di), str(_pair[1][1]), str(_pair[1][0])), (0, np.nan, np.nan))\n                        if _m0[0] == 1 and _m1[0] == 1 and np.isfinite(_m0[1]) and np.isfinite(_m1[1]) and np.isfinite(_m0[2]) and np.isfinite(_m1[2]):\n                            if (_m1[1] > _m0[1]) and (_m1[2] < _m0[2]):\n                                _pair = [_pair[1], _pair[0]]\n                    _bounded.extend(_pair)\n                combined = _bounded\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated from immutable R10 stage-3; V6 pairwise-skill-gated adjacent AI tie-break.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


def pairwise_decisions(pred: pd.DataFrame) -> pd.DataFrame:
    x = pred.copy()
    x["orig_score"] = np.where(x["strategy"].eq("R7"), x["r7_score"], x["r05_score"])
    rows = []
    for d, gd in x.groupby("date", sort=True):
        gd = gd.sort_values("orig_score", ascending=False).reset_index(drop=True)
        for k in range(0, len(gd), 2):
            if k + 1 >= len(gd):
                continue
            a, b = gd.iloc[k], gd.iloc[k+1]
            if not (bool(a.historical_fill) and bool(b.historical_fill)):
                continue
            vals = [a.ai_pair_return, b.ai_pair_return, a.ai_pair_fail, b.ai_pair_fail, a.y_ret_20, b.y_ret_20]
            if not all(np.isfinite(v) for v in vals):
                continue
            if (b.ai_pair_return > a.ai_pair_return) and (b.ai_pair_fail < a.ai_pair_fail):
                uplift = float(b.y_ret_20 - a.y_ret_20)
                rows.append({"date": int(d), "year": int(d)//10000, "upper_code": str(a.code), "lower_code": str(b.code),
                             "uplift20": uplift, "hit": float(uplift > 0)})
    return pd.DataFrame(rows)


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

    decisions = pairwise_decisions(pred)
    gate_rows = []
    enabled_by_year = {}
    for yr in [2022, 2023, 2024, 2025]:
        prev = decisions[decisions["year"] < yr] if not decisions.empty else decisions
        n = int(len(prev))
        hit = float(prev["hit"].mean()) if n else np.nan
        uplift = float(prev["uplift20"].mean()) if n else np.nan
        enabled = bool(n >= 30 and np.isfinite(hit) and np.isfinite(uplift) and hit >= 0.55 and uplift > 0.0)
        enabled_by_year[yr] = enabled
        gate_rows.append({"test_year": yr, "prior_pair_decisions": n, "prior_hit_rate": hit, "prior_mean_uplift20": uplift, "enabled": enabled})

    pred["ai_enabled"] = pred["date"].map(lambda d: int(enabled_by_year.get(int(d)//10000, False)))
    keep = ["date","code","strategy","r7_score","r05_score","historical_fill","y_ret_20","ai_enabled","ai_pair_return","ai_pair_fail","fold_train_cutoff"]
    keep += [f"pred_ret_{h}" for h in HORIZONS] + [f"pred_fail_{h}" for h in HORIZONS]
    pred[keep].to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)
    decisions.to_csv(RUN_ROOT / "AI_PAIRWISE_DECISIONS.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(RUN_ROOT / "AI_PAIRWISE_GATE.csv", index=False)

    locked = v2.mod.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR / "runner.py"
    runner.write_text(build_runner(locked), encoding="utf-8")
    v2.run_py_verbose(runner, VARIANT_DIR, "execution.log")

    comp = pd.DataFrame([v2.mod.summarize(BASELINE_DIR, "BASELINE_R10"), v2.mod.summarize(VARIANT_DIR, "AI_PAIRWISE_SKILL")])
    b = comp.iloc[0]; r = comp.iloc[1]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V6_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V6_COMPARISON.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")
    gate = {"cagr_better": bool(r.cagr > b.cagr), "pf_not_worse": bool(r.pnl_profit_factor >= b.pnl_profit_factor),
            "dd_not_worse_over_1pp": bool(r.max_drawdown >= b.max_drawdown - 0.01)}
    gate["pass"] = all(gate.values())
    (RUN_ROOT / "AI_V6_SUCCESS_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("=== V6 PAIRWISE GATE ===")
    print(pd.DataFrame(gate_rows).to_string(index=False))
    print("\n=== V6 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
