#!/usr/bin/env python3
"""AlphaPilot AI causal reranker V9: execution-boundary pair meta-calibration.

Why V9 exists:
- V8 produced many permitted adjacent swaps but zero portfolio change because most swaps
  occurred inside the already-selected block (e.g. rank 1/2 when two slots were free).
- The economically relevant intervention is the selection frontier: the last candidate
  that R10 would submit versus the first candidate excluded by the remaining slot count.

Preregistered contract:
- Locked R10 eligibility/execution/exits/sizing/caps/common cash remain immutable.
- Expanding-year OOS with 60-session purge remains unchanged.
- AI never deletes candidates and may move a name by at most one R10 rank.
- Meta training uses only prior-year OOS pairs that were execution-relevant under the
  locked R10 baseline: upper candidate received an ENTRY order, adjacent lower candidate
  did not, and both had a historical T+1 fill opportunity for label comparability.
- For the future test year, the meta model scores all adjacent pairs using only OOS AI
  margins. The runner may act only on the live dynamic slot boundary (slots_free-1 / slots_free).
- Base AI must prefer the lower-ranked name on BOTH expected return and failure risk.
- Meta models are fixed: LogisticRegression for P(uplift>0), Ridge for expected uplift.
- No intervention unless prior labeled boundary pairs >= 30 and both hit classes exist.
- Promotion only when meta P(hit)>=0.60 and predicted 20D uplift>0; otherwise abstain.
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
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v8.py"
spec = importlib.util.spec_from_file_location("ai_v8", src)
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)
v2 = v8.v2

RUN_ROOT = ROOT / "ai_causal_reranker_v9_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANT_DIR = RUN_ROOT / "AI_EXEC_BOUNDARY"
HORIZONS = [20, 40, 60]

v2.RUN_ROOT = RUN_ROOT
v2.BASELINE_DIR = BASELINE_DIR
v2.VARIANTS = {"AI_EXEC_BOUNDARY": VARIANT_DIR}
v2.mod.RUN_ROOT = RUN_ROOT
v2.mod.BASELINE_DIR = BASELINE_DIR
v2.mod.VARIANT_DIR = VARIANT_DIR
v2.mod.HORIZONS = HORIZONS


def all_adjacent_pairs(pred: pd.DataFrame) -> pd.DataFrame:
    x = pred.copy()
    x["orig_score"] = np.where(x.strategy.eq("R7"), x.r7_score, x.r05_score)
    rows = []
    for d, gd in x.groupby("date", sort=True):
        gd = gd.sort_values("orig_score", ascending=False).reset_index(drop=True)
        for k in range(len(gd) - 1):
            a, b = gd.iloc[k], gd.iloc[k + 1]
            vals = [a.ai_pair_return, b.ai_pair_return, a.ai_pair_fail, b.ai_pair_fail]
            if not all(np.isfinite(v) for v in vals):
                continue
            uplift = np.nan
            hit = np.nan
            if bool(a.historical_fill) and bool(b.historical_fill) and np.isfinite(a.y_ret_20) and np.isfinite(b.y_ret_20):
                uplift = float(b.y_ret_20 - a.y_ret_20)
                hit = float(uplift > 0)
            rows.append({
                "date": int(d), "year": int(d) // 10000, "upper_rank": int(k + 1),
                "upper_code": str(a.code), "upper_strategy": str(a.strategy),
                "lower_code": str(b.code), "lower_strategy": str(b.strategy),
                "ret_margin": float(b.ai_pair_return - a.ai_pair_return),
                "fail_margin": float(a.ai_pair_fail - b.ai_pair_fail),
                "uplift20": uplift, "hit": hit,
                "upper_fill": bool(a.historical_fill), "lower_fill": bool(b.historical_fill),
            })
    return pd.DataFrame(rows)


def baseline_boundary_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    orders = pd.read_csv(BASELINE_DIR / "r10max_formal_orders.csv", dtype={"code": str})
    orders["code"] = orders["code"].astype(str).str.zfill(4)
    entry = orders[(orders.side.astype(str) == "BUY") & (orders.reason.astype(str) == "ENTRY")].copy()
    selected = {}
    for r in entry.itertuples(index=False):
        selected.setdefault(int(r.signal_date), set()).add((str(r.code), str(r.strategy)))

    out = pairs.copy()
    out["baseline_upper_selected"] = [
        (str(c), str(s)) in selected.get(int(d), set())
        for d, c, s in zip(out.date, out.upper_code, out.upper_strategy)
    ]
    out["baseline_lower_selected"] = [
        (str(c), str(s)) in selected.get(int(d), set())
        for d, c, s in zip(out.date, out.lower_code, out.lower_strategy)
    ]
    out["execution_boundary"] = out.baseline_upper_selected & ~out.baseline_lower_selected
    return out


def build_runner(locked_source: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source:
        raise RuntimeError("stage3 marker missing")
    stage3 = locked_source.split(marker, 1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "_bp = pd.read_csv('../AI_BOUNDARY_PERMISSIONS.csv', dtype={'upper_code': str, 'lower_code': str})\n"
        "_bp['upper_code'] = _bp['upper_code'].astype(str).str.zfill(4)\n"
        "_bp['lower_code'] = _bp['lower_code'].astype(str).str.zfill(4)\n"
        "PAIR_ALLOW = {(int(r.date), str(r.upper_code), str(r.upper_strategy), str(r.lower_code), str(r.lower_strategy)): bool(r.meta_allow) for r in _bp.itertuples(index=False)}"
    )
    if old_load not in stage3:
        raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load, new_load, 1)

    old = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            # V9: only the live selection frontier can be changed. If N slots are free,\n            # compare original rank N (last selected) with rank N+1 (first excluded).\n            if slots_free > 0 and len(combined) > slots_free:\n                _k = int(slots_free) - 1\n                _a, _b = combined[_k], combined[_k + 1]\n                _key = (int(di), str(_a[1]), str(_a[0]), str(_b[1]), str(_b[0]))\n                if PAIR_ALLOW.get(_key, False):\n                    combined[_k], combined[_k + 1] = combined[_k + 1], combined[_k]\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated from immutable R10 stage-3; V9 execution-boundary AI tie-break.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3


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

    pairs = baseline_boundary_pairs(all_adjacent_pairs(pred))
    trainable = pairs[pairs.execution_boundary & pairs.upper_fill & pairs.lower_fill & pairs.uplift20.notna() & pairs.hit.notna()].copy()
    feature_cols = ["ret_margin", "fail_margin"]
    gate_rows, scored_rows = [], []

    for yr in [2022, 2023, 2024, 2025]:
        prior = trainable[trainable.year < yr].copy()
        cur = pairs[pairs.year == yr].copy()
        eligible = len(prior) >= 30 and prior.hit.nunique() >= 2
        enabled_n = 0
        if eligible and len(cur):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42))
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            clf.fit(prior[feature_cols], prior.hit.astype(int))
            reg.fit(prior[feature_cols], prior.uplift20.astype(float))
            cur["meta_p_hit"] = clf.predict_proba(cur[feature_cols])[:, 1]
            cur["meta_pred_uplift20"] = reg.predict(cur[feature_cols])
            cur["base_pareto"] = (cur.ret_margin > 0) & (cur.fail_margin > 0)
            cur["meta_allow"] = cur.base_pareto & (cur.meta_p_hit >= 0.60) & (cur.meta_pred_uplift20 > 0)
            enabled_n = int(cur.meta_allow.sum())
        else:
            cur["meta_p_hit"] = np.nan
            cur["meta_pred_uplift20"] = np.nan
            cur["base_pareto"] = (cur.ret_margin > 0) & (cur.fail_margin > 0)
            cur["meta_allow"] = False
        scored_rows.append(cur)
        gate_rows.append({
            "test_year": yr,
            "prior_execution_boundary_n": int(len(prior)),
            "eligible": bool(eligible),
            "allowed_adjacent_pairs": enabled_n,
        })

    scored = pd.concat(scored_rows, ignore_index=True) if scored_rows else pd.DataFrame()
    scored.to_csv(RUN_ROOT / "AI_BOUNDARY_PERMISSIONS.csv", index=False)
    pred.to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)
    pairs.to_csv(RUN_ROOT / "AI_EXECUTION_BOUNDARY_PAIRS.csv", index=False)
    trainable.to_csv(RUN_ROOT / "AI_EXECUTION_BOUNDARY_TRAINING.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(RUN_ROOT / "AI_V9_META_GATE.csv", index=False)

    locked = v2.mod.LOCKED.read_text(encoding="utf-8")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    v2.link_inputs(VARIANT_DIR)
    runner = VARIANT_DIR / "runner.py"
    runner.write_text(build_runner(locked), encoding="utf-8")
    v2.run_py_verbose(runner, VARIANT_DIR, "execution.log")

    comp = pd.DataFrame([
        v2.mod.summarize(BASELINE_DIR, "BASELINE_R10"),
        v2.mod.summarize(VARIANT_DIR, "AI_EXEC_BOUNDARY"),
    ])
    b = comp.iloc[0]; r = comp.iloc[1]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V9_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V9_COMPARISON.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")

    gate = {
        "cagr_better": bool(r.cagr > b.cagr),
        "pf_not_worse": bool(r.pnl_profit_factor >= b.pnl_profit_factor),
        "dd_not_worse_over_1pp": bool(r.max_drawdown >= b.max_drawdown - 0.01),
    }
    gate["pass"] = all(gate.values())
    (RUN_ROOT / "AI_V9_SUCCESS_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("=== V9 EXECUTION-BOUNDARY META GATE ===")
    print(pd.DataFrame(gate_rows).to_string(index=False))
    print("\n=== V9 TRAINING SAMPLE ===")
    print(trainable.groupby("year").agg(n=("hit","size"), hit_rate=("hit","mean"), mean_uplift20=("uplift20","mean")).reset_index().to_string(index=False) if len(trainable) else "NO TRAINABLE BOUNDARY PAIRS")
    print("\n=== V9 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

# CI trigger: workflow already present on branch.
