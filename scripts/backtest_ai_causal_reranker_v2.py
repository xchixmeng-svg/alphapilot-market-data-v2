#!/usr/bin/env python3
"""AlphaPilot AI causal reranker v2: mid-horizon rank-only overlay.

Preregistered research contract:
- Locked R10 MAX engine is immutable.
- Same T+1 execution, fees/tax/slippage, integer shares, common cash, exits/caps.
- Expanding-year OOS only with 60-session purge.
- AI uses only 20/40/60-session targets; 5/10-session targets are excluded after V1 showed weak/noisy OOS rank signal.
- No candidate is deleted by AI. R10 eligibility remains authoritative.
- Two structural variants are tested:
  1) AI_MID_RANK: reorder otherwise-eligible R10 candidates by OOS mid-horizon AI score.
  2) AI_MID_BLEND: 50/50 rank blend of original R10 rank and AI rank.
- AI score is the equal-weight mean predicted net return across 20/40/60 sessions
  minus 5% times the equal-weight mean probability of <= -5% outcome.
- 2021 remains locked R10 because no prior training year exists.
"""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT / "scripts" / "backtest_ai_causal_reranker_v1.py"
spec = importlib.util.spec_from_file_location("ai_v1_base", BASE_SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RUN_ROOT = ROOT / "ai_causal_reranker_v2_results"
BASELINE_DIR = RUN_ROOT / "baseline"
VARIANTS = {
    "AI_MID_RANK": RUN_ROOT / "AI_MID_RANK",
    "AI_MID_BLEND": RUN_ROOT / "AI_MID_BLEND",
}
HORIZONS = [20, 40, 60]

mod.HORIZONS = HORIZONS
mod.RUN_ROOT = RUN_ROOT
mod.BASELINE_DIR = BASELINE_DIR
mod.VARIANT_DIR = VARIANTS["AI_MID_RANK"]
mod.ACCEPT_MIN_PRED_RETURN = -999.0
mod.ACCEPT_MAX_FAIL_PROB = 1.0

def run_py_verbose(script: Path, cwd: Path, log_name: str) -> None:
    p = subprocess.run([sys.executable, str(script)], cwd=cwd, text=True, capture_output=True)
    text = p.stdout + "\n--- STDERR ---\n" + p.stderr
    (cwd / log_name).write_text(text, encoding="utf-8")
    print(f"[{cwd.name}] returncode={p.returncode}")
    if p.stdout:
        print("\n".join(p.stdout.splitlines()[-24:]))
    if p.stderr:
        print("--- STDERR TAIL ---")
        print("\n".join(p.stderr.splitlines()[-40:]))
    if p.returncode:
        raise RuntimeError(f"runner failed: {script}")

def link_inputs(dst: Path) -> None:
    hist = ROOT / "data" / "history" / "2020-2025"
    ref = ROOT / "data" / "reference"
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "institutional_2020_2025.parquet",
        "ohlcv_2020.parquet", "ohlcv_2021.parquet", "ohlcv_2022.parquet",
        "ohlcv_2023.parquet", "ohlcv_2024.parquet", "ohlcv_2025.parquet",
    ]:
        out = dst / name
        if not out.exists():
            out.symlink_to((hist / name).resolve())
    out = dst / "official_corporate_actions_2020_2025.csv"
    if not out.exists():
        out.symlink_to((ref / "official_corporate_actions_2020_2025.csv").resolve())

mod.run_py = run_py_verbose
mod.link_inputs = link_inputs
_orig_candidate_events = mod.candidate_events

def candidate_events_fixed(px: pd.DataFrame) -> pd.DataFrame:
    ev = _orig_candidate_events(px)
    dates = sorted(int(x) for x in px.date.unique().tolist())
    didx = {d: i for i, d in enumerate(dates)}
    unique = ev[["date", "code"]].drop_duplicates().sort_values(["date", "code"]).copy()
    p5_map, p10_map, history = {}, {}, {}
    for r in unique.itertuples(index=False):
        d = int(r.date); c = str(r.code); i = didx[d]
        hist = history.setdefault(c, [])
        p5_map[(d,c)] = sum(1 for x in hist if i - x < 5)
        p10_map[(d,c)] = sum(1 for x in hist if i - x < 10)
        hist.append(i)
    ev["persist5_prior"] = [p5_map[(int(d), str(c))] for d,c in zip(ev.date, ev.code)]
    ev["persist10_prior"] = [p10_map[(int(d), str(c))] for d,c in zip(ev.date, ev.code)]
    return ev
mod.candidate_events = candidate_events_fixed

def build_runner(locked_source: str, mode: str) -> str:
    marker = "# ============================================================\n# 步驟三：組合層回測引擎\n# ============================================================\n\n"
    if marker not in locked_source:
        raise RuntimeError("stage3 marker missing")
    stage3 = locked_source.split(marker, 1)[1]
    old_load = "px = pd.read_pickle('r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)"
    new_load = (
        "px = pd.read_pickle('../baseline/r10max_signals_final.pkl').sort_values(['date', 'code']).reset_index(drop=True)\n"
        "_ai = pd.read_csv('../AI_OOS_PREDICTIONS.csv', dtype={'code': str})\n"
        "_ai['code'] = _ai['code'].astype(str).str.zfill(4)\n"
        "AI_MAP = {(int(r.date), str(r.code), str(r.strategy)): float(r.ai_mid_score) for r in _ai.itertuples(index=False)}"
    )
    if old_load not in stage3:
        raise RuntimeError("signal load anchor missing")
    stage3 = stage3.replace(old_load, new_load, 1)

    old = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if mode == "rank":
        new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            if int(di) >= 20220101:\n                combined.sort(key=lambda z: -AI_MAP.get((int(di), str(z[1]), str(z[0])), -1e18))\n            else:\n                combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    elif mode == "blend":
        new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            if int(di) >= 20220101 and len(combined) > 1:\n                _orig = sorted(range(len(combined)), key=lambda j: -(combined[j][2]['r7_score'] if combined[j][0] == 'R7' else combined[j][2]['r05_score']))\n                _air = sorted(range(len(combined)), key=lambda j: -AI_MAP.get((int(di), str(combined[j][1]), str(combined[j][0])), -1e18))\n                _orank = {j:i for i,j in enumerate(_orig)}\n                _arank = {j:i for i,j in enumerate(_air)}\n                combined = [combined[j] for j in sorted(range(len(combined)), key=lambda j: (_orank[j] + _arank[j], _orank[j]))]\n            else:\n                combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    else:
        raise ValueError(mode)
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated from immutable R10 stage-3; AI changes ranking only.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3

def main() -> None:
    mod.prepare_baseline()
    px = pd.read_pickle(BASELINE_DIR / "r10max_signals_final.pkl")
    px["code"] = px["code"].astype(str).str.zfill(4)
    px = mod.add_features(px)
    ev = mod.candidate_events(px)
    ev = mod.attach_labels(ev, px)
    pred, diag = mod.fit_oos(ev, px)
    pred["ai_mid_score"] = (
        pred[[f"pred_ret_{h}" for h in HORIZONS]].mean(axis=1)
        - 0.05 * pred[[f"pred_fail_{h}" for h in HORIZONS]].mean(axis=1)
    )
    pred["ai_accept"] = 1
    keep = ["date","code","strategy","best_horizon","pred_net_return","pred_fail_prob","ai_score","ai_mid_score",
            "ai_accept","fold_train_cutoff"] + [f"pred_ret_{h}" for h in HORIZONS] + [f"pred_fail_{h}" for h in HORIZONS]
    pred[keep].to_csv(RUN_ROOT / "AI_OOS_PREDICTIONS.csv", index=False)
    diag.to_csv(RUN_ROOT / "AI_OOS_MODEL_DIAGNOSTICS.csv", index=False)

    evalp = pred[pred.historical_fill].copy()
    drows = []
    for yr, gy in evalp.groupby(evalp.date // 10000):
        top_rets, bot_rets = [], []
        for _, gd in gy.groupby("date"):
            if len(gd) < 2:
                continue
            gd = gd.sort_values("ai_mid_score", ascending=False)
            k = max(1, len(gd)//2)
            top_rets.extend(gd.head(k).y_ret_20.dropna().tolist())
            bot_rets.extend(gd.tail(k).y_ret_20.dropna().tolist())
        drows.append({
            "year": int(yr), "top_n": len(top_rets), "bottom_n": len(bot_rets),
            "top20_mean": float(np.mean(top_rets)) if top_rets else np.nan,
            "bottom20_mean": float(np.mean(bot_rets)) if bot_rets else np.nan,
            "spread20": (float(np.mean(top_rets)) - float(np.mean(bot_rets))) if top_rets and bot_rets else np.nan,
            "top20_win": float(np.mean(np.array(top_rets)>0)) if top_rets else np.nan,
            "bottom20_win": float(np.mean(np.array(bot_rets)>0)) if bot_rets else np.nan,
        })
    pd.DataFrame(drows).to_csv(RUN_ROOT / "AI_V2_SELECTION_DIAGNOSTICS.csv", index=False)

    locked = mod.LOCKED.read_text(encoding="utf-8")
    for name, outdir in VARIANTS.items():
        outdir.mkdir(parents=True, exist_ok=True)
        link_inputs(outdir)
        mode = "rank" if name == "AI_MID_RANK" else "blend"
        runner = outdir / "runner.py"
        runner.write_text(build_runner(locked, mode), encoding="utf-8")
        run_py_verbose(runner, outdir, "execution.log")

    rows = [mod.summarize(BASELINE_DIR, "BASELINE_R10")]
    rows += [mod.summarize(path, name) for name, path in VARIANTS.items()]
    comp = pd.DataFrame(rows)
    b = comp.iloc[0]
    comp["cagr_delta_pp"] = (comp.cagr - b.cagr) * 100
    comp["dd_improvement_pp"] = (comp.max_drawdown - b.max_drawdown) * 100
    comp["pf_delta"] = comp.pnl_profit_factor - b.pnl_profit_factor
    comp["win_delta_pp"] = (comp.win_rate - b.win_rate) * 100
    comp.to_csv(RUN_ROOT / "AI_CAUSAL_V2_COMPARISON.csv", index=False)
    (RUN_ROOT / "AI_CAUSAL_V2_COMPARISON.json").write_text(comp.to_json(orient="records", indent=2), encoding="utf-8")

    gate = {}
    for _, r in comp.iloc[1:].iterrows():
        gate[r["variant"]] = {
            "cagr_better": bool(r.cagr > b.cagr),
            "pf_not_worse": bool(r.pnl_profit_factor >= b.pnl_profit_factor),
            "dd_not_worse_over_1pp": bool(r.max_drawdown >= b.max_drawdown - 0.01),
        }
        gate[r["variant"]]["pass"] = all(gate[r["variant"]].values())
    (RUN_ROOT / "AI_V2_SUCCESS_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("=== AI CAUSAL V2 PORTFOLIO ===")
    print(comp.to_string(index=False))
    print("\n=== OOS MODEL DIAGNOSTICS ===")
    print(diag.to_string(index=False))
    print("\n=== V2 SELECTION DIAGNOSTICS ===")
    print(pd.DataFrame(drows).to_string(index=False))
    print("\n=== SUCCESS GATE ===")
    print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
