#!/usr/bin/env python3
"""AlphaPilot AI causal reranker v3: conservative 25% AI / 75% locked-R10 rank blend.

Purpose: V2's 50/50 blend improved CAGR and preserved max DD but reduced profit factor.
V3 changes only the rank-mixing strength, preserving the same expanding-year OOS
predictions, 60-session purge, T+1 execution, fees/tax/slippage, exits, caps and
common cash. No AI gating/deletion is allowed.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v2.py"
spec = importlib.util.spec_from_file_location("ai_v2", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RUN_ROOT = ROOT / "ai_causal_reranker_v3_results"
mod.RUN_ROOT = RUN_ROOT
mod.BASELINE_DIR = RUN_ROOT / "baseline"
mod.VARIANTS = {"AI_MID_BLEND25": RUN_ROOT / "AI_MID_BLEND25"}
mod.mod.RUN_ROOT = RUN_ROOT
mod.mod.BASELINE_DIR = mod.BASELINE_DIR
mod.mod.VARIANT_DIR = mod.VARIANTS["AI_MID_BLEND25"]


def build_runner_v3(locked_source: str, mode: str) -> str:
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
    new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            if int(di) >= 20220101 and len(combined) > 1:\n                _orig = sorted(range(len(combined)), key=lambda j: -(combined[j][2]['r7_score'] if combined[j][0] == 'R7' else combined[j][2]['r05_score']))\n                _air = sorted(range(len(combined)), key=lambda j: -AI_MAP.get((int(di), str(combined[j][1]), str(combined[j][0])), -1e18))\n                _orank = {j:i for i,j in enumerate(_orig)}\n                _arank = {j:i for i,j in enumerate(_air)}\n                combined = [combined[j] for j in sorted(range(len(combined)), key=lambda j: (3*_orank[j] + _arank[j], _orank[j]))]\n            else:\n                combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated from immutable R10 stage-3; V3 changes ranking only.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3

mod.build_runner = build_runner_v3
mod.main()
