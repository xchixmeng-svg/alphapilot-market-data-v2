#!/usr/bin/env python3
"""AlphaPilot AI causal reranker v4: bounded adjacent tie-break only.

Preregistered structural change after V3 failure:
- Locked R10 eligibility, execution, exits, sizing, caps and common cash remain immutable.
- Same expanding-year OOS predictions with 60-session purge as V2/V3.
- AI is NOT allowed to delete candidates or globally rerank the list.
- Start from locked R10 order. Partition the ordered candidate list into adjacent
  pairs (ranks 1-2, 3-4, ...). Within each pair only, AI may swap the two names
  if its OOS 20/40/60 mid-horizon score prefers the lower-ranked candidate.
- Therefore any candidate can move by at most one R10 rank. This is a structural
  tie-break/abstention design, not another tuned blend weight.
- Success gate remains preregistered: CAGR > baseline, PF >= baseline, and max DD
  no more than 1 percentage point worse than baseline.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_ai_causal_reranker_v2.py"
spec = importlib.util.spec_from_file_location("ai_v2", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RUN_ROOT = ROOT / "ai_causal_reranker_v4_results"
mod.RUN_ROOT = RUN_ROOT
mod.BASELINE_DIR = RUN_ROOT / "baseline"
mod.VARIANTS = {"AI_ADJ_TIEBREAK": RUN_ROOT / "AI_ADJ_TIEBREAK"}
mod.mod.RUN_ROOT = RUN_ROOT
mod.mod.BASELINE_DIR = mod.BASELINE_DIR
mod.mod.VARIANT_DIR = mod.VARIANTS["AI_ADJ_TIEBREAK"]


def build_runner_v4(locked_source: str, mode: str) -> str:
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
    new = """            combined = [('R7', c, r) for c, r in r7.iterrows()] + [('R05', c, r) for c, r in r05.iterrows()]\n            combined.sort(key=lambda z: -(z[2]['r7_score'] if z[0] == 'R7' else z[2]['r05_score']))\n            if int(di) >= 20220101 and len(combined) > 1:\n                _bounded = []\n                for _k in range(0, len(combined), 2):\n                    _pair = combined[_k:_k+2]\n                    if len(_pair) == 2:\n                        _a0 = AI_MAP.get((int(di), str(_pair[0][1]), str(_pair[0][0])), -1e18)\n                        _a1 = AI_MAP.get((int(di), str(_pair[1][1]), str(_pair[1][0])), -1e18)\n                        if _a1 > _a0:\n                            _pair = [_pair[1], _pair[0]]\n                    _bounded.extend(_pair)\n                combined = _bounded\n            created_mv, reserved_cost, used = 0.0, 0.0, set()\n"""
    if old not in stage3:
        raise RuntimeError("candidate block anchor missing")
    stage3 = stage3.replace(old, new, 1)
    return "# Generated from immutable R10 stage-3; V4 bounded adjacent AI tie-break only.\nimport pandas as pd\nimport numpy as np\nimport json\nfrom pathlib import Path\n\n" + stage3

mod.build_runner = build_runner_v4
mod.main()
