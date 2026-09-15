#!/usr/bin/env python3
"""V14 preregistered decision-level evidence-reuse gate.

Executes the validated V13 engine with one additional causal constraint: within an
OOS test year, the same frozen historical evidence regime may authorize at most
one intervention for a strategy/context route. This directly tests the V14
preregistration diagnosis without changing R10, labels, neighbor thresholds,
execution, sizing, exits, fees, or the portfolio success gate.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root/'scripts'/'backtest_ai_context_neighbor_v13.py').read_text(encoding='utf-8')

# Isolate outputs and labels while preserving the validated V13 implementation.
src = src.replace("ai_context_neighbor_v13_results", "ai_context_policy_v14_results")
src = src.replace("AI_CONTEXT_NEIGHBOR", "AI_CONTEXT_POLICY")
src = src.replace("V13 contextual-neighbor strategy-slot AI tie-break", "V14 decision-level evidence-reuse policy gate")
src = src.replace("AI_V13_", "AI_V14_")
src = src.replace("=== V13 INTERVENTIONS ===", "=== V14 INTERVENTIONS ===")

# Preregistered anti-reuse rule. Prior evidence is frozen to years < test year,
# therefore (year, route, strategy, market_state) identifies an unchanged evidence
# regime. Only its first qualifying decision may consume that evidence.
needle = "scored=[]; audit=[]\n    for yr in [2022,2023,2024,2025]:"
repl = "scored=[]; audit=[]\n    used_evidence=set()\n    for yr in [2022,2023,2024,2025]:"
assert needle in src
src = src.replace(needle, repl, 1)

needle = "nn,mean,hit,med_u,k=neighbor_evidence(sub,row); allow=bool((row.ret_margin>0) and (row.fail_margin>0) and (mean>0) and (hit>=0.60))"
repl = "nn,mean,hit,med_u,k=neighbor_evidence(sub,row); evidence_key=(yr,route,str(row.strategy),str(row.market_state) if route=='STRATEGY_STATE' else '*'); raw_allow=bool((row.ret_margin>0) and (row.fail_margin>0) and (mean>0) and (hit>=0.60)); allow=bool(raw_allow and evidence_key not in used_evidence); used_evidence.add(evidence_key) if allow else None"
assert needle in src
src = src.replace(needle, repl, 1)

# Keep CI success distinct from research success; existing CAGR/PF/DD gate remains.
code = compile(src, str(root/'scripts'/'backtest_ai_context_neighbor_v13.py'), 'exec')
exec(code, {'__name__':'__main__','__file__':str(root/'scripts'/'backtest_ai_context_neighbor_v13.py')})
