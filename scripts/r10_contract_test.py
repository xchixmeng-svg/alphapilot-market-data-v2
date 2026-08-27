#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10 locked-rule contract test.

Purpose: verify the complete locked R10 rule contract in seconds, without
network access, historical orders, historical trades, or performance targets.
Historical End NAV/CAGR/DD/trade counts are outputs and are never contract
answers.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import backtest_r10_stress as bt
import build_r10_scan as core
import r10_true_validation as true_engine

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "contract_results"
OUT.mkdir(parents=True, exist_ok=True)

checks: list[dict] = []

def ok(rule: str, condition: bool, detail: str = "") -> None:
    checks.append({"rule": rule, "pass": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{rule} FAIL: {detail}")

# ---------------------------------------------------------------------------
# A. Causality/execution contract and portfolio constants
# ---------------------------------------------------------------------------
ok("P01_INITIAL_NAV", bt.INITIAL_CAPITAL == 1_300_000.0)
ok("P02_R7_BASE", bt.R7_BASE == 0.22)
ok("P03_R05_BASE", bt.R05_BASE == 0.20)
ok("P04_MAX_STOCKS", bt.MAX_POSITIONS == 5)
ok("P05_SINGLE_CAP", bt.MAX_SINGLE == 0.25)
ok("P06_TOTAL_EXPOSURE", bt.MAX_TOTAL == 0.95)
ok("P07_R05_SLOTS", bt.R05_MAX_SLOTS == 3)
ok("P08_ADV_CAP", bt.ADV_CAP == 0.02)
ok("P09_FORCE_DD", bt.FORCE_DD == -0.14)
ok("P10_FORCE_TARGET", bt.FORCE_TARGET_EXPOSURE == 0.50)
ok("P11_NO_BUY_DAYS", bt.FORCE_NO_BUY_DAYS == 10)
ok("P12_COOLDOWN_DAYS", bt.FORCE_COOLDOWN_DAYS == 15)
ok("E01_BUY_FEE", bt.BUY_FEE == 0.000855)
ok("E02_SELL_FEE", bt.SELL_FEE == 0.000855)
ok("E03_SELL_TAX", bt.SELL_TAX == 0.003)
ok("E04_SELL_SLIPPAGE", bt.SELL_SLIPPAGE == 0.005)

# Performance numbers are explicitly NOT contract assertions.
contract_source = Path(__file__).read_text(encoding="utf-8")
ok("B01_NO_END_NAV_ANSWER_GATE", 'B02_CAUSAL_NAV' not in contract_source)
ok("B02_TRUE_ENGINE_NO_FIXTURE_INPUT", 'tests/fixtures' not in (ROOT / "scripts" / "r10_true_validation.py").read_text(encoding="utf-8"))
ok("B03_TRUE_ENGINE_NO_BENCHMARK_INPUT", 'CAUSAL_BENCHMARK' not in (ROOT / "scripts" / "r10_true_validation.py").read_text(encoding="utf-8"))
ok("B04_TRUE_ENGINE_NO_LEGACY_INPUT", 'LEGACY_CONTAMINATED_BENCHMARK' not in (ROOT / "scripts" / "r10_true_validation.py").read_text(encoding="utf-8"))

# DD throttle boundary behavior.
ok("DD01_GT_6", bt.dd_multiplier(-0.059999) == 1.00)
ok("DD02_AT_6", bt.dd_multiplier(-0.06) == 0.85)
ok("DD03_AT_9", bt.dd_multiplier(-0.09) == 0.45)
ok("DD04_AT_14", bt.dd_multiplier(-0.14) == 0.45)
ok("DD05_AT_15", bt.dd_multiplier(-0.15) == 0.40)

# ---------------------------------------------------------------------------
# B. T -> T+1 execution mechanics (generic synthetic values only)
# ---------------------------------------------------------------------------
ok("E05_BUY_OPEN_ADVERSE", bt.buy_fill(98.0, 97.0, 99.0) == 98.5, f"got={bt.buy_fill(98.0,97.0,99.0)}")
ok("E05A_BUY_TICK_LT50", bt.buy_fill(17.8, 17.35, 18.05) == 17.9)
ok("E05B_BUY_TICK_LT50_B", bt.buy_fill(20.5, 19.9, 20.7) == 20.6)
ok("E05C_BUY_TICK_50_100", bt.buy_fill(53.0, 52.1, 54.1) == 53.3)
ok("E05D_BUY_TICK_LT50_C", bt.buy_fill(22.65, 22.2, 22.9) == 22.8)
ok("E06_BUY_LOW_TOUCH", bt.buy_fill(101.0, 98.5, 99.0) == 99.0)
ok("E07_BUY_NO_TOUCH", bt.buy_fill(101.0, 99.1, 99.0) is None)
expected_sell = float(core.floor_tick(100.0 * 0.995))
ok("E08_SELL_T1_ADVERSE", bt.legal_sell_price(100.0) == expected_sell, f"got={bt.legal_sell_price(100.0)}")
ok("E09_R7_LIMIT", core.floor_tick(100.0 * 0.98) == 98.0)
ok("E10_R05_LIMIT", core.floor_tick(40.0 * 0.995) == 39.8)

# Active quantity semantics are tested against the clean causal engine, not a
# historical order row or old FAST implementation.
shares, mode = true_engine.size_shares_true(260_000, 100.0, 100_000)
ok("E11_BOARD_LOT_INTEGER", shares % 1000 == 0 and mode == "BOARD_LOT", f"shares={shares},mode={mode}")
ok("E12_ADV_2PCT", shares <= 2_000, f"shares={shares}")
shares2, mode2 = true_engine.size_shares_true(80_000, 200.0, 100_000)
ok("E13_ODDLOT_ONLY_WHEN_LOT_EXCEEDS_TARGET", mode2 == "HIGH_PRICE_ODDLOT" and isinstance(shares2, int), f"shares={shares2},mode={mode2}")
shares3, mode3 = true_engine.size_shares_true(260_000, 27.0, 10_000_000)
ok("E14_BOARD_LOT_FLOOR_NOT_ROUND_UP", shares3 == 9_000 and mode3 == "BOARD_LOT", f"shares={shares3},mode={mode3}")
ok("E15_BOARD_LOT_NEVER_EXCEEDS_TARGET", shares3 * 27.0 <= 260_000 + 1e-9, f"shares={shares3}")
shares4, mode4 = true_engine.size_shares_true(80_000, 500.0, 10_000_000)
ok("E16_HIGH_PRICE_ONLY_ODDLOT", 0 < shares4 < 1000 and mode4 == "HIGH_PRICE_ODDLOT", f"shares={shares4},mode={mode4}")

# ---------------------------------------------------------------------------
# C. R0.5 executable exit state machine
# ---------------------------------------------------------------------------
def pos(mode="NORMAL", entry=100.0, peak=100.0, hold=0):
    return bt.Position("R05", "1234", "TEST", 1000, 20260101, entry, entry, entry*1000, peak, mode, hold)

def row(adj, amount_ratio=1.0):
    return SimpleNamespace(aclose=float(adj), amount_ratio=float(amount_ratio))

p = pos(); ok("R05_EXIT_HARD_10", bt.r05_exit_reason(p, row(90.0)) == "HARD")
p = pos(); reason = bt.r05_exit_reason(p, row(140.0, 2.0)); ok("R05_RUNNER_PROMOTION", p.mode == "RUNNER" and reason is None)
p = pos(mode="RUNNER", peak=150.0); ok("R05_RUNNER_TRAIL_14", bt.r05_exit_reason(p, row(129.0)) == "RUNNER_TRAIL")
p = pos(mode="MEGA", peak=200.0); ok("R05_MEGA_TRAIL_16", bt.r05_exit_reason(p, row(168.0)) == "MEGA_TRAIL")
p = pos(mode="TARGET", peak=200.0); ok("R05_TARGET_TRAIL_20", bt.r05_exit_reason(p, row(160.0)) == "TARGET_TRAIL")
p = pos(mode="TARGET", peak=300.0); ok("R05_TARGET_200", bt.r05_exit_reason(p, row(300.0)) == "TARGET_200")
p = pos(peak=180.0); ok("R05_BASE_TRAIL_12_AFTER_50", bt.r05_exit_reason(p, row(158.0)) == "BASE_TRAIL")
p = pos(hold=59); ok("R05_NORMAL_TIME_60", bt.r05_exit_reason(p, row(105.0)) == "TIME")
p = pos(mode="RUNNER", hold=119, peak=150); ok("R05_RUNNER_TIME_120", bt.r05_exit_reason(p, row(145.0)) == "RUNNER_TIME")

# ---------------------------------------------------------------------------
# D. Exact scanner/portfolio source contract
# ---------------------------------------------------------------------------
scanner = (ROOT / "scripts" / "build_r10_scan.py").read_text(encoding="utf-8")
engine = (ROOT / "scripts" / "backtest_r10_stress.py").read_text(encoding="utf-8")
true_source = (ROOT / "scripts" / "r10_true_validation.py").read_text(encoding="utf-8")

def contains(rule: str, text: str, *snips: str) -> None:
    ok(rule, all(s in text for s in snips), "missing locked source clause")

contains("R7_REGIME_BEAR", scanner, 'mr20<=-.08', 'm<ma120 and mr60<0 and breadth<.40', '"Bear",0.,0')
contains("R7_REGIME_REPAIR", scanner, 'm<ma120*1.02', 'breadth>.42', 'breadth>bmean', '"Repair",.60,2')
contains("R7_REGIME_STRONG", scanner, 'm>ma60 and m>ma120', 'breadth>=.60', 'adv>=.52', '"Strong Bull",1.,4')
contains("R7_REGIME_NORMAL", scanner, 'm>ma120 and mr60>0 and breadth>=.45', '"Normal Bull",.80,3')
contains("R7_REGIME_WEAK", scanner, 'm>ma120*.98 and breadth>=.38', '"Weak",.20,2')
contains("R7_SCORE", scanner, '.26*x.p10+.22*x.p20+.10*x.p60+.14*x.pf+.12*x.pa+.08*x.pc+.08*x.pn')
contains("R7_FILTERS", scanner, '(x.amt20>=30_000_000)', '(x.aclose>x.ma120)', '(x.nearhigh>=.78)')
contains("R7_REBALANCE_15", scanner, 'n>=15')
contains("R7_HARD_STOP_12", engine, 'float(r.aclose) <= p.entry_adj * 0.88', 'reason = "HARD"')
contains("R7_REGIME_EXIT", engine, 'reason = "REB_REGIME0"', 'reason = "REB_RANK"', '"EXPO"')

contains("R05_RISK_ON", scanner, 'e.close>e.m60 and e.r20x>0 and e.r60x>0')
contains("R05_SCORE", scanner, '.5251*x.pclv10+.2465*x.pamt+.0683*x.pclv5+.0628*x.pf3-.0778*x.pf10+.0195*x.pt5-.2*x.pgap')
contains("R05_FILTER_PRICE", scanner, 'x.close.between(10,40)')
contains("R05_FILTER_AMT", scanner, 'x.amt20>=50_000_000')
contains("R05_FILTER_RATIO", scanner, 'x.amount_ratio>=1')
contains("R05_FILTER_R20", scanner, 'x.r20.between(0,.20)')
contains("R05_FILTER_MA20GAP", scanner, 'x.ma20gap<=.18')
contains("R05_FILTER_PRIOR60", scanner, 'x.prior60_position>=-.15')
contains("R05_BREAKOUT_10", scanner, 'x.aclose>x.prior_high10')
contains("R05_INST_ROLLS", scanner, 'Foreign3D', 'Foreign10D', 'Trust5D')

contains("PORT_FORCE_DD_EXEC", true_source, 'dd <= bt.FORCE_DD', 'force_target = nav * bt.FORCE_TARGET_EXPOSURE', '"FORCE_DD"')
contains("PORT_NO_BUY_EXEC", true_source, 'no_buy_until = max(no_buy_until, i + bt.FORCE_NO_BUY_DAYS)')
contains("PORT_COOLDOWN_EXEC", true_source, 'force_cooldown_until = i + bt.FORCE_COOLDOWN_DAYS')
contains("PORT_MAX5_EXEC", true_source, 'len(codes_after) >= bt.MAX_POSITIONS')
contains("PORT_SINGLE25_EXEC", true_source, 'nav * bt.MAX_SINGLE - current_code')
contains("PORT_TOTAL95_EXEC", true_source, 'nav * bt.MAX_TOTAL - base_exposure - reserved_exposure')
contains("PORT_T1_SELLS_FIRST", true_source, 'pending_sells.pop(di', 'pending_buys.pop(di')
ok("PORT_T1_SELLS_PRECEDE_BUYS", true_source.index('pending_sells.pop(di') < true_source.index('pending_buys.pop(di'))
contains("PORT_ACTUAL_CASH_CHECK_AT_EXECUTION", true_source, 'if cost > cash + 1e-6:', '"CASH_SHORT_AT_EXECUTION"')
contains("CAUSAL_T1_PENDING", engine, 'T+1 buys: fixed T order, no chasing.', 'T close fixed buy orders')

ok("NO_PROFILE_OVERRIDE", bt.ADV_CAP == 0.02 and bt.FORCE_DD == -0.14 and bt.dd_multiplier(-0.06) == 0.85)

summary = {
    "status": "PASS",
    "tests": len(checks),
    "passed": sum(1 for x in checks if x["pass"]),
    "failed": sum(1 for x in checks if not x["pass"]),
    "network_used": False,
    "historical_data_used": False,
    "historical_performance_target_used": False,
    "contract": "docs/R10_LOCKED_RULES_SOURCE_OF_TRUTH.md",
    "checks": checks,
}
(OUT / "r10_contract_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("status","tests","passed","failed")}, ensure_ascii=False))
