#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10 locked-rule validation wrapper.

This wrapper exists because the prior LOCKED_BASELINE helper disabled three
controls that docs/R10_STRESS_BACKTEST.md explicitly defines as part of the
locked R10 portfolio layer.  This file restores those documented rules before
running the optimized causal five-year engine.

It does NOT change R7 or R0.5 selection formulas and does NOT use future data.
"""
from __future__ import annotations

import json
from pathlib import Path

import r10_fast_validation as fast

bt = fast.bt
full = fast.full

# Restore the documented LOCKED R10 portfolio controls.
bt.ADV_CAP = full._BASE_ADV_CAP                 # 2% of T-known 20D ADV
bt.FORCE_DD = full._BASE_FORCE_DD               # -14% portfolio DD trigger
bt.dd_multiplier = full._BASE_DD_MULTIPLIER     # -6/-9/-15 entry-size throttle
bt.VERSION = "AlphaPilot-R10-FastValidation-v2-DOCUMENTED-LOCKED-RULES"


def assert_locked_rules() -> None:
    # Portfolio layer
    assert abs(bt.INITIAL_CAPITAL - 1_300_000.0) < 1e-9
    assert abs(bt.R7_BASE - 0.22) < 1e-12
    assert abs(bt.R05_BASE - 0.20) < 1e-12
    assert bt.MAX_POSITIONS == 5
    assert abs(bt.MAX_SINGLE - 0.25) < 1e-12
    assert abs(bt.MAX_TOTAL - 0.95) < 1e-12
    assert abs(bt.ADV_CAP - 0.02) < 1e-12
    assert abs(bt.FORCE_DD - (-0.14)) < 1e-12
    assert abs(bt.FORCE_TARGET_EXPOSURE - 0.50) < 1e-12
    assert bt.FORCE_NO_BUY_DAYS == 10
    assert bt.FORCE_COOLDOWN_DAYS == 15

    # DD throttle
    assert abs(bt.dd_multiplier(-0.059) - 1.00) < 1e-12
    assert abs(bt.dd_multiplier(-0.060) - 0.85) < 1e-12
    assert abs(bt.dd_multiplier(-0.090) - 0.45) < 1e-12
    assert abs(bt.dd_multiplier(-0.150) - 0.40) < 1e-12

    # Execution / costs
    assert abs(bt.BUY_FEE - 0.000855) < 1e-12
    assert abs(bt.SELL_FEE - 0.000855) < 1e-12
    assert abs(bt.SELL_TAX - 0.003) < 1e-12
    assert abs(bt.SELL_SLIPPAGE - 0.005) < 1e-12

    # Current strategy-specific hard stops in the locked implementation:
    # R7: adjusted close <= entry adjusted price * 0.88 -> HARD, decision T, sell T+1.
    # R0.5: adjusted close <= entry adjusted price * 0.90 -> HARD, decision T, sell T+1.
    # These are asserted by source audit in docs/R10_LOCKED_RULES_SOURCE_OF_TRUTH.md;
    # do not silently alter them to improve backtest results.


def write_rule_audit(result: dict) -> None:
    out = bt.OUT_ROOT / "latest" / "validation2021_2025"
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "rule_profile": bt.VERSION,
        "source_of_truth": "docs/R10_LOCKED_RULES_SOURCE_OF_TRUTH.md",
        "r7_r05_selection_changed": False,
        "future_data_used": False,
        "portfolio": {
            "initial_capital": bt.INITIAL_CAPITAL,
            "r7_new_signal_nav_pct": bt.R7_BASE,
            "r05_new_signal_nav_pct": bt.R05_BASE,
            "max_positions": bt.MAX_POSITIONS,
            "max_single_nav_pct": bt.MAX_SINGLE,
            "max_total_exposure": bt.MAX_TOTAL,
            "adv_cap": bt.ADV_CAP,
            "force_dd": bt.FORCE_DD,
            "force_target_exposure": bt.FORCE_TARGET_EXPOSURE,
            "force_no_buy_days": bt.FORCE_NO_BUY_DAYS,
            "force_cooldown_days": bt.FORCE_COOLDOWN_DAYS,
        },
        "execution": {
            "T_close_decision_T1_execution": True,
            "r7_buy_limit_multiplier": 0.98,
            "r05_buy_limit_multiplier": 0.995,
            "sell_slippage": bt.SELL_SLIPPAGE,
            "buy_fee": bt.BUY_FEE,
            "sell_fee": bt.SELL_FEE,
            "sell_tax": bt.SELL_TAX,
        },
        "hard_stop_implementation": {
            "R7": "T adjusted close <= entry adjusted price * 0.88; sell T+1",
            "R05": "T adjusted close <= entry adjusted price * 0.90; sell T+1",
        },
        "result_snapshot": {
            "end_nav": result.get("end_nav"),
            "cagr": result.get("cagr"),
            "max_dd": result.get("max_dd"),
            "completed_trades": result.get("completed_trades"),
        },
    }
    (out / "locked_rule_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    assert_locked_rules()
    bt.log("[RULES] documented locked R10 controls asserted: PASS")
    result = fast.simulate_fast()
    write_rule_audit(result)
