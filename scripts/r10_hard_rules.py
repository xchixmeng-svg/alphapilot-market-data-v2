#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical hard rules for AlphaPilot R10 research/backtests.

The optimizer may choose alpha logic and market-regime allocation, but it may
never change execution reality or overweight portfolio limits. A run that
violates this contract is invalid regardless of CAGR.
"""
from __future__ import annotations

import math
from typing import Any

INITIAL_CAPITAL = 1_300_000.0

# ---------- execution clock ----------
DECISION_CLOCK = "T_CLOSE"
BUY_EXECUTION = "T+1_PRECOMMITTED_LIMIT"
SELL_EXECUTION = "T+1_OPEN"
SELL_ADVERSE_SLIPPAGE = 0.005
MIN_HOLD_TRADING_DAYS = 3

# ---------- transaction costs; RAW executable prices only ----------
BUY_FEE_RATE = 0.000855
SELL_FEE_RATE = 0.000855
SELL_TAX_RATE = 0.003

# ---------- portfolio / capital ----------
COMMON_CASH_POOL = True
NO_LEVERAGE = True
NO_SHORTING = True
MAX_DISTINCT_STOCKS = 5
MAX_SINGLE_STOCK_NAV = 0.25
MAX_TOTAL_STOCK_NAV = 0.95
R7_BASE_TARGET_NAV = 0.22
R05_BASE_TARGET_NAV = 0.20
R05_MAX_SLOTS = 3
MIN_CASH_RESERVE_NAV = 1.0 - MAX_TOTAL_STOCK_NAV
ADV_CAP = 0.02
BOARD_LOT = 1000

# DD controls from the locked R10 portfolio layer.
DD_LEVEL1 = -0.06
DD_LEVEL2 = -0.09
DD_LEVEL3 = -0.15
DD_MULT1 = 0.85
DD_MULT2 = 0.45
DD_MULT3 = 0.40
FORCE_DD = -0.14
FORCE_TARGET_EXPOSURE = 0.50
FORCE_NO_BUY_DAYS = 10
FORCE_COOLDOWN_DAYS = 15


def dd_multiplier(dd: float) -> float:
    if dd <= DD_LEVEL3:
        return DD_MULT3
    if dd <= DD_LEVEL2:
        return DD_MULT2
    if dd <= DD_LEVEL1:
        return DD_MULT1
    return 1.0


def harden_portfolio_params(p: dict[str, Any]) -> dict[str, Any]:
    """Lock overweight/execution-risk knobs; preserve legal strategy allocation choices."""
    q = dict(p)
    q["max_positions"] = MAX_DISTINCT_STOCKS
    q["max_single"] = MAX_SINGLE_STOCK_NAV
    q["max_total"] = MAX_TOTAL_STOCK_NAV
    q["r7_base"] = R7_BASE_TARGET_NAV
    q["r05_base"] = R05_BASE_TARGET_NAV
    # Strategy activation is optimizer-owned, but never above the locked cap.
    q["r05_max_slots"] = max(0, min(R05_MAX_SLOTS, int(q.get("r05_max_slots", R05_MAX_SLOTS))))
    for key in ["strong_slots", "normal_slots", "repair_slots", "weak_slots"]:
        if key in q:
            q[key] = max(0, min(MAX_DISTINCT_STOCKS, int(q[key])))
    for key in ["strong_exposure", "normal_exposure", "repair_exposure", "weak_exposure"]:
        if key in q:
            q[key] = max(0.0, min(1.0, float(q[key])))
    q["dd_level1"] = DD_LEVEL1
    q["dd_level2"] = DD_LEVEL2
    q["dd_level3"] = DD_LEVEL3
    q["dd_mult1"] = DD_MULT1
    q["dd_mult2"] = DD_MULT2
    q["dd_mult3"] = DD_MULT3
    q["force_dd"] = FORCE_DD
    q["force_target_exposure"] = FORCE_TARGET_EXPOSURE
    q["force_no_buy"] = FORCE_NO_BUY_DAYS
    q["force_cooldown"] = FORCE_COOLDOWN_DAYS
    return q


def raw_buy_cost(raw_price: float, shares: int) -> dict[str, float]:
    gross = float(raw_price) * int(shares)
    fee = gross * BUY_FEE_RATE
    return {"gross": gross, "fee": fee, "total": gross + fee}


def raw_sell_proceeds(raw_price: float, shares: int) -> dict[str, float]:
    gross = float(raw_price) * int(shares)
    fee = gross * SELL_FEE_RATE
    tax = gross * SELL_TAX_RATE
    return {"gross": gross, "fee": fee, "tax": tax, "net": gross - fee - tax}


def full_trade_costs(buy_raw_price: float, sell_raw_price: float, shares: int) -> dict[str, float]:
    b = raw_buy_cost(buy_raw_price, shares)
    s = raw_sell_proceeds(sell_raw_price, shares)
    pnl = s["net"] - b["total"]
    ret = pnl / b["total"] if b["total"] else float("nan")
    return {
        "buy_gross": b["gross"], "buy_fee": b["fee"], "buy_total": b["total"],
        "sell_gross": s["gross"], "sell_fee": s["fee"], "sell_tax": s["tax"],
        "sell_net": s["net"], "pnl": pnl, "return": ret,
    }


def size_shares(target_cash: float, raw_limit: float, avg_vol20: float, base_target_cash: float) -> tuple[int, str]:
    """Board-lot first; odd lot only when one full lot exceeds intended base target."""
    if target_cash <= 0 or raw_limit <= 0 or not math.isfinite(float(avg_vol20)):
        return 0, "NONE"
    per_share_cost = float(raw_limit) * (1.0 + BUY_FEE_RATE)
    one_lot_cost = per_share_cost * BOARD_LOT
    liq_shares = max(0, int(math.floor(float(avg_vol20) * ADV_CAP + 1e-12)))
    oddlot_allowed = one_lot_cost > float(base_target_cash) + 1e-9

    if oddlot_allowed:
        shares = int(math.floor(float(target_cash) / per_share_cost + 1e-12))
        return max(0, min(shares, liq_shares)), "HIGH_PRICE_ODDLOT"

    lots_by_cash = int(math.floor(float(target_cash) / one_lot_cost + 1e-12))
    lots_by_liq = liq_shares // BOARD_LOT
    lots = max(0, min(lots_by_cash, lots_by_liq))
    if lots <= 0:
        return 0, "SKIP_RESIDUAL_NO_ODDLOT"
    return lots * BOARD_LOT, "BOARD_LOT"


def can_emit_sell(hold_days: int) -> bool:
    return int(hold_days) >= MIN_HOLD_TRADING_DAYS


def contract_dict() -> dict[str, Any]:
    return {
        "version": "R10-HARD-CONTRACT-v2",
        "initial_capital_twd": INITIAL_CAPITAL,
        "signal_timeframe": "DAILY_K_OHLCV_AT_T_CLOSE",
        "decision_clock": DECISION_CLOCK,
        "buy_execution": BUY_EXECUTION,
        "sell_execution": SELL_EXECUTION,
        "sell_adverse_slippage": SELL_ADVERSE_SLIPPAGE,
        "minimum_hold_trading_days": MIN_HOLD_TRADING_DAYS,
        "cost_basis": "RAW_EXECUTABLE_PRICE",
        "buy_fee_rate": BUY_FEE_RATE, "sell_fee_rate": SELL_FEE_RATE, "sell_tax_rate": SELL_TAX_RATE,
        "common_cash_pool": COMMON_CASH_POOL, "no_leverage": NO_LEVERAGE, "no_shorting": NO_SHORTING,
        "max_distinct_stocks": MAX_DISTINCT_STOCKS,
        "max_single_stock_nav": MAX_SINGLE_STOCK_NAV,
        "max_total_stock_nav": MAX_TOTAL_STOCK_NAV,
        "r7_base_target_nav": R7_BASE_TARGET_NAV,
        "r05_base_target_nav": R05_BASE_TARGET_NAV,
        "r05_max_slots": R05_MAX_SLOTS,
        "min_cash_reserve_nav": MIN_CASH_RESERVE_NAV,
        "adv_cap": ADV_CAP, "board_lot": BOARD_LOT,
        "optimizer_may_choose": ["alpha family", "signal/filter parameters", "candidate ranking", "T+1 legal limit multiplier", "market-regime exposure within hard caps", "market-regime slots within hard caps", "strategy activation", "exit thresholds subject to min hold"],
        "optimizer_may_not_choose": ["execution clock", "minimum hold", "fees/tax", "slippage semantics", "max 5 stocks", "25% single-stock cap", "95% total cap", "R7 22% base", "R05 20% base", "DD throttle/force-DD", "integer/board-lot rules", "cash precommit semantics"],
        "odd_lot_rule": "only when one board lot exceeds intended base target; residual capacity cannot create tiny odd lots",
        "pending_sell_slot_rule": "pending T+1 sells occupy slots and exposure at T close",
        "cash_precommit_rule": "T-day buys reserve known T cash cumulatively; planned T+1 sell proceeds cannot finance new T orders",
        "same_stock_rule": "one distinct stock exposure across all sleeves; combined <=25% NAV",
        "invalid_if_any_audit_fails": True,
    }


def validate_hard_params(p: dict[str, Any]) -> None:
    exact = {
        "max_positions": MAX_DISTINCT_STOCKS, "max_single": MAX_SINGLE_STOCK_NAV,
        "max_total": MAX_TOTAL_STOCK_NAV, "r7_base": R7_BASE_TARGET_NAV,
        "r05_base": R05_BASE_TARGET_NAV, "dd_level1": DD_LEVEL1, "dd_level2": DD_LEVEL2,
        "dd_level3": DD_LEVEL3, "dd_mult1": DD_MULT1, "dd_mult2": DD_MULT2,
        "dd_mult3": DD_MULT3, "force_dd": FORCE_DD,
        "force_target_exposure": FORCE_TARGET_EXPOSURE, "force_no_buy": FORCE_NO_BUY_DAYS,
        "force_cooldown": FORCE_COOLDOWN_DAYS,
    }
    for key, expected in exact.items():
        actual = p.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12:
                raise AssertionError(f"hard rule drift: {key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            raise AssertionError(f"hard rule drift: {key}={actual!r}, expected {expected!r}")
    if not (0 <= int(p.get("r05_max_slots", 0)) <= R05_MAX_SLOTS):
        raise AssertionError("r05_max_slots exceeds locked maximum")
    for key in ["strong_slots", "normal_slots", "repair_slots", "weak_slots"]:
        if key in p and not (0 <= int(p[key]) <= MAX_DISTINCT_STOCKS):
            raise AssertionError(f"{key} exceeds locked stock-count cap")
    for key in ["strong_exposure", "normal_exposure", "repair_exposure", "weak_exposure"]:
        if key in p and not (0.0 <= float(p[key]) <= 1.0):
            raise AssertionError(f"{key} outside legal regime-allocation range")
