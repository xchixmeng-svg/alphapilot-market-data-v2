"""AlphaPilot clean backtest engine.

Contract-first event loop. No strategy alpha is embedded here.
All strategy decisions must be produced from information known by T close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

COMMISSION_RATE = 0.001425
MAX_SINGLE_STOCK_NAV = 0.20
BUY_OPEN_SLIPPAGE = 0.005
SELL_OPEN_ADVERSE_SLIPPAGE = 0.02
BOARD_LOT = 1000
ODD_LOT_STEP = 100


@dataclass(frozen=True)
class BuyOrder:
    decision_date: str
    execute_date: str
    code: str
    shares: int
    limit_price: float


@dataclass(frozen=True)
class SellOrder:
    decision_date: str
    execute_date: str
    code: str
    shares: int
    full_exit: bool


def validate_shares(shares: int) -> None:
    if not isinstance(shares, int) or shares <= 0:
        raise ValueError("shares must be a positive integer")
    if shares % ODD_LOT_STEP != 0:
        raise ValueError("shares must use 100-share increments")


def buy_fill_price(order: BuyOrder, next_open: float, next_low: float) -> Optional[float]:
    """Strict T+1 precommitted limit fill. No chasing."""
    validate_shares(order.shares)
    if next_open <= order.limit_price:
        return min(order.limit_price, next_open * (1.0 + BUY_OPEN_SLIPPAGE))
    if next_low <= order.limit_price:
        return order.limit_price
    return None


def sell_fill_price(next_open: float) -> float:
    """Conservative T+1 open sale model."""
    if next_open <= 0:
        raise ValueError("next_open must be positive")
    return next_open * (1.0 - SELL_OPEN_ADVERSE_SLIPPAGE)


def commission(raw_trade_value: float) -> float:
    """No-discount commission. Tax is kept separate by instrument."""
    if raw_trade_value < 0:
        raise ValueError("trade value cannot be negative")
    return raw_trade_value * COMMISSION_RATE


def enforce_single_stock_cap(position_market_value: float, nav: float) -> None:
    if nav <= 0:
        raise ValueError("NAV must be positive")
    if position_market_value > nav * MAX_SINGLE_STOCK_NAV + 1e-9:
        raise ValueError("single-stock exposure exceeds 20% NAV")


def run_backtest(*args, **kwargs):
    """Future single event-loop entry point.

    Intentionally refuses to produce performance until the data adapter,
    tax model, causal feature builder, strategy interface, ledger, and
    contract audits are implemented and validated.
    """
    raise NotImplementedError(
        "Clean engine skeleton only: no performance result is valid yet."
    )
