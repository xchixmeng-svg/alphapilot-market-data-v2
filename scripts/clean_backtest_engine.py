"""AlphaPilot clean backtest engine.

Contract-first primitives only. No strategy alpha is embedded here.
Any T-day strategy decision must use information known by T close.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

COMMISSION_RATE = 0.001425
STOCK_TRANSACTION_TAX_RATE = 0.003
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


@dataclass(frozen=True)
class LedgerEntry:
    side: str
    decision_date: str
    execute_date: str
    code: str
    shares: int
    raw_price: float
    modeled_price: float
    raw_trade_value: float
    modeled_trade_value: float
    commission: float
    transaction_tax: float
    cash_delta: float


def _iso_day(value: str) -> date:
    return date.fromisoformat(value)


def validate_t_plus_one(decision_date: str, execute_date: str) -> None:
    """Basic chronology guard. Trading-calendar adjacency is checked by the event loop."""
    if _iso_day(execute_date) <= _iso_day(decision_date):
        raise ValueError("execution must occur after decision date")


def validate_information_cutoff(decision_date: str, latest_input_date: str) -> None:
    """Reject future leakage into a T-close decision."""
    if _iso_day(latest_input_date) > _iso_day(decision_date):
        raise ValueError("future information leakage detected")


def validate_shares(shares: int) -> None:
    if not isinstance(shares, int) or isinstance(shares, bool) or shares <= 0:
        raise ValueError("shares must be a positive integer")
    if shares % ODD_LOT_STEP != 0:
        raise ValueError("shares must use 100-share increments")


def buy_fill_price(order: BuyOrder, next_open: float, next_low: float) -> Optional[float]:
    """Strict precommitted limit fill; no T+1 repricing or chasing."""
    validate_t_plus_one(order.decision_date, order.execute_date)
    validate_shares(order.shares)
    if order.limit_price <= 0 or next_open <= 0 or next_low <= 0:
        raise ValueError("prices must be positive")
    if next_open <= order.limit_price:
        return min(order.limit_price, next_open * (1.0 + BUY_OPEN_SLIPPAGE))
    if next_low <= order.limit_price:
        return order.limit_price
    return None


def sell_fill_price(next_open: float) -> float:
    """Contract Rule 3: conservative next-open sale at open minus 2%."""
    if next_open <= 0:
        raise ValueError("next_open must be positive")
    return next_open * (1.0 - SELL_OPEN_ADVERSE_SLIPPAGE)


def commission(raw_trade_value: float) -> float:
    """No-discount brokerage commission, calculated from raw trade value."""
    if raw_trade_value < 0:
        raise ValueError("trade value cannot be negative")
    return raw_trade_value * COMMISSION_RATE


def stock_transaction_tax(raw_trade_value: float) -> float:
    """Ordinary Taiwan stock transaction tax, kept separate from commission/slippage."""
    if raw_trade_value < 0:
        raise ValueError("trade value cannot be negative")
    return raw_trade_value * STOCK_TRANSACTION_TAX_RATE


def enforce_single_stock_cap(position_market_value: float, nav: float) -> None:
    if nav <= 0:
        raise ValueError("NAV must be positive")
    if position_market_value > nav * MAX_SINGLE_STOCK_NAV + 1e-9:
        raise ValueError("single-stock exposure exceeds 20% NAV")


def make_buy_ledger(order: BuyOrder, raw_reference_price: float, modeled_fill_price: float) -> LedgerEntry:
    validate_shares(order.shares)
    raw_value = raw_reference_price * order.shares
    modeled_value = modeled_fill_price * order.shares
    fee = commission(raw_value)
    return LedgerEntry(
        side="BUY",
        decision_date=order.decision_date,
        execute_date=order.execute_date,
        code=order.code,
        shares=order.shares,
        raw_price=raw_reference_price,
        modeled_price=modeled_fill_price,
        raw_trade_value=raw_value,
        modeled_trade_value=modeled_value,
        commission=fee,
        transaction_tax=0.0,
        cash_delta=-(modeled_value + fee),
    )


def make_sell_ledger(order: SellOrder, raw_open_price: float) -> LedgerEntry:
    validate_t_plus_one(order.decision_date, order.execute_date)
    validate_shares(order.shares)
    modeled = sell_fill_price(raw_open_price)
    raw_value = raw_open_price * order.shares
    modeled_value = modeled * order.shares
    fee = commission(raw_value)
    tax = stock_transaction_tax(raw_value)
    return LedgerEntry(
        side="SELL",
        decision_date=order.decision_date,
        execute_date=order.execute_date,
        code=order.code,
        shares=order.shares,
        raw_price=raw_open_price,
        modeled_price=modeled,
        raw_trade_value=raw_value,
        modeled_trade_value=modeled_value,
        commission=fee,
        transaction_tax=tax,
        cash_delta=modeled_value - fee - tax,
    )


def run_backtest(*args, **kwargs):
    """Reserved for the single causal event loop.

    Performance remains intentionally disabled until the trading-calendar
    adapter, historical 2020-2025 dataset adapter, position lifecycle,
    causal feature interface, and full contract audit are connected.
    """
    raise NotImplementedError(
        "Core accounting is ready; full causal event loop not enabled yet."
    )
