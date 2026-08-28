"""Executable contract checks for the clean AlphaPilot engine."""
from clean_backtest_engine import (
    BuyOrder,
    SellOrder,
    COMMISSION_RATE,
    STOCK_TRANSACTION_TAX_RATE,
    MAX_SINGLE_STOCK_NAV,
    buy_fill_price,
    sell_fill_price,
    validate_information_cutoff,
    validate_shares,
    enforce_single_stock_cap,
    make_buy_ledger,
    make_sell_ledger,
)


def must_fail(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError(f"expected failure: {fn.__name__}{args!r}")


def main() -> None:
    assert COMMISSION_RATE == 0.001425
    assert STOCK_TRANSACTION_TAX_RATE == 0.003
    assert MAX_SINGLE_STOCK_NAV == 0.20

    validate_shares(100)
    validate_shares(500)
    validate_shares(1000)
    must_fail(validate_shares, 437)
    must_fail(validate_shares, 0)
    must_fail(validate_shares, 100.0)

    # T close precommits next-day BUY.
    o = BuyOrder("2026-08-26", "2026-08-27", "2330", 500, 100.0)
    assert abs(buy_fill_price(o, 98.0, 97.0) - 98.49) < 1e-9
    assert buy_fill_price(o, 101.0, 99.5) == 100.0
    assert buy_fill_price(o, 101.0, 100.5) is None

    # No future leakage.
    validate_information_cutoff("2026-08-26", "2026-08-26")
    must_fail(validate_information_cutoff, "2026-08-26", "2026-08-27")

    # Single-stock hard cap.
    enforce_single_stock_cap(200_000, 1_000_000)
    must_fail(enforce_single_stock_cap, 200_001, 1_000_000)

    # SELL at next open with -2% adverse modeled price.
    s = SellOrder("2026-08-26", "2026-08-27", "2330", 500, True)
    assert sell_fill_price(100.0) == 98.0
    sl = make_sell_ledger(s, 100.0)
    assert sl.raw_trade_value == 50_000.0
    assert sl.modeled_trade_value == 49_000.0
    assert abs(sl.commission - 71.25) < 1e-9
    assert abs(sl.transaction_tax - 150.0) < 1e-9
    assert abs(sl.cash_delta - 48_778.75) < 1e-9

    bl = make_buy_ledger(o, 98.0, 98.49)
    assert abs(bl.commission - 69.825) < 1e-9
    assert bl.transaction_tax == 0.0

    print("CLEAN CONTRACT AUDIT: PASS")
    print("checks: strict_t1, no_future, precommit_fill, sell_minus_2pct, 100_share_step, 20pct_cap, fees, stock_tax, ledger")


if __name__ == "__main__":
    main()
