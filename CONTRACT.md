# AlphaPilot Clean Contract v0.1

This contract defines the immutable execution, accounting, risk, data, and anti-lookahead rules for the rebuilt AlphaPilot engine. Strategy research may optimize only inside these boundaries.

## Rule 1 — Strict T+1

- Formal decisions are made only after the full T-day daily bar is complete.
- All BUY and SELL execution begins no earlier than T+1.
- A T-day decision may use only information genuinely known by T close.
- T+1 or later information must never influence a T-day decision.

## Rule 2 — BUY must be precommitted

At T close the engine must precommit the T+1 stock, integer share quantity, and limit price.

Fill model:

- If T+1 open <= limit price, the order may fill at `min(limit_price, open * 1.005)`.
- If T+1 open > limit price but T+1 low <= limit price, fill at the precommitted limit price.
- If T+1 low > limit price, the order does not fill.
- No chasing, no intraday repricing after seeing T+1 movement.
- An unfilled T+1 order expires that day. A new order requires a fresh T+1 close decision for T+2.

## Rule 3 — SELL at next open

- A SELL signal generated at T close must explicitly output the stock, share quantity, and whether the position is fully exited.
- Execution occurs at T+1 open.
- Backtest sell execution price is conservatively modeled as `T+1 open * 0.98`.

## Rule 4 — Earliest exit is D2

- Entry fill day is D1.
- No same-day sell on D1.
- A SELL signal may be generated at D1 close and executed at D2 open.
- There is no mandatory three-day holding period.

## Rule 5 — Capital allocation

- Number of holdings is not fixed; the strategy decides based on market conditions and candidate quality.
- Total equity exposure is not fixed; the strategy may hold high cash or be fully in cash when appropriate.
- Market regime may determine total capital deployment.
- A single stock may never exceed 20% of total portfolio NAV.

## Rule 6 — Share quantity

- Whole board lots are preferred; one board lot is 1,000 shares.
- When odd lots are required, position sizing uses 100-share increments only, such as 400, 500, or 700 shares.
- Quantities such as 437 or 583 shares are invalid.
- Position sizing must never breach the 20% single-stock NAV cap.

## Rule 7 — Trading costs and ledger

- Brokerage commission is 0.1425% on buys and sells, with no discount assumption.
- Applicable transaction tax is charged separately on sells.
- Slippage, commissions, taxes, and net P&L must be recorded separately for every completed trade.
- Performance is evaluated only after all modeled costs.

## Rule 8 — Exit logic belongs to strategy research

- No fixed stop-loss or take-profit percentage is imposed in advance.
- Strategy research may evaluate trend breaks, moving averages, momentum, volume/price behavior, institutional flow, market regime, trailing exits, holding duration, and other causal exit logic.
- All exit logic must obey Rules 1–7.

## Rule 9 — Core data sources

The first rebuilt strategy generation uses only:

- TWSE / TPEx daily OHLCV.
- Three-institution trading data: foreign investors, investment trusts, and dealers.
- 0050 / broad-market trend data.
- Causal derived features such as moving averages, returns, breakouts, volatility, volume behavior, and relative strength.

News, narrative fundamentals, and historically hard-to-align data are excluded from the first rebuild.

## Rule 10 — Optimization objective

- Maximum Drawdown must be strictly below 20%.
- Any strategy with Max Drawdown >= 20% is rejected.
- Among strategies that pass the drawdown and robustness requirements, maximize CAGR.
- 20% is a rejection boundary, not a target; lower drawdown is preferred all else equal.

## Rule 11 — Anti-overfitting validation

- 2020 is warm-up only.
- 2021–2025 is the formal historical evaluation period.
- Evaluate full-period and year-by-year results.
- Require walk-forward / out-of-sample validation.
- Reject parameter choices that work only in a narrow historical segment.
- Prefer stable parameter regions over isolated best points.
- Run stress tests with worse execution assumptions and reduced fill quality.
- All reported metrics must include modeled transaction costs.

## Rule 12 — Strategy freedom, zero future leakage

- The optimizer may research stock selection, entry price logic, exit logic, holding duration, number of holdings, and market-regime capital allocation.
- Rules 1–11 are immutable and must not be optimized away.
- No look-ahead bias, future functions, hindsight fills, future institutional data, future OHLC, or any other future information is permitted.

## Enforcement principle

A strategy result is invalid if it violates any single hard rule above, regardless of CAGR or other performance metrics.
