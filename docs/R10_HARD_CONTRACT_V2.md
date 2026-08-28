# AlphaPilot R10 — HARD CONTRACT v2

This contract supersedes any research-factory parameter search that conflicts with the locked R10 execution and portfolio rules. A run is **INVALID regardless of CAGR** if any item below is violated.

## 1. Decision clock / causality

- Every signal and every BUY/SELL decision uses information available by **T close only**.
- BUY created at T close is a fixed, precommitted **T+1 limit order**.
- If T+1 market never touches the locked limit, there is **no fill and no chase**.
- SELL created at T close executes at **T+1 open** using the locked adverse sell-slippage model.
- No T+1 intraday information may revise quantity, limit, candidate rank, allocation, or exit decision.
- A queued T+1 sell still occupies its stock slot and counts in T-close exposure.
- A T+1 open sale may free cash/slot, but it may not retroactively create a new T+1 buy. The next new decision is made only at T+1 close for T+2.

## 2. Minimum holding period

- **Minimum holding period = 3 trading days.**
- Entry session counts as held session 1 at that session's close.
- No HARD, rank exit, regime exit, exposure exit, trail exit, time exit, force-DD exit, or any other strategy exit may create a SELL order while `hold_days < 3`.
- Therefore no completed trade may report holding days 1 or 2.
- Any backtest containing a completed trade with `hold_days < 3` is invalid.

## 3. Transaction costs — raw executable prices only

All fees/taxes are calculated from the **actual raw executable trade price × integer shares**. Adjusted/total-return prices may be used for signals and corporate-action continuity, but never as the fee/tax base.

- Buy fee = raw buy gross × **0.000855**.
- Sell fee = raw sell gross × **0.000855**.
- Sell tax = raw sell gross × **0.003**.
- Buy total cost = raw buy gross + buy fee.
- Sell net proceeds = raw sell gross − sell fee − sell tax.
- Net P&L = sell net proceeds − buy total cost.
- Trade return = net P&L ÷ buy total cost.

Every exported trade row must explicitly contain:
`buy_raw_price`, `shares`, `buy_gross`, `buy_fee`, `buy_total`, `sell_raw_price`, `sell_gross`, `sell_fee`, `sell_tax`, `sell_net`, `net_pnl`, `net_return`.

A buy and sell at the same displayed price can never report 0% return because fees/tax still apply.

## 4. Capital allocation — NO OVERWEIGHT

Capital allocation is a hard portfolio rule, **not an optimizer-owned parameter**.

- One common cash pool; no permanent 62/38 buckets.
- R7 new-position base target = **22% of current NAV** before DD throttle/caps.
- R0.5 / replacement-alpha new-position base target = **20% of current NAV** before DD throttle/caps unless a future strategy is explicitly assigned a different locked base by the user.
- Maximum **5 distinct stocks** at once.
- Same stock across all strategy sleeves combined <= **25% NAV**.
- Total stock exposure at T order construction <= **95% NAV**.
- Therefore at least ~5% NAV remains outside new stock exposure under normal construction.
- No borrowing, no leverage, no shorting.
- Planned T+1 sell proceeds may not be used to finance a new T-day order because they are unknown at T close.
- T-day BUY orders reserve their fee-inclusive known cash cumulatively. Later orders may use only unreserved T cash.

The optimizer is prohibited from searching or changing:
`max_positions`, `max_single`, `max_total`, R7 base size, R0.5 base size, DD throttle, force-DD capital rules, fee/tax rates, slippage model, minimum hold, execution clock, or share-integrality rules.

## 5. DD capital controls

Locked R10 new-position sizing throttle:

- DD > -6%: 100% of base target.
- DD <= -6%: 85%.
- DD <= -9%: 45%.
- DD <= -15%: 40%.

Locked DD defense:

- At DD <= -14%, create weakest-first T-close sells for T+1 until target stock exposure is approximately 50%.
- No new buys for 10 trading days.
- Force-DD cooldown = 15 trading days.
- The minimum-3-day hold rule still applies; the DD engine may not create an exit for a position before it is eligible to sell.

## 6. Share quantity / board lot / odd lot

- Shares are always whole integers.
- Board lot = 1,000 shares.
- Board-lot first.
- If one full board lot fits inside the intended base position target, quantity must be an integer multiple of 1,000, rounded down by cash/liquidity limits.
- Odd lots are allowed only for a genuinely high-priced stock where one full 1,000-share lot itself exceeds the intended base position target.
- A tiny leftover portfolio capacity may **not** create nonsensical low-price 1-share/2-share positions. In that case the order is skipped.
- Per-order quantity <= 2% of T-known 20D average share volume.

## 7. Stock/slot collision rules

- A stock code already held by any sleeve is one portfolio stock exposure; another sleeve may not create a duplicate independent position in the same code.
- Same-code exposure across sleeves is always combined for the 25% cap.
- Pending sells remain held until T+1 execution and therefore keep their slot at T close.
- A full 5-stock portfolio cannot create a sixth T+1 BUY merely because one current position is queued to sell the next morning.

## 8. Execution order at T+1

- Execute already-committed sells first at T+1 open.
- Then process already-committed buys using the exact T-close quantities/limits/priorities.
- Sell proceeds may make enough actual cash available for a previously committed buy, but may never increase that buy's precommitted quantity.
- If actual cash is still insufficient, the buy does not fill; no resizing with T+1 hindsight.

## 9. What the optimizer MAY search

Within the hard contract it may search alpha/research logic such as:

- signal features and feature weights;
- eligibility thresholds;
- market/regime definitions only when not overriding the hard portfolio cap;
- candidate ranking/depth;
- legal precommitted limit-price multiplier;
- exit thresholds/trailing parameters/max-hold rules, subject to minimum 3-day hold;
- strategy activation/mix, subject to the common portfolio/capital rules.

It may not search its way around the hard contract.

## 10. Required audit before accepting any performance result

A result is accepted only if all are PASS:

1. T-close causality / Strict T+1 contract.
2. `min(completed_trade.hold_days) >= 3`.
3. Integer shares for every order/fill/trade.
4. No residual low-price 1–999 share position when a board lot fits the intended base target.
5. Maximum distinct held stocks <= 5.
6. Same-stock combined T-order exposure <= 25% NAV.
7. Total T-order stock exposure <= 95% NAV.
8. No negative cash / leverage.
9. Pending sells still occupy T slots/exposure.
10. Raw-price fee recomputation exactly matches exported buy fee, sell fee, tax, net proceeds, P&L and return.
11. No untouched T+1 limit counted as a fill.
12. No T+1 hindsight changes to order quantity/price/priority.

If any audit fails, End NAV, CAGR, Max DD, win rate and all other performance statistics from that run are **void**.

## 11. Versioning

- Contract name: `R10-HARD-CONTRACT-v2`.
- Canonical machine constants/helpers: `scripts/r10_hard_rules.py`.
- Research factories must call the hard-rule layer before simulation and before accepting/exporting a winner.
- Prior Trial 292 / 40.14% output is invalid because it violated this contract (minimum holding-period and audit-output requirements) and must not be used as a performance reference.
