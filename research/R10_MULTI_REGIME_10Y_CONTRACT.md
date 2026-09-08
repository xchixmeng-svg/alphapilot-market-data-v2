# AlphaPilot Multi-Regime 10-Year Research Contract

## Isolation

- Formal R10 MAX remains frozen at commit `3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9`.
- This branch is research-only and may not overwrite the formal strategy.
- Reuse existing audited data; do not refetch historical data.
- 2015 is warm-up. Every decision uses information available by T close and executes on T+1.
- Integer shares, common cash pool, original fees, tax, slippage and official corporate actions remain mandatory.

## Causal market regimes

The controller classifies every trading day from observable data, never from calendar-year labels:

1. Crisis/Bear: 0050 below MA120 with negative 60-day return, severe 20-day decline, or breadth collapse.
2. Repair: 0050 near or below MA120, positive 20-day return and improving breadth.
3. Calm Bull: 0050 above MA120, positive 60-day return, moderate breadth and subdued realized volatility.
4. Momentum Bull: 0050 above MA60 and MA120, positive 20/60-day returns and broad participation.
5. Distribution/Weak: index remains elevated while breadth deteriorates, momentum stalls or volatility expands.

Thresholds are selected only in the development window and then frozen.

## Coexisting sleeves

- R7 momentum sleeve: largest allocation in Momentum Bull; smaller allocation in Calm Bull and Repair.
- R0.5 breakout sleeve: enabled only in regimes where its development-window conditional PF is acceptable.
- Calm-trend sleeve: liquid stocks above MA120 with lower volatility, positive medium-term relative strength and non-deteriorating institutional flow.
- Cash/defense sleeve: absorbs unused budget in Distribution, Bear and Crisis.
- Holding Sentinel remains an external live risk layer. Historical news protection is not claimed without point-in-time news archives.

## Validation protocol

1. 2016-2019: develop the calm-trend sleeve and regime allocation.
2. 2020-2022: sealed walk-forward validation across crash, rebound and bear market.
3. 2023-2025: final untouched validation.
4. 2016-2025: continuous common-pool replay only after all rules are frozen.

No rule may use future data, calendar-year identity or hindsight regime labels.

## Acceptance gate

- Target Max DD at or below 22%; reject above 25%.
- Profit factor at least 1.50.
- Preserve nonnegative cash, five-position limit, 95% total order cap and 25% single-stock order cap.
- Report CAGR, annual returns, PF, win rate, exposure, worst trade and large-loss counts.
- Report dependence on the five largest winners and terminal open positions.
- Compare against locked R10 and 0050 in every sealed segment.
- Prefer the simplest rule set that passes; do not select solely by maximum terminal wealth.
