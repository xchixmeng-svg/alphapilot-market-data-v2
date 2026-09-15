# Full-Market AI V4 — Decision-Level Opportunity-Cost Gate

## Purpose
V1–V3 established that full-market OOS selection can discover names outside R10, but relative/ranking skill did not translate into superior portfolio returns. V4 tests a different question: should a candidate occupy scarce portfolio capital at all?

## Frozen before V4 results
- Universe remains the full executable Taiwan-equity universe from the immutable causal market-data pipeline. R10 `r7_hard` / `r05_hard` are not AI eligibility gates.
- Locked R10 remains the benchmark and its T+1 execution, raw-price fills, fees/tax/slippage, integer shares, common cash pool, corporate actions, 25% single-name cap, 95% total cap and max-five-position mechanics remain unchanged.
- 60-session purge and expanding-year OOS structure remain unchanged.
- Market context is still discovered using only prior data with the same 3-cluster KMeans context router.
- No V3 portfolio outcome is used to tune numeric thresholds.

## V4 change
V4 adds a conservative conditional-return model using `HistGradientBoostingRegressor(loss='quantile', quantile=0.25)` on the same causal features and prior-only training rows.

For each stock/date the AI predicts:
1. mean 20-session net return,
2. 25th-percentile 20-session net return,
3. relative-to-same-day-market 20-session alpha,
4. probability of a <= -5% 20-session outcome.

A stock may consume a portfolio slot only when all four conditions hold:
- predicted mean net return > 0,
- predicted 25th-percentile net return > 0,
- predicted relative alpha > 0,
- predicted failure probability < 0.50.

This makes cash (0% return) an explicit hurdle. The AI is allowed to leave slots empty. Among qualified names, daily priority is determined by the conservative 25th-percentile return; at most five names are exposed to the locked execution engine.

## Research success gate
V4 is successful only if the full portfolio replay simultaneously satisfies:
- CAGR > locked R10 CAGR 13.110267%,
- profit factor >= locked R10 PF 1.862730946,
- Max DD no more than 1 percentage point worse than locked R10 (-19.515439%),
- all data and contract audits PASS.

Model diagnostics, ranking correlation, candidate hit rates, or CI success are not research success.