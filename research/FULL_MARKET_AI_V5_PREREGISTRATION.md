# Full-Market AI V5 — Held-Out Calibration Gate

## Motivation
V4 proved that a model-predicted conservative quantile is not itself a reliable cash-allocation gate. V5 does not tune the V4 q25 threshold. Instead it separates model fitting from an independent prior-only calibration window and asks whether the strongest signals actually earned positive net returns in that held-out past window before allowing capital to be deployed.

## Locked invariants
- R10 MAX benchmark engine remains immutable at SHA256 `2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0` and formal commit `3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9`.
- Same causal OHLCV/corporate-action data, T+1 fixed-limit fills, fees, tax, slippage, integer shares, common cash pool, max 5 positions, 25% single-name cap and 95% total exposure.
- R10 `r7_hard` / `r05_hard` flags are never entry eligibility gates for AI.
- No test-year data may influence model fitting or calibration.

## Outer OOS folds
2021 through 2025 are evaluated independently. For each test year, the outer training cutoff remains 60 trading sessions before the first test date.

## Inner held-out calibration
For each outer fold:
1. Reserve the last 60 trading sessions before the outer cutoff as the calibration window.
2. Leave a further 20-session gap before calibration.
3. Fit the stock-selection models only on data strictly before that inner model cutoff.
4. The calibration window is never used to fit the stock models.
5. Calibration labels are fully observable before the test year because of the outer 60-session purge.

## Model and context
- Full executable Taiwan-equity universe: valid 4-digit code, non-KY, 20-day traded value >= NT$30m, sufficient MA120 history.
- Three market-context clusters are learned only from prior market data.
- Fixed HistGradientBoosting family predicts 20-session net return, 20-session cross-sectional alpha, and <= -5% failure probability.
- Daily score = 45% return rank + 35% alpha rank + 20% safety rank.

## Calibration policy gate
For each context cluster, inspect only calibration observations in the daily top 1% of model score. A context is authorized for the next OOS year only when:
- at least 50 filled calibration observations exist;
- mean realized 20-session net return minus one standard error is > 0;
- realized calibration profit factor is >= 1.0.

These thresholds are fixed before the V5 run. They are not adjusted using V5 results.

## Test-year decision
A stock may use a portfolio slot only when all are true:
- current context is calibration-authorized;
- daily model-score percentile >= 99%;
- predicted absolute net return > 0;
- predicted cross-sectional alpha > 0;
- predicted failure probability < 50%.

At most 5 names per day are passed to the unchanged execution layer. The system may stay fully in cash.

## Success gate
V5 is successful only if full 2021–2025 portfolio replay satisfies all:
- CAGR > locked R10 CAGR 13.110267%;
- profit factor >= 1.862730946;
- max drawdown no worse than R10 by more than 1 percentage point;
- all data/contract audits PASS.

Model diagnostics, calibration statistics, or a higher win rate alone are never success.