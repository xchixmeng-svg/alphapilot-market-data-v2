# AlphaPilot Independent Buy Selector V3 — Path Quality

## Research question
Can a fully independent full-market selector identify Taiwan stocks that are not merely relatively strong at fixed endpoints, but exhibit a better post-selection price path: stronger upside opportunity, persistent endpoint strength, and smaller adverse excursion?

## Hard independence
- No R10 engine, candidate pool, score, regime, entry, exit, sizing, holding-period, stop-loss, or portfolio rule.
- Reads only immutable 2020–2025 OHLCV and institutional-flow history.
- R10 is not used as an input or label.

## What V3 is and is not
V3 only ranks stocks by buy-candidate quality. It defines no holding period and no sell rule.
The 20/60/120-session horizons below are outcome-characterisation windows used only for research labels and diagnostics. They are not mandatory holding periods.

## Features
Exactly the causal feature set already preregistered in Independent Buy Selector V1: price momentum/reversal, volatility, moving-average distance, distance from highs, liquidity/volume acceleration, close-location/range, point-in-time institutional flow, same-day cross-sectional ranks, and same-day market breadth/context.

## OOS protocol
- Expanding yearly OOS folds: 2021, 2022, 2023, 2024, 2025.
- 120-session purge before each test year.
- Full causally available training data; no every-fifth-date sampling.
- Same full executable universe/data-quality filter as V1/V2.
- Top 10 per signal date for diagnostics only.

## Path-quality target
For each stock/date and each research horizon H in {20,60,120}:
- endpoint return = close[t+H] / close[t] - 1
- MFE(H) = maximum close from t+1..t+H / close[t] - 1
- MAE(H) = minimum close from t+1..t+H / close[t] - 1

Within each signal date, convert endpoint return, MFE and MAE to cross-sectional percentile ranks (higher MAE rank means less adverse excursion / better resilience). The target is the equal-weight mean of all nine percentile ranks. Equal weighting is structural and fixed before OOS results; no post-result tuning.

Corporate-action/data discontinuities use the same V1 invalidation logic.

## Model
One fixed HistGradientBoostingRegressor using the same fixed hyperparameters as V1. No search or tuning on 2021–2025 outcomes.

## Evidence reported
For each OOS year and overall:
- selected mean endpoint return and same-date market-relative alpha at 5/20/60/120 sessions
- selected mean MFE and MAE at 20/60/120 sessions
- corresponding executable-universe MFE/MAE means
- yearly rank IC against path-quality target
- candidate counts and unique codes

## Research gate
V3 is only called promising if all are true:
1. Overall selected mean absolute endpoint return > 0 at 20/60/120.
2. Overall selected mean relative alpha > 0 at 5/20/60/120.
3. At least 4/5 OOS years have positive absolute mean endpoint return at 20/60/120.
4. Selected mean MFE exceeds executable-universe mean MFE at 20/60/120.
5. Selected mean MAE is no worse (greater/less negative) than executable-universe mean MAE at 20/60/120.
6. Median yearly rank IC > 0.

This is a selector research gate only, not a complete trading-strategy success claim.
