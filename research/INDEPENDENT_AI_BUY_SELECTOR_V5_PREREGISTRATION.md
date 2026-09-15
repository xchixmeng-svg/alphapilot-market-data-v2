# AlphaPilot Independent AI Buy Selector V5 — Bottleneck Quality + Precision-Controlled Abstention

## Why V5 exists
V4 established that a raw-tape AI model can produce positive OOS absolute return, positive market-relative alpha, and shallower MAE without a fixed Top-N rule. However, V4's binary classifier did not separate genuinely exceptional buys strongly enough: its preregistered precision-lift gate failed and it emitted too many candidates. V5 changes the learning target and the abstention policy, not the raw information set.

## Hard independence
- No R10/R7/R0.5 engine, score, candidate pool, regime, entry, exit, sizing, holding-period, stop-loss, take-profit, or portfolio rule.
- No V1/V2/V3 hand-engineered factor module.
- The input remains the V4 minimally transformed raw temporal tape only.
- R10 may only be used later as an external benchmark.

## Input representation
Use the same preregistered raw temporal tape as V4 at lags {0,1,2,5,10,20,60,119}: one-session close return, open gap, intraday range, close/open body return, traded-amount log change, foreign net/volume, trust net/volume, same-day market median one-session return, advance fraction, and return dispersion.

## Research outcomes
Horizons are labels/diagnostics only, never holding rules. For each stock/date, calculate future 20/60/120 endpoint return and 60/120 MAE. Convert each of those five quantities to same-date cross-sectional percentile ranks, with higher MAE rank meaning less adverse excursion.

## Bottleneck target
Instead of V4's binary target, V5 learns a continuous joint-quality target:
- if any of fwd20/fwd60/fwd120 is non-positive, joint quality = 0;
- otherwise joint quality = minimum of rank(fwd20), rank(fwd60), rank(fwd120), rank(MAE60), rank(MAE120).

This is a weakest-link objective. A stock cannot compensate for one poor horizon or poor path resilience by being spectacular on another dimension. There are no manually chosen weights.

The V4 `worth_buy_label` remains only an evaluation/calibration event: all three endpoint returns positive and all five percentile ranks above 0.5.

## Model
One fixed HistGradientBoostingRegressor on the raw tape. No OOS hyperparameter search.

## Precision-controlled abstention
There is still no fixed Top-N and no requirement to emit a candidate.

For every OOS fold:
1. Reserve the trailing 252 sessions inside the causally available training history as a calibration block.
2. Fit the model only on history before that block.
3. Score the calibration block.
4. Evaluate only the fixed score-tail quantiles {0.90, 0.95, 0.975, 0.99, 0.995}.
5. A threshold is eligible only if the calibration tail contains at least 200 labeled observations, observed `worth_buy_label` precision lift is >= 1.50x the calibration-universe base rate, and the 95% Wilson lower bound of precision is above the calibration-universe base rate.
6. Choose the least stringent eligible threshold, preserving more candidates while still demonstrating calibrated enrichment.
7. If none is eligible, the fold threshold is infinity and the selector abstains for that OOS year.
8. Refit the model on all causally available training data, then apply the learned raw-score threshold to the OOS year without modification.

This rule is fixed before V5 OOS results. The 1.50x target is inherited directly from the already-preregistered V4 research gate, not chosen after seeing V5 outcomes.

## OOS protocol
- Training history begins in 2016.
- Test years: 2021, 2022, 2023, 2024, 2025.
- 120-session purge before each test year.
- Same corporate-action/data discontinuity invalidation as V4.
- No OOS year is used to tune the model, tail quantiles, or threshold.

## Research gate
V5 is promising only if all are true:
1. Overall selected mean absolute return > 0 at 20/60/120.
2. Overall selected market-relative alpha > 0 at 5/20/60/120.
3. At least 4/5 OOS years that emit candidates have positive absolute mean return at 20/60/120; additionally candidates must be emitted in at least 3 of 5 OOS years.
4. Median yearly selected precision lift across emitting years >= 1.50x.
5. Selected MAE is no worse than the universe at 60 and 120 sessions.
6. The system demonstrates genuine abstention on at least one OOS date.
7. Total OOS selected rows are > 0 and materially below the V4 selected count; this is reported, not separately tuned.

Research-gate success is still not a complete trading strategy: no sell rule, holding rule, sizing rule, CAGR, PF, or portfolio DD is defined.

## Live inference
For the latest data date, train only on observations whose full 120-session label is already knowable. Use the same 252-session internal calibration and fixed precision-controlled abstention algorithm. Output BUY_CANDIDATES only when the calibration rule proves sufficient enrichment; otherwise output NONE.
