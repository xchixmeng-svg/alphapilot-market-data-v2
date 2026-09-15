# AlphaPilot Independent AI Buy Selector V4 — Raw Tape + Abstention

## Goal
Build the first version that behaves like a genuine independent buy selector rather than a fixed Top-N ranking engine. The system must decide which Taiwan stocks are worth buying now, and it must be allowed to return NONE on any date.

## Hard independence
- No R10/R7/R0.5 engine, score, candidate pool, regime, entry, exit, sizing, holding-period, stop-loss, take-profit, or portfolio rule.
- R10 may only be used later as an external benchmark.
- No inheritance of V1/V2/V3 engineered factor list as the model feature module.

## Universe
All four-digit Taiwan equity codes beginning 1-9 with valid OHLCV and at least 120 observed sessions. No alpha screen, market-regime screen, liquidity ranking, or fixed candidate pool is applied before AI scoring.

## Data
- 2016-2019 immutable full-market TWSE/TPEx OHLCV archive.
- 2020-2025 immutable full-market OHLCV plus point-in-time institutional flows.
- 2026 YTD data is rebuilt from the existing official-data pipeline: OHLCV weekly release assets plus official TWSE T86 / TPEx institutional APIs, then merged with repository daily folders so current inference reaches the latest repository date.

## Input representation
V4 does not feed the old momentum/MA/high-distance/volatility factor module to the model. It feeds a minimally transformed raw temporal tape sampled at lags {0,1,2,5,10,20,60,119}:
- one-session close return
- open gap vs prior close
- intraday high-low range vs prior close
- close/open body return
- traded-amount log change
- foreign net shares / daily volume
- investment-trust net shares / daily volume
- same-day market median one-session return
- same-day advance fraction
- same-day cross-sectional return dispersion

The lagged tape is causal. Historical-flow absence before 2020 is represented as zero flow rather than backfilled future information.

## Supervised definition of a genuinely good buy
Research horizons are labels only; they are not holding rules.
For each stock/date, compute future 20/60/120-session endpoint returns and 60/120-session MAE. A positive training label requires all of the following simultaneously:
- absolute 20/60/120-session endpoint returns are positive;
- the stock is above the same-date cross-sectional median for 20, 60 and 120-session return;
- the stock is above the same-date cross-sectional median for 60 and 120-session MAE (smaller adverse excursion).

This is deliberately conjunctive rather than a manually weighted score: a stock cannot compensate for a bad long-horizon path with one spectacular short-horizon metric.

## Model
One fixed HistGradientBoostingClassifier with balanced classes. No hyperparameter search on 2021-2025 OOS results.

## Learned abstention threshold
There is no fixed Top-N and no requirement to emit a stock every day.
For each OOS fold, an internal trailing 252-session calibration block inside the training history is used only to learn the probability threshold. The threshold is the larger of 0.50 and the model-score quantile implied by the observed positive-label prevalence in that calibration block. The final fold model is then refit on all causally available training rows and applied to the OOS year.

A stock is emitted only if its buy probability is at or above the learned threshold. Candidate count may be zero, one, or many.

## OOS protocol
- Test years: 2021, 2022, 2023, 2024, 2025.
- 120-session purge before each test year.
- Training begins in 2016 so the earliest fold sees several different market regimes.
- Corporate-action/data discontinuities greater than 35% in one raw-close session invalidate affected future labels.
- OOS test rows are never used to set model parameters or thresholds.

## Evidence reported
For each OOS year and overall:
- candidate count, signal days and explicit no-signal days;
- future 5/20/60/120-session absolute return and same-date market-relative alpha;
- 20/60/120 MFE and MAE versus the executable universe;
- realized positive-label precision, universe base rate and precision lift;
- learned threshold.

## Research gate
V4 is only called promising if all are true:
1. Overall selected mean absolute return is positive at 20/60/120 sessions.
2. Overall selected mean relative alpha is positive at 5/20/60/120 sessions.
3. At least 4/5 OOS years have positive selected mean absolute return at 20/60/120 sessions.
4. Median yearly positive-label precision lift is at least 1.5x the universe base rate.
5. Selected MAE is no worse than the universe at 60 and 120 sessions.
6. The selector demonstrates real abstention on at least one OOS date.
7. The selector emits at least one OOS candidate.

Research-gate success is not a complete trading-strategy claim because V4 still has no sell rule, holding-period rule, sizing rule, CAGR, PF or portfolio DD definition.

## Live inference
After OOS evaluation, train only on labels whose full 120-session future is already observable as of the latest data date, calibrate the threshold using training history only, and output the latest-date full-market file with BUY_CANDIDATES or NONE.
