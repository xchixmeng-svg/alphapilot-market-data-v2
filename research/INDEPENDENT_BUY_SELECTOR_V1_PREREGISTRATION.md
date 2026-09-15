# AlphaPilot Independent Buy Selector V1 — Preregistration

## Purpose
Build an AI selector that is fully independent from R10. The system's only job is to rank Taiwan-listed stocks by how attractive they are to buy now.

## Independence contract
- No import of any R10 script, signal, filter, score, execution rule, holding rule, exit rule, or portfolio rule.
- No use of R10 candidate membership as a feature or label.
- No R10 fingerprint check in the research workflow.
- R10 may only be compared later as an external benchmark after this selector is evaluated on its own terms.

## Inputs
Only immutable market data under `data/history/2020-2025/`:
- OHLCV 2020-2025 parquet files
- institutional_2020_2025.parquet
- SHA256SUMS / history audit

## Universe
- Taiwan common-stock-like 4-digit codes.
- Exclude obvious KY names.
- Require sufficient historical observations and rolling traded value for model stability / practical tradability.
These are data-quality / feasibility requirements, not alpha rules.

## What the model predicts
The model does NOT predict a fixed holding period and does NOT define an exit.
For research labels only, each historical stock/date observation is evaluated at several future horizons (5, 20, 60, 120 trading sessions). These horizons are diagnostic views of future path quality, not instructions to sell on those dates.

A future-path quality target is built from cross-sectional forward-return ranks across all available horizons. The goal is to identify stocks that later became persistent market leaders rather than optimizing one arbitrary horizon.

## Causal validation
- 2020 provides initial history/training only.
- 2021-2025 are expanding out-of-sample years.
- Before each test year, a 120-session purge is applied so no forward target information overlaps the test period.
- Features at date T use only data available by T close.
- No post-hoc threshold tuning from test-year results.

## Model output
For every test date, rank the full executable universe and save the Top 10 with:
- AI score
- rank
- core causal features
- realized forward diagnostics (research audit only, never available to live selection)

The selector itself outputs only ranked buy candidates. It does not output a holding duration, stop-loss, take-profit, portfolio allocation, or sell rule.

## Evaluation
Because this research intentionally has no exit rule, CAGR/PF/Max-DD are not valid metrics for this stage and must not be fabricated.
Evaluate selection quality using:
- annual and overall rank IC between AI score and future-path quality
- Top-10 mean return at 5/20/60/120-session diagnostic horizons
- Top-10 excess return vs same-day market median at each horizon
- positive-return hit rate and positive-excess hit rate at each horizon
- year-by-year stability
- candidate concentration / turnover diagnostics

A model score improvement alone is never called success. The selector is promising only if its selected stocks show robust positive excess performance across multiple horizons and multiple OOS years.

## Research-change rule
If results are far from robust, change the framework/target/features rather than micro-tuning cutoffs or weights. Micro-tuning is allowed only after the selector is already close to a robust result.