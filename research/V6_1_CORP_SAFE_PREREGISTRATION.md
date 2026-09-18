# V6.1 Corporate-Action-Safe Repair Preregistration

Status: PRELOCK
Date: 2026-09-18
Parent: V6 CPU Quantile R2 lock / frozen Stage0F panel
Purpose: repair only the raw-price label/stock-path contamination discovered after V6 Actionability Audit.

## Non-negotiable isolation
- R10 remains untouched.
- Frozen Stage0F source panel remains unchanged and must match SHA256 `afd4b86fafed8d3ab576bf5fcc39e6a8ed5fbd0a015909534fe7ac070628c358`.
- No V6 model hyperparameter is retuned from OOS results.
- 2025 remains sealed until the V6.1 repair-validation gate passes.
- 2026 remains live-only / sealed from research evaluation.

## Corporate-action / reference-price-reset-safe path rule
A ticker row is a reset boundary when absolute raw close-to-close return is strictly greater than 20%.

For each ticker:
1. increment `price_segment_id` at every reset boundary;
2. all stock-derived returns, lags, rolling ranges/volatility, seasonal stock returns and forward labels are computed only inside the same segment;
3. a forward label is invalid if its target session is not in the same segment;
4. the reset day itself has no prior-close return/gap/range observation;
5. a ticker becomes model-eligible only after 120 observations inside its current segment;
6. no imputation is allowed to bridge a reset boundary.

Rationale: Taiwan ordinary daily trading moves do not explain the previously observed +800% paths; the >20% rule is a conservative fail-closed detector for capital reduction, relisting/reference-price reset, split-like or other discontinuous price-basis events. It is not used to claim the exact corporate-action type.

## Frozen model family
Unchanged from V6:
- HistGradientBoostingRegressor quantile
- quantiles: 0.10, 0.25, 0.50, 0.75, 0.90
- horizons: 1..120 sessions
- audit horizons: 5, 10, 20, 40, 60, 120
- purge: 120 sessions
- max long train rows: 480000
- seed: 926616
- learning_rate 0.05
- max_iter 160
- max_leaf_nodes 31
- min_samples_leaf 120
- l2_regularization 4.0

## Validation chronology
- 2021 is NOT a formal V6.1 score year because the frozen Stage0F starts too late to satisfy the already-fixed 120-session in-segment warmup plus 120-session purge and all six naive-horizon support requirements. This is a support constraint discovered before V6.1 lock, not a model-score exclusion.
- V6.1 repair-validation: 2022, 2023, 2024.
- These years are not described as untouched confirmation because prior V6 diagnostics were already viewed.
- 2025 is the first still-sealed confirmation year and may be opened only after the locked repair-validation gate passes.
- 2026 remains sealed/live-only.

## Repair-validation gate
The original V6 gate is preserved rather than tuned:
- six horizons 5,10,20,40,60,120;
- stationary block bootstrap mean block 20, 2000 replicates;
- Bonferroni family alpha 0.05;
- significant pinball improvement at >=4 horizons;
- significant degradation at 0 horizons;
- MAE improved at >=4 horizons;
- q10-q90 coverage not materially worse at >=5 horizons with absolute coverage-error tolerance 0.03.

If the gate fails: STOP V6.1, no retune on the same scored years, keep 2025 sealed.

## Human-readable Actionability Audit
After the repair-validation run, produce Top1/Top3/Top5 diagnostics:
- positive terminal-return rate;
- +5%, +10%, +20% hit rate;
- maximum favorable excursion;
- maximum adverse excursion;
- sessions to threshold;
- q50 target error.

This audit is descriptive and cannot modify the locked model or gate.
