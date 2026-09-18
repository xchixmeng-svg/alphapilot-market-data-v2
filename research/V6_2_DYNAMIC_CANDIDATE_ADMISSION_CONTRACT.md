# V6.2 Dynamic Candidate Admission Contract

Status: DEVELOPMENT ONLY. 2022-2024 only. 2025 sealed. 2026 live-only/sealed. Formal R10 is outside this branch and MUST NOT be modified.

## Objective
V6.2 is an abstaining full-market candidate admission engine, not a daily ranking engine. Each stock is judged independently. A day may contain zero, one, or many candidates. Top-N is forbidden as an admission rule; ranking is display-only after admission.

## Admission output
Every stock-date must emit one of REJECT / WATCH / CANDIDATE / HIGH_CONVICTION and preserve the continuous admission confidence. CANDIDATE/HIGH_CONVICTION form the dynamic pool.

Each admitted stock must retain: evidence-family flags, estimated barrier probabilities for +5/+10/+20%, expected path/target zone, horizon diagnostics (not a forced holding period), predicted downside/MAE, invalidation evidence, valuation/industry/flow/event/price-structure evidence where available.

## Evidence routing
Evidence families are independent. A stock is not required to activate every family. The engine may admit different stocks for different evidence combinations. Missing/non-applicable families must not be treated as negative evidence by default.

## Corporate-action safety
All price-derived features and forward labels must remain inside V6.1 price_segment_id boundaries. No path may cross a reference-price reset/corporate-action boundary.

## Development protocol
Only 2022-2024 may be used for development/repair validation. 2025 remains untouched until an explicit model/admission lock. 2026 remains live-only/sealed.

Admission thresholds must be learned from prior/development training support without using the evaluated day's cross-sectional rank or future labels. A threshold may be calibrated from training predictions/outcomes, but cannot be selected to force a candidate count.

## Required quality report
For each year and aggregate report:
- candidate count distribution per day, including zero-candidate days
- admission coverage (candidate stock-days / eligible stock-days)
- +5%, +10%, +20% hit rates
- full-market eligible base rates and candidate lift for each barrier
- MFE and MAE distributions
- time-to-hit distributions for successful barriers
- confidence calibration/reliability bins
- performance by evidence-family combination
- candidate count versus quality diagnostics

## Fail conditions
The layer is not qualified if it cannot abstain; if every eligible day is forced to have a candidate; if candidate count is fixed by Top-K; if lift over full-market base rate is absent/unstable; if confidence is materially miscalibrated; or if corporate-action boundary violations are non-zero.

No workflow completion, rerun, empty commit, or unchanged output constitutes layer completion. Layer completion requires new quality evidence satisfying this contract.
