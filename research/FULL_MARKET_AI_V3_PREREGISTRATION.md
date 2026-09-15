# Full-Market AI V3 — preregistration

V2 is frozen as a failed experiment. V3 addresses a structural objective mismatch observed in V2: positive cross-sectional alpha can still correspond to a negative absolute investment return. No V2 portfolio outcome is used to tune numeric cutoffs.

## Locked before V3 OOS run
- Full executable TWSE/TPEx universe remains visible to AI; R10 hard eligibility is NOT a candidate gate.
- Same immutable 2020–2025 inputs, 60-session purge, expanding annual OOS folds, T+1 fills, fees/tax, integer shares, common cash pool and R10 execution constraints.
- Context clustering remains causal and trained only on pre-fold history.
- Train two return objectives on the same pre-fold rows: 20-session absolute net return and same-date cross-sectional 20-session alpha, plus the existing failure classifier.
- Selection is sparse: at most 5 names per decision date, matching portfolio capacity.
- A name may be selected only when predicted absolute return > 0, predicted alpha > 0, and predicted failure probability < 0.50. These are sign / neutral-probability gates, not thresholds tuned from V2 outcomes.
- Within eligible names, rank score is fixed ex ante: 45% absolute-return percentile + 35% alpha percentile + 20% safety percentile.
- No R10 candidate flag is used in the score or eligibility.
- R10 remains immutable benchmark.

## Success gate
Research success requires ALL: CAGR > locked R10; profit factor >= locked R10; Max DD no worse than R10 by more than 1 percentage point; contract audit passes. Model scores alone never constitute success.