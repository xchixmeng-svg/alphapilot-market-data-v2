# AlphaPilot V6 CPU-Quantile — Preregistration

Status: PRE-OOS / NOT YET SOURCE-FROZEN.
Branch: `research-ai-market-reasoning-v6-cpu-quantile-20260916`

## 1. Resource-constrained objective
This version is the CPU-feasible baseline for the independent AI Market Reasoning + Path Forecast Engine. Deep/self-supervised representation learning is explicitly deferred to a future, separately preregistered successor version. It may not be introduced to rescue or reinterpret V6 CPU-Quantile after OOS is viewed.

The model must remain independent from R10/R7/R6 candidate pools, regimes, scores, entries, exits, and holding logic.

## 2. Human/AI boundary
Humans define the causal information universe, forecast question, model family, validation procedure, and resource limits before OOS. The model learns nonlinear relationships and interactions without hand-written buy weights, fixed Top-N quotas, or hard-coded market/industry trading rules.

Observable transforms may exist only to make causal time-series/context data numerically usable by the CPU model. They are frozen before OOS and are not tuned from OOS performance.

## 3. Core model family — locked choice before OOS
The only formal V6 CPU-Quantile forecasting family is:

- `sklearn.ensemble.HistGradientBoostingRegressor`
- loss: `quantile`
- fixed quantiles: 0.10, 0.25, 0.50, 0.75, 0.90
- horizon-conditioned model: horizon is an input, so the same five models represent terminal return distributions for horizons 1 through 120 sessions
- deterministic non-crossing projection: sort predicted quantiles at each asset/date/horizon
- no Normalizing Flow, Diffusion, neural network, Transformer, alternate tree family, or architecture tournament in this version

Fixed model parameters before OOS:
- learning_rate = 0.05
- max_iter = 160
- max_leaf_nodes = 31
- min_samples_leaf = 120
- l2_regularization = 4.0
- RNG seed base = 926616

Changing the model family or these parameters after formal OOS is viewed requires a successor version and complete preregistration.

## 4. Path forecast definition
The core prediction object is the conditional terminal-return distribution as a function of horizon:

`Q_q(R_{T->T+h} | information available by T close), h=1..120, q in {0.10,0.25,0.50,0.75,0.90}`.

The system does NOT train separate models for +5%, +10%, +20%, +30% events. Those thresholds, if reported later, are diagnostic queries/calibrated events derived from the frozen path-distribution model and cannot be used to choose the model architecture.

Formal reporting checkpoints 5/10/20/40/60/120 sessions are audit slices only. They are not forced holding periods and cannot be selected post hoc because one horizon looks best.

## 5. Training construction
To avoid 600 separate models while representing all horizons:
- each horizon 1..120 is represented in the long-format training sample;
- each horizon receives the same maximum sampling budget;
- total long-format fit rows are capped at 480,000 for CPU feasibility;
- sampling is deterministic from the preregistered RNG seed;
- horizon conditioning uses only `h/120` and `sqrt(h/120)`; these encode forecast distance, not market rules.

The 120-session purge is mandatory so no forward label overlaps the OOS decision block.

## 6. Stage 0 — data/PIT gate remains mandatory
No formal model claim is allowed until the causal context layer passes its audit. Historical OOS may only use information available by that historical date. Current semantic labels or news without reconstructible historical timestamps remain live/shadow only.

Successful staged data units must be checkpointed and reused; failures/unknowns cannot be silently converted into zero or valid no-data observations.

## 7. Stage A / Stage B / live separation
- Stage A formal development OOS: 2021–2024, one locked execution.
- Stage B sealed confirmation: 2025. It remains unopened unless preregistered Stage-A gates pass. Opening 2025 consumes it for this locked version.
- Stage C: 2026 onward is live/forward evidence and cannot tune historical claims.

The program defaults to Stage A only. `V6_OPEN_2025=1` is forbidden until the lock/audit records that Stage A passed its gates.

## 8. Stage 2 distribution diagnostics
Before any BUY/selection layer, publish for every audit year and audit horizon:
- pinball loss for each frozen quantile;
- empirical CDF coverage for each quantile;
- q10–q90 interval coverage;
- median absolute error of q50;
- sample count;
- train cutoff and fit-row count.

A visually attractive forecast is not sufficient for validation.

## 9. Stage 3 probability calibration
Raw quantiles are not user-facing event probabilities. Any event probability derived from the forecast distribution must be calibrated using only past-only calibration data according to `AI_MARKET_REASONING_FORECAST_V6_VALIDATION_PROTOCOL.md`:
- Reliability Diagram with fixed 0.1 probability buckets;
- Brier score, ECE, calibration slope/intercept and sample support;
- Platt default; Isotonic only under the predeclared sample-support rule;
- no calibrator selection using the OOS year being judged.

Until Stage 3 is implemented and passed, the live decision is explicitly `NONE_UNTIL_STAGE3_CALIBRATION`.

## 10. Multiplicity / researcher degrees of freedom
Internal tree splits and latent nonlinear interactions are not treated as separate hypotheses. Researcher choices are counted: architecture variants, model-family changes, target/event definitions, horizon/report choices, subgroup analyses, calibrator variants, thresholds, feature-schema revisions, and successor versions.

All formal claims must use the frozen hypothesis registry. Failed/abandoned research versions remain in the audit history.

## 11. Explanation discipline
No narrative explanation may increase forecast probability. Historical-neighbor or perturbation explanations, if added later, are diagnostic only and inherit the model's validation limitations. They are not independent confirmation evidence.

## 12. Preregistration lock
Formal OOS is prohibited until a SOURCE FREEZE commit and subsequent machine-readable LOCK manifest hash all research code, this preregistration, validation protocol, hypothesis registry, runtime specification, and data manifests. The verifier must fail closed on mismatch.

No bad OOS result may be silently repaired under the same scientific lock. Engineering-only fixes can retain a lock only when an equivalence audit proves unchanged research semantics/predictions; otherwise create a successor version.

## 13. Success philosophy
Precision and calibration dominate signal quantity. NONE is valid. The first objective is a resource-feasible, auditable baseline that survives preregistration and OOS. Deep representation learning is only reconsidered after this baseline demonstrates enough value to justify dedicated GPU/cloud research under a new preregistered version.
