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

## 6. Stage 0 — layered data/PIT gate
No formal model claim is allowed until every required Stage-0 layer passes its own audit, is frozen with a machine-readable manifest, and the final cross-layer assembly passes Stage 0F. The controlling contract is `research/V6_STAGE0_LAYER_CONTRACT.json`.

Required layers are:
- Stage 0A Market
- Stage 0B Industry
- Stage 0C Company/Fundamental
- Stage 0D Macro
- Stage 0E Event/Time
- Stage 0F Final PIT Assembly

The layers are an engineering/data lineage decomposition only. They must NOT become separate predictive models, hand-written layer scores, or weighted voting systems. All frozen layers are assembled once into the single preregistered V6 CPU-Quantile model.

Every frozen layer manifest must record at least: source lineage, dataset-level time semantics, start/end dates, row counts, missingness, release/as-of rules, file SHA256 hashes, and the exact generation commit. Successful frozen layer artifacts are reusable only while their manifest/hash remains unchanged.

### 6.1 Fallback-source equivalence gate
A fallback or mirror transport is not accepted merely because it downloads successfully. Before a fallback-backed series may be marked `FROZEN_PASS`, it must pass the preregistered equivalence audit in `V6_STAGE0_LAYER_CONTRACT.json`:
- semantic identity of the economic series, units, frequency, and source lineage;
- date alignment over a preregistered overlap sample;
- numeric agreement within the frozen absolute tolerance;
- publication/availability lag audit so a mirror cannot introduce future information;
- immutable evidence and hashes.

If the primary host is inaccessible from GitHub Actions, the fallback remains `PROVISIONAL` until the equivalence audit is performed in an independent reachable environment and its evidence is committed before SOURCE FREEZE. Transport failure is an engineering issue; it does not waive the data-quality gate.

For the current FED/H15 candidate mirror, successful connectivity alone is NOT a Macro-layer PASS.

### 6.2 Stage 0F cross-layer alignment gate
Independent layer PASS results are insufficient. Stage 0F must audit the assembled decision-date/ticker panel for:
- cross-layer date range and eligible-decision-date coverage;
- unmatched dates and explicit missingness by layer/dataset;
- duplicate keys and timezone/session-date normalization;
- as-of joins for periodic fundamentals/revenue/macro/events with `available_at <= decision_time`;
- stale-age/availability-lag distributions for non-daily data;
- zero future-join violations;
- zero silent zero-fill violations;
- source/manifest hashes used in the exact assembly.

Daily required layers must meet the frozen eligible-date coverage gate; periodic/event layers are judged by causal as-of availability rather than pretending they are daily observations.

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
Internal tree splits and latent nonlinear interactions are not treated as separate hypotheses. Researcher choices are counted: architecture variants, model-family changes, target/event definitions, horizon/report choices, subgroup analyses, calibrator variants, thresholds, feature-schema revisions, data-layer schema revisions, and successor versions.

All formal claims must use the frozen hypothesis registry. Failed/abandoned research versions remain in the audit history.

## 11. Explanation and layer-ablation discipline
No narrative explanation may increase forecast probability. Historical-neighbor or perturbation explanations, if added later, are diagnostic only and inherit the model's validation limitations. They are not independent confirmation evidence.

Layer Ablation Audit is also diagnostic only. After the scientific lock or after viewing OOS, a result such as “removing Macro improves performance” may be reported as an observation but may NOT be used to remove/reweight that layer and rerun under the same V6 lock. Any such research-driven schema change requires a separately preregistered successor version.

## 12. Preregistration lock
Formal OOS is prohibited until a SOURCE FREEZE commit and subsequent machine-readable LOCK manifest hash all research code, this preregistration, validation protocol, hypothesis registry, Stage-0 layer contract, layer manifests, runtime specification, and data manifests. The verifier must fail closed on mismatch.

No bad OOS result may be silently repaired under the same scientific lock. Engineering-only fixes can retain a lock only when an equivalence audit proves unchanged research semantics/predictions; otherwise create a successor version.

## 13. Success philosophy
Precision and calibration dominate signal quantity. NONE is valid. The first objective is a resource-feasible, auditable baseline that survives preregistration and OOS. Deep representation learning is only reconsidered after this baseline demonstrates enough value to justify dedicated GPU/cloud research under a new preregistered version.
