# AlphaPilot V6 — Validation, Calibration, Multiplicity, and Preregistration Lock Protocol

Status: PRE-OOS. This protocol is frozen before any formal V6 OOS result is viewed.

## 1. Purpose
This document governs statistical validation of the independent AI Market Reasoning + Path Forecast Engine. It is intentionally stricter than a single-strategy backtest because V6 has large research degrees of freedom across market, industry, company, macro, event, seasonal, target, horizon, and setup dimensions.

No result may be called validated merely because it is visually strong, economically attractive, or selected from a large search.

## 2. OOS staging and preservation
V6 uses sequentially sealed evidence.

### Stage A — development walk-forward OOS
- Years: 2021–2024.
- Each year is predicted strictly from information available before that test year.
- Purge/embargo must be at least the longest forward label horizon (120 sessions unless the preregistration is superseded before lock).
- Stage A is executed once for a locked research version.

### Stage B — sealed confirmation OOS
- Year: 2025.
- 2025 MUST NOT be evaluated, inspected, or used for model/calibration/hypothesis choice until Stage A gates defined before execution have passed.
- Once 2025 is opened for a locked version, it is consumed for that version. Any subsequent architecture/feature/label/calibration/hypothesis change requires a new version, branch, lock, and preregistration. The same 2025 result cannot be represented as untouched OOS again.

### Stage C — live/forward evidence
- Year: 2026 onward.
- 2026 observations are live/forward evidence only and are never tuning data for historical OOS claims.
- Live forecasts are append-only evidence. Failed live forecasts remain in the record.

## 3. Probability calibration

### 3.1 Past-only fitting
For every event probability shown to the user:
1. fit the base model only on data whose forward labels are fully known before the calibration period;
2. generate calibration predictions on a later, non-training, past-only calibration block;
3. fit the probability calibrator only on that calibration block;
4. apply the frozen calibrator to the next OOS block/year;
5. never fit, refit, select, or tune a calibrator using the OOS block it is judged on.

No raw model confidence may be displayed as an empirical probability.

### 3.2 Reliability diagram
For every formal target/horizon event, publish a reliability table and diagram using fixed equal-width buckets:
[0,0.1), [0.1,0.2), ..., [0.9,1.0].

Each bucket must report:
- number of forecasts;
- mean predicted probability;
- realized OOS event rate;
- Wilson 95% confidence interval for the realized rate.

The 45-degree line is perfect calibration.

Also report:
- Brier score;
- log loss where numerically defined;
- Expected Calibration Error (ECE);
- calibration intercept/slope where estimable;
- base event rate;
- sample count and positive/negative event counts.

### 3.3 Platt / Isotonic rule
The official calibrator is selected without looking at the target OOS period:
- Platt scaling is the default official calibrator.
- Isotonic regression may replace Platt only when the immediately preceding past-only calibration set contains at least 2,000 usable observations, at least 200 positive events, and at least 200 negative events for that exact target/horizon event.
- The choice is therefore determined by predeclared sample support, not by whichever method looks better on the OOS year.
- Both methods may be reported as diagnostics, but only the preregistered official method supplies user-facing probabilities.

If support is insufficient, the system must lower confidence, mark the probability as unverified, or abstain. It may not extrapolate a high-confidence bucket.

### 3.4 90% production probability
A user-facing `>=90%` probability requires:
- calibrated probability from the official past-only calibrator;
- sufficient sample support in the relevant probability region;
- realized reliability compatible with the stated probability within uncertainty;
- no failure of the multiplicity/confirmation requirements below.

Otherwise output a lower probability or NONE/WAIT.

## 4. Multiple-comparison control

### 4.1 Frozen hypothesis registry
Before OOS execution, every formal hypothesis family must be enumerated in a machine-readable registry with stable IDs.

A distinct test includes any material combination of:
- industry/peer family;
- setup or opportunity state;
- feature/factor family;
- target event;
- forecast horizon;
- model/architecture variant if compared for selection.

The total number `m` in each family is frozen before viewing OOS results. Post-hoc splitting of families to reduce `m` is prohibited.

### 4.2 Exploratory discoveries
For exploratory research claims, control Benjamini–Hochberg False Discovery Rate at q <= 0.05 within the preregistered family.

Every reported discovery must include:
- raw p-value;
- BH-adjusted q-value;
- family ID;
- family size m;
- effect size;
- sample size.

An FDR discovery is not by itself sufficient for a production BUY claim.

### 4.3 Confirmatory claims
Any claim that an edge is formally validated or eligible to support production BUY must survive family-wise error control:
- Bonferroni adjusted threshold alpha = 0.05 / m for its preregistered confirmatory family.

No unadjusted p-value may be used to label a result validated when more than one hypothesis was searched.

### 4.4 Dependence-aware inference
Stock-day rows are not treated as independent observations.
Formal significance must be calculated on date/episode-level evidence using a preregistered dependence-aware procedure (e.g. moving/stationary block bootstrap or equivalent clustered/HAC inference appropriate to the forecast horizon). The exact implementation and block/lag choice must be frozen in the hypothesis registry before Stage A.

## 5. Research-degree-of-freedom audit
Every locked version must publish:
- number of candidate feature families;
- number of setup/state definitions;
- number of industry/peer partitions;
- number of model variants;
- number of target levels;
- number of horizons;
- total formal hypotheses tested;
- total exploratory analyses viewed but not used for formal claims.

Hidden experiments are prohibited. Failed tests remain counted.

## 6. Preregistration lock mechanism

### 6.1 Two-commit lock
Formal OOS may run only after:
1. a SOURCE FREEZE commit contains all research logic, data manifests, feature schema, label definitions, calibration rules, hypothesis registry, package versions, and success gates;
2. a subsequent LOCK commit adds a machine-readable lock manifest containing:
   - SOURCE FREEZE commit SHA;
   - SHA256 for every locked file;
   - lock ID/version;
   - creation timestamp;
   - OOS stages authorized by the lock.

The lock manifest does not hash itself.

### 6.2 Mandatory verification
Before any OOS step, a verifier must:
- recompute SHA256 for all locked files;
- verify the expected SOURCE FREEZE commit is in the execution ancestry;
- verify required data manifests/checksums;
- verify the hypothesis registry is unchanged;
- verify package/runtime versions;
- fail closed on any mismatch.

Every OOS artifact must record:
- execution commit SHA;
- lock ID;
- SOURCE FREEZE commit SHA;
- lock manifest SHA256;
- data manifest hashes.

### 6.3 After OOS is viewed
After any locked OOS result is viewed:
- research logic may not be changed and rerun under the same lock;
- feature, label, target, horizon, model, calibration, selection, threshold, Top-N, weighting, hypothesis-family, and success-gate changes require a new version/branch and a complete new preregistration/lock;
- an engineering-only fix may preserve the same scientific claim only if it does not change any locked input/output semantics and an equivalence audit demonstrates identical research inputs and predictions. Otherwise it is a new research version.

A bad OOS result is recorded as a failed research version, not silently patched.

## 7. Model-selection discipline
No architecture, factor, threshold, Top-N, or weight is selected because it looks best on Stage A/Stage B OOS.

If a version is far from success, a principled architecture change is allowed only in a successor preregistered version. Broad hyperparameter sweeps that choose the best OOS outcome are prohibited.

## 8. Existing principles retained
- Engineering failure and research failure are separate.
- API/data transport failures do not count as model failure.
- Point-in-time causality remains mandatory.
- R10/R7/R6 candidate pools, regime, scores, entries, exits, and holding logic remain prohibited from V6.
- The formal R10 branch remains untouched.
- NONE is a valid and preferred output when calibrated evidence is insufficient.
