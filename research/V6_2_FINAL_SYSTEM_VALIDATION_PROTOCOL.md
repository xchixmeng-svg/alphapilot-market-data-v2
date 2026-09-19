# V6.2 Final Daily Output and Historical Validation Protocol

Status: IMPLEMENTATION_CONTRACT
Date: 2026-09-19

## Goal
Build the complete V6.2 AI full-market opportunity detector first, then validate the historical accuracy of the **final AI outputs**, not only the numerical submodels.

## Daily system output
Every eligible stock independently receives one of:
- REJECT
- WATCH
- CANDIDATE
- HIGH_CONVICTION

Candidate count is emergent and may be 0, 1, 2, 15, or more. No TopK admission quota is allowed.

For WATCH/CANDIDATE/HIGH_CONVICTION the final output must expose:
- why an opportunity may be forming now;
- best entry status and price zone;
- all 11 calibrated barrier probabilities;
- expected upside range and bull case;
- expected MAE range;
- likely start window and main opportunity window;
- bull thesis;
- bear/counter evidence;
- invalidation;
- evidence quality.

The LLM may reason over evidence and decide the stock state, but it may not invent or alter calibrated numerical probabilities.

## Historical replay validation

### Primary objective: real profit opportunity
The primary success question is NOT whether price reached an upside barrier before a small downside barrier. A stock that first falls -5% or -10% and later rallies +30%, +50%, or more can still be a genuine profitable opportunity.

The historical replay must therefore evaluate the final AI answers primarily by eventual opportunity magnitude and timing:
1. Probability of reaching +5%, +10%, +20%, +30%, and +50% within multiple research windows (20/40/60/120 sessions), regardless of whether a smaller drawdown occurred first.
2. Realized maximum favorable excursion (MFE) over 20/40/60/120 sessions.
3. Time to first reach each upside level and time to realized peak.
4. Predicted base/bull upside ranges vs realized MFE.
5. Candidate / HIGH_CONVICTION vs full eligible market on large-winner rates, MFE distribution, and expected upside.
6. WATCH missed-winner diagnostics, especially stocks that later produced +20%, +30%, or +50% moves.
7. HIGH_CONVICTION vs CANDIDATE separation on realized upside magnitude and large-winner frequency.
8. No-candidate-day diagnostics: whether abstention days truly had weaker future opportunity distributions.
9. Candidate count distribution; zero candidate days are valid.
10. Results by year, evidence family, market regime, and opportunity hypothesis type.

### Secondary objective: path quality and risk
Path metrics remain important, but they are risk/comfort diagnostics rather than the main definition of success:
- +5% before -3%, +10% before -5%, +20% before -10%;
- realized MAE before first upside hit and before realized peak;
- drawdown depth before a later large rally;
- recovery time from interim drawdown;
- predicted MAE range vs realized MAE.

A case such as -8% first and then +50% must be classified as a large profitable winner with poor/intermediate path quality, NOT as a failed opportunity.

### Calibration
Probability calibration must be tested separately for:
- unconditional upside probabilities (reach +X within Y sessions regardless of interim drawdown);
- barrier/path probabilities (reach +X before -Y).
Predictions near X% should realize near X% within their own definition.

## Chronology
- Build/dev/replay without opening sealed 2025 confirmation.
- 2024 can be used as development validation because prior V6/V6.1 results have already been seen.
- 2025 remains sealed until the final V6.2 contract, prompt/model version, candidate decision policy, and numerical engine are frozen.
- 2026 remains live-only.
- R10 remains isolated and untouched.

## Anti-cheating
- No future evidence.
- No rank-only admission.
- No fixed daily candidate count.
- No tuning a probability threshold on the same historical slice and then describing it as untouched validation.
- Missing evidence must be represented explicitly; the AI may not hallucinate unavailable EPS revisions, industry pricing, or other evidence.
- Historical LLM outputs must be persisted and replayed; later model upgrades create a new version rather than silently rewriting history.
