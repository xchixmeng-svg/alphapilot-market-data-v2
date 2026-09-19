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
The historical replay must evaluate the final daily AI answers. Required tests:
1. Candidate pool actual +5%, +10%, +20% outcomes.
2. Candidate barrier hit rates: +5 before -3, +10 before -5, +20 before -10.
3. Probability calibration: predictions near X% must realize near X%.
4. Candidate vs full eligible market base-rate lift.
5. Predicted upside range vs realized MFE/terminal return.
6. Predicted MAE range vs realized MAE.
7. Predicted start/main timing windows vs realized first-hit times.
8. HIGH_CONVICTION vs CANDIDATE separation.
9. WATCH conversion / missed-opportunity diagnostics.
10. No-candidate-day diagnostics: whether abstention days truly had weaker opportunity distributions.
11. Candidate count distribution; zero candidate days are valid.
12. Results by year, evidence family, market regime, and opportunity hypothesis type.

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
