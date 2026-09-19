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
1. Probability of reaching +10%, +20%, +30%, and +50% within multiple research windows (20/40/60/120 sessions), regardless of whether an interim drawdown occurred first. Moves below +10% are not primary profit opportunities for V6.2.
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
- +10% before -5% and +20% before -10% as secondary path-quality diagnostics; +5% before -3% is removed from the formal target set because sub-10% moves are not the strategy objective;
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


## Minimum opportunity magnitude
V6.2 does not target 3%-5% trades. A stock must be assessed on the possibility of at least a +10% future move to qualify as a meaningful opportunity. +3%/+5% movements may appear inside path diagnostics but must not be treated as a profitable-opportunity success, candidate admission target, or headline forecast.


## Timing contract: launch and profit are separate
V6.2 must not confuse launch timing with profit-target timing.

Launch:
- expected start window is 1-30 trading sessions after entry;
- earlier is better, but any evidence-supported launch within 30 sessions can qualify.

Profit:
- minimum meaningful opportunity remains +10%;
- +20%, +30%, and +50% extensions are explicitly modeled;
- profit-target timing must be predicted independently from launch timing rather than forced into a fixed 20-30-session deadline.

Continuation:
- once an upward move has launched, the opportunity can remain active beyond session 30 while price progress and thesis/catalyst remain intact;
- if the thesis breaks, price rolls over, or the move stalls materially, the continuation state can become REASSESS or END.

## Launch-window preference
V6.2 launch timing is a forecast window of 1-30 trading sessions after entry.

Core rule:
- a valid opportunity should have a credible expectation that its upward move begins within 1-30 trading sessions;
- next-session or 1-3-session launch is ideal and receives higher entry-quality assessment, but it is not a hard requirement;
- a launch on session 10, 20, or 30 can still be valid if the evidence and profit thesis remain intact;
- failure to rise immediately must not invalidate a stock before the 30-session launch window has expired unless the thesis itself breaks.

Launch and profit magnitude are distinct:
- launch timing asks when the upward move begins;
- profit objective asks whether the move can ultimately deliver at least +10%;
- the system must not force +10% to be completed by the same deadline used for launch prediction.

Historical validation must separately test:
1. launch within 1 session;
2. launch within 3 sessions;
3. launch within 5 sessions;
4. launch within 10 sessions;
5. launch within 20 sessions;
6. launch within 30 sessions;
7. MAE before launch;
8. eventual +10/+20/+30/+50 upside after launch;
9. whether HIGH_CONVICTION launches earlier and/or produces larger post-launch upside than ordinary CANDIDATE.



## AI-first selection contract
V6.2 is an AI opportunity-selection system, not a deterministic screening-rule engine.

The AI must make the final stock-level decision from the full admitted evidence bundle and numerical forecasts. Numerical values such as +10/+20/+30/+50 probabilities, launch timing, MAE, valuation, flow, trend, revenue, events, sector context, and catalyst evidence are inputs to reasoning, not standalone admission rules.

Forbidden implementations include:
- fixed factor filters that mechanically decide CANDIDATE/HIGH_CONVICTION;
- rules such as p_hit10 > X, MA20 > MA60, foreign flow > X, launch probability > X, or any fixed conjunction used as the final selection gate;
- ranking the market and admitting a fixed TopK;
- using the historical validation targets themselves as hard-coded screening thresholds.

Required implementation:
- every stock is independently assessed by the AI;
- the AI identifies the relevant causal opportunity story for that stock;
- the AI may weigh different evidence families differently for different stocks;
- the AI can reject a stock despite strong numerical scores when the evidence story is weak or contradictory;
- the AI can keep a stock as WATCH when timing/entry is not attractive even if long-run upside exists;
- final admission is the AI judgment, while deterministic code only enforces data integrity, schema validity, no-hallucination constraints, and numerical immutability.

The historical validation must evaluate whether AI decisions add value over the raw numerical forecasts. If a deterministic threshold performs the same function as the AI decision layer, the architecture has failed the intended design.


## Failure-exit validation
Every actionable historical AI output must include an AI Failure Exit Price defined before entry.

Historical replay must test:
1. whether the failure exit price was known before entry;
2. frequency of exit-price triggers;
3. realized loss at actual executable exit, including adverse gaps/slippage;
4. whether a triggered exit prevented materially larger subsequent loss;
5. false-exit rate: cases where the exit triggered but the stock later became a large winner;
6. whether AI-raised exit prices protected gains after a successful launch;
7. no downward revision of the failure exit price after entry.

The failure exit price is not the same as MAE prediction. MAE describes expected path risk; the failure exit price is the action level where the AI concedes that its thesis no longer deserves capital.
