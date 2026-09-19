# V6.2 Architecture Alignment Decisions

Status: HARD ARCHITECTURE CONTRACT
Date: 2026-09-19

## Decision 1 — Primary profit labels are unconditional

The previous 11-task barrier matrix (+X before -Y) is retained only as a secondary path/risk diagnostic.

It is NOT the primary V6.2 profit model and its results must not be used to judge the new primary objective.

The primary numerical engine must be retrained on unconditional future outcomes:
- hit +10% within 20/30/40/60/120 sessions;
- hit +20% within 20/30/40/60/120 sessions;
- hit +30% within 20/30/40/60/120 sessions;
- hit +50% within 20/30/40/60/120 sessions.

Interim drawdown does not turn a later large winner into a failure.

The old u20_d10_h120 weakness is evidence only about that barrier-first task. It is not evidence that unconditional +20%/+30%/+50% prediction is weak.

Barrier outputs remain available for:
- path quality;
- MAE context;
- false-exit analysis;
- execution-risk diagnostics.

## Decision 2 — Launch is an auxiliary event within the Numerical Opportunity Engine

Do NOT create a third independent stock-selection model.

Architecturally V6.2 remains:
1. Numerical Opportunity Engine
2. AI Reasoning / Decision Engine

The numerical engine may contain multiple calibrated heads/estimators internally, but they do not independently admit candidates.

### Realized launch label

Launch must be tied to the actual profitable move rather than an arbitrary green-day rule.

For a stock-date that eventually reaches at least +10%:
- find the first session that reaches +10% from the reference entry price;
- inspect the path from entry through that first +10% hit;
- identify the final pre-run trough / lowest structural point from which the successful upward leg proceeds to the first +10% hit;
- realized launch is the first trading session after that trough where the upward leg begins.

If the stock never reaches the minimum +10% opportunity, launch-to-success is censored rather than fabricated.

The exact deterministic label implementation must be audited for edge cases (entry itself is the trough, equal lows, price resets, gaps, corporate actions) before model training.

The engine predicts:
- P(launch by 1/3/5/10/20/30 sessions);
- expected launch session / interval;
- pre-launch MAE;
while profit probabilities remain separate unconditional targets.

These predictions are evidence for the AI, never candidate admission rules.

## Decision 3 — Failure Exit requires thesis-dependent evidence sufficiency

The AI Failure Exit Price may only rely on evidence that actually exists point-in-time.

Current evidence status:
- price/volume/structure: AVAILABLE;
- valuation/revenue/frozen 0F context: AVAILABLE;
- institutional flow: AVAILABLE in admitted V1 supplemental evidence;
- MOPS historical material-information titles: source AVAILABLE in frozen 0E; must be joined through valid V2;
- EPS revisions / analyst consensus history: UNAVAILABLE;
- industry pricing / inventory / supply-demand history: UNAVAILABLE;
- broad historical news semantic corpus: not yet admitted.

Therefore:
- a price-structure or flow thesis may produce a supported failure exit now;
- a disclosed-event thesis may do so only after valid V2 event text is present;
- EPS-revision, industry-pricing, inventory/supply-demand, or unsupported-news claims must be marked UNAVAILABLE and cannot justify a candidate or failure-exit reason;
- missing evidence must never be inferred from unrelated proxies.

Evidence sufficiency is thesis-dependent, not a universal factor checklist. The AI may use different evidence families for different stocks.

## Scientific alignment gate

Before opening sealed 2025:
1. retrain/evaluate unconditional primary profit heads;
2. audit and freeze realized-launch labeling;
3. rebuild/validate V2 event-text evidence against valid V1;
4. define thesis-dependent evidence availability in the reasoning contract;
5. validate AI final decisions against raw numerical forecasts;
6. keep barrier matrix strictly secondary.

If the AI final decisions merely reproduce a fixed numerical threshold, V6.2 fails its intended architecture.
