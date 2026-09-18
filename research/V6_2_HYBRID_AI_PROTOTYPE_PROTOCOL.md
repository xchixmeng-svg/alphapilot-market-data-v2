# V6.2 Hybrid AI Prototype Protocol (NOT LOCKED)

Status: PROTOTYPE_ONLY
Date: 2026-09-18

This document is deliberately pre-preregistration. Its purpose is to test whether the proposed Hybrid AI architecture is technically coherent before any scientific lock.

## A. Architecture under test

1. Deterministic Evidence Bundle
2. Numerical Opportunity Engine
   - quantile/path model family (later)
   - competing-risk/barrier event model family (later)
3. LLM Reasoning + Decision
   - reads fixed evidence bundle + fixed numerical outputs
   - never edits numerical probabilities
   - outputs structured reasoning and REJECT/WATCH/CANDIDATE/HIGH_CONVICTION

No daily Top-K admission is permitted. Candidate count may be zero.

## B. Fixed prototype barrier matrix

Barrier labels are anchored to decision-day close only for this prototype. This does NOT yet define executable T+1 entry mechanics.

- UP +5% / DOWN -3% at horizons 5, 10, 20 sessions
- UP +10% / DOWN -5% at horizons 10, 20, 40, 60 sessions
- UP +20% / DOWN -10% at horizons 20, 40, 60, 120 sessions

For every decision row and task, path state is one of:
- UP_FIRST
- DOWN_FIRST
- NEITHER
- AMBIGUOUS_SAME_BAR
- CENSORED_RESET_OR_END

Daily OHLC cannot identify ordering when both barriers are crossed inside the same bar. Such rows must never be silently assigned to UP_FIRST or DOWN_FIRST.

Corporate-action safety inherits V6.1 segmentation. No label may cross price_segment_id.

## C. Evidence retrieval contract

Retrieval is deterministic. The LLM is NOT allowed to decide what raw evidence is fetched.

Every stock/day bundle always contains these 12 families in this exact order:

1. price_volume_structure
2. institutional_flow
3. revenue_earnings
4. eps_revisions
5. valuation
6. industry_cycle
7. industry_pricing_supply_demand
8. peer_sector_condition
9. macro_rates_fx_commodities
10. corporate_events
11. news_disclosed_catalysts
12. seasonality_historical_context

Each family must have:
- status: AVAILABLE | PARTIAL | UNAVAILABLE
- as_of / freshness where applicable
- evidence fields
- explicit missing_reason when not fully available
- stable evidence IDs

The LLM may assess relevance after seeing all 12 families; it may not hide a family before inference.

## D. LLM output contract

The LLM must return structured JSON only:

- hypothesis_type
- hypothesis
- primary_evidence_ids[]
- secondary_evidence_ids[]
- counter_evidence_ids[]
- bull_thesis
- bear_thesis
- invalidation
- evidence_quality: STRONG | MODERATE | WEAK
- decision: REJECT | WATCH | CANDIDATE | HIGH_CONVICTION
- decision_reason

Forbidden:
- generating a new probability not present in numerical_outputs;
- modifying/rounding numerical probability into a different value;
- rank-only admission;
- requiring at least one candidate per day;
- uncalibrated confidence score 0-100.

## E. Reproducibility

Numeric layers require strict reproducibility.

LLM layer requires a frozen:
- evidence bundle schema/version
- family order
- prompt hash
- model/provider/version
- generation settings
- structured response schema

Formal historical evaluation must persist raw structured responses and replay them later rather than re-querying a newer LLM.

## F. Operational prototype gates before preregistration

Before scientific LOCK:
1. barrier label audit: zero cross-reset labels;
2. evidence coverage audit: quantify each family, do not hallucinate missing families;
3. bundle size audit: p50/p95 bytes and rough token estimate;
4. LLM batch invariance benchmark must be implemented before batched production inference;
5. full-market completion SLO must be defined from real provider benchmark;
6. 2025 remains sealed during prototype/development.

This prototype does NOT authorize opening 2025.
