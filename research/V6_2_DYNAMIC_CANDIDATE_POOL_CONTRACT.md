# V6.2 Dynamic Candidate Pool — Core Contract

Status: DESIGN_FROZEN_FOR_IMPLEMENTATION
Date: 2026-09-18
Parent: V6.1 corporate-action-safe repair branch
Purpose: correct the product/evaluation direction. V6.2 is a market-wide candidate discovery engine, not a forced daily ranking engine.

## 1. User-facing behavior

Every trading day, scan the full eligible Taiwan-stock universe and output **all stocks that independently qualify as worth further evaluation**.

The number of candidates is NOT fixed:
- valid output may be 0 candidates;
- valid output may be 1 candidate;
- valid output may be several or many candidates.

A daily Top1 / Top3 / Top5 quota is explicitly prohibited as an admission rule.

Ranking is allowed **only after** the candidate pool is formed, to make the list easier to read. Ranking must never force a stock into the pool and must never force the pool to be non-empty.

The user, not the engine, decides which candidate to trade.

## 2. What each candidate must explain

For every emitted stock, V6.2 should eventually provide:
- stock code / name;
- why the AI thinks an opportunity may be forming now;
- which evidence families are active (not every family must be active);
- estimated probability of meaningful upside;
- estimated upside path / target zones;
- likely time window(s), without imposing a fixed forced holding period;
- expected adverse excursion / downside risk;
- invalidation evidence;
- confidence / evidence quality;
- relevant catalyst / valuation / sector / flow / price-structure context.

Possible evidence families include, when relevant:
- base / box / breakout formation;
- earnings and EPS revision;
- valuation / re-rating;
- industry cycle;
- freight / commodity / rate / macro changes;
- sector relative strength / drawdown;
- institutional / flow evidence;
- event timing;
- seasonality;
- price/volume structure.

The engine must be allowed to use different evidence combinations for different stocks.

## 3. No forced opportunity

The following are forbidden:
- "there must be at least one BUY every day";
- "always take the highest-scoring stock";
- "select exactly K names per day";
- using daily rank alone as the candidate criterion;
- treating absence of candidates as a model failure.

NONE / empty candidate set is valid.

## 4. Candidate qualification principle

Candidate admission must be **stock-local and evidence-based**, not quota-based.

Implementation requirement:
- estimate opportunity evidence for each stock independently;
- estimate uncertainty / confidence;
- emit a stock only when its own evidence clears the frozen candidate-evidence test;
- the candidate-evidence test must be frozen before formal evaluation;
- candidate count emerges from the market state.

The evidence test may use calibrated probability / expected-upside / downside-asymmetry / evidence-consistency signals, but it may not use a fixed daily count.

## 5. Multi-horizon path, no forced holding-period exit

Evaluate opportunity across multiple forward windows, including:
5, 10, 20, 40, 60, 120 sessions.

These are observation windows, not mandatory holding periods.

The engine should identify where the opportunity appears strongest rather than forcing all candidates into one horizon.

## 6. Corporate-action safety

Inherit V6.1 reset-safe segmentation:
- raw close-to-close absolute jump >20% is a fail-closed price-basis reset boundary;
- stock features and forward labels may not cross the boundary;
- reset day does not inherit prior-segment return/gap;
- 120 in-segment observations are required before stock eligibility.

This remains a data-safety rule, not an alpha rule.

## 7. Human-readable evaluation

The primary evaluation is candidate-pool quality, not TopK quality.

Required reports:
- number of candidate days;
- number of no-candidate days;
- candidates per day distribution;
- total candidates;
- candidate +5%, +10%, +20% hit rates across relevant windows;
- candidate MFE / MAE;
- time-to-hit distributions;
- false-positive rate;
- calibration by confidence bucket;
- results by evidence family / regime;
- comparison against the full eligible universe/base rate.

Optional ranking diagnostics may be shown only after candidate admission and must be labeled secondary.

## 8. Research chronology

- 2022–2024 may be used as development/repair-validation because results from prior V6/V6.1 research have already been viewed.
- 2025 remains sealed and is the first untouched confirmation year for a locked V6.2.
- 2026 remains live-only/sealed for research evaluation unless explicitly authorized later.
- R10 remains isolated and untouched.

## 9. Success meaning

V6.2 succeeds only if the **independently qualified candidate pool** shows useful lift versus the eligible-market base rate with acceptable calibration and downside behavior.

A model that merely ranks stocks every day but cannot abstain does not satisfy V6.2.
