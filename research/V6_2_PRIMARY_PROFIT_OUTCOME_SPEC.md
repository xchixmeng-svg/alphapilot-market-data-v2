# V6.2 Primary Profit Outcome Specification

Status: IMPLEMENTATION_CONTRACT
Date: 2026-09-19

## Core principle
V6.2 is designed to find stocks with a genuine profitable future move, not merely stocks whose path is smooth.

A stock may experience an interim drawdown and still be a successful opportunity if it later produces a sufficiently large rally. Therefore:

- profit opportunity magnitude is the primary target;
- timing is a prediction target, not a forced exit;
- drawdown/path quality is a separate risk dimension;
- barrier-first labels are secondary diagnostics only.

## Primary future outcomes per stock-date

For each decision date, compute without using future information in the decision itself:

### Unconditional target-hit outcomes
Whether the stock reaches:
- +10%
- +20%
- +30%
- +50%

Sub-10% upside is not a primary success target.

within:
- 20 sessions
- 40 sessions
- 60 sessions
- 120 sessions

These labels remain successful even if price first falls below -3%, -5%, or -10%.

### Magnitude outcomes
For 20/40/60/120-session windows record:
- maximum favorable excursion (MFE);
- terminal return;
- session of maximum favorable excursion;
- time to first +10/+20/+30/+50 hit.

### Path/risk outcomes
Record separately:
- maximum adverse excursion (MAE);
- MAE before first target hit;
- MAE before realized peak;
- recovery time after interim drawdown;
- secondary barrier diagnostics (+10 before -5, +20 before -10). The +5 before -3 task is retired from the formal V6.2 objective set.

## Interpretation examples

- First -8%, later +50%: LARGE_WINNER, path quality poor/intermediate.
- First -2%, later +20%: WINNER, path quality good.
- First +10%, then collapses and never develops further: small/short-lived opportunity; not equivalent to a +30/+50 winner.
- Never develops meaningful upside: failed/low-opportunity case.

## AI output implications
The AI should estimate:
- probability of reaching +10/+20/+30/+50 within relevant windows;
- expected/base/bull MFE or upside range;
- likely launch window and main move window;
- expected MAE separately;
- path quality / drawdown risk separately.

No fixed holding-period sell rule is implied by these horizons. They are research and forecast windows only.

## Historical validation priority
Primary:
1. large-winner frequency;
2. MFE distribution;
3. upside-probability calibration;
4. timing calibration;
5. Candidate/High Conviction separation;
6. missed large winners among WATCH/REJECT.

Secondary:
1. barrier-first probabilities;
2. MAE;
3. recovery difficulty;
4. clean-path vs deep-drawdown winner mix.

2025 remains sealed until the final architecture and prediction contract are frozen.


## Hard opportunity floor
A forecast whose expected opportunity is below +10% is not a V6.2 profit opportunity. It may be WATCH/REJECT for context, but it must not become CANDIDATE or HIGH_CONVICTION merely because a +3% or +5% move appears likely.
