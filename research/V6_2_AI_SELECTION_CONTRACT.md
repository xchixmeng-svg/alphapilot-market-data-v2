# V6.2 AI Selection Contract

Status: HARD ARCHITECTURE CONTRACT
Date: 2026-09-19

## Principle
V6.2 must select opportunities by AI reasoning, not by fixed screening conditions.

The system may compute structured numerical forecasts and evidence features, but those values do not mechanically decide admission.

## What deterministic code may do
- retrieve and timestamp evidence;
- enforce point-in-time causality;
- calculate calibrated probabilities and path statistics;
- validate schema;
- prevent numerical mutation;
- prevent hallucinated evidence;
- record hashes and audit trails.

## What deterministic code may NOT do
It may not decide CANDIDATE/HIGH_CONVICTION using fixed threshold logic such as:
- probability > X;
- return > X;
- MA crossover;
- valuation < X;
- institutional flow > X;
- launch probability > X;
- any fixed AND/OR combination;
- TopK or rank quota.

## Final decision owner
The AI reasoning layer owns:
- why this stock may rise now;
- which evidence families actually matter for this case;
- whether evidence conflicts;
- whether the current price/entry is attractive;
- whether the opportunity is REJECT, WATCH, CANDIDATE, or HIGH_CONVICTION;
- whether a launched winner should CONTINUE, REASSESS, or END.

The same evidence family does not have to matter equally for every stock.

## Numerical forecasts
Forecasts such as p_hit10/20/30/50, launch timing, MAE, MFE, and path probabilities are evidence for the AI. They are not automatic admission gates.

## Scientific test
Historical replay must compare:
1. raw numerical forecast quality;
2. AI final decisions;
3. whether AI decisions improve opportunity concentration, large-winner capture, entry quality, and missed-winner control beyond raw numbers alone.

If AI decisions merely reproduce a fixed numerical threshold, V6.2 has failed its intended architecture.


## AI Failure Exit Price (mandatory)
Every CANDIDATE and HIGH_CONVICTION decision must include a pre-entry AI Failure Exit Price.

This price answers:
"If the AI thesis is wrong after entry, at what price must the position be exited rather than defended?"

Rules:
- the exit price is stock-specific and AI-reasoned, not a fixed percentage stop;
- it must be supported by the stock's own price structure, opportunity thesis, catalyst, valuation/expectation context, and plausible failure path;
- it is mandatory before entry;
- after entry, the failure exit price may stay unchanged or be raised, but must not be lowered to avoid admitting a failed thesis;
- a non-price thesis invalidation may also force exit earlier;
- touching/breaching the trigger does not imply an impossible fill at the exact trigger price: historical/live execution must use the next realistically tradable price with adverse gap/slippage handling.

Deterministic code may enforce that the field exists and that it is not lowered after entry, but it may not calculate the exit price from a fixed percentage formula. The AI owns the exit-price judgment.
