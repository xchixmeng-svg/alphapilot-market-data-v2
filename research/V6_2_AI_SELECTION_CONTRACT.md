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
