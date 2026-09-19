# V6.2 AI Failure Exit Contract

Status: HARD ARCHITECTURE CONTRACT
Date: 2026-09-19

## Purpose
If V6.2 makes a wrong prediction and a position is entered, the system must already know where it will concede the thesis has failed.

## Mandatory output
For every actionable CANDIDATE / HIGH_CONVICTION:
- planned entry zone;
- AI Failure Exit Price;
- implied loss from planned entry;
- explanation of what breaks at that price;
- any non-price invalidation that can force an earlier exit.

## Not a fixed stop-loss rule
The exit price must NOT be generated from a constant -3%, -5%, -8%, or other universal percentage.

The AI must reason from the specific stock:
- price structure / support failure;
- expected launch path;
- catalyst or thesis failure;
- valuation / expectation reset;
- flow or event evidence;
- counter-evidence.

## No moving the goalposts
After entry:
- failure exit price may stay unchanged;
- failure exit price may move upward as the trade succeeds;
- failure exit price must never be lowered.

## Execution realism
The exit price is a trigger, not a guaranteed fill.
If the stock gaps or is not tradable at the trigger, replay/live accounting must use the next realistically executable price with adverse gap/slippage assumptions.

## Scientific validation
Measure:
- trigger rate;
- realized loss after trigger;
- avoided tail loss;
- false exits before later +10/+20/+30/+50 winners;
- gain protection from raised exits;
- whether exit logic improves portfolio drawdown without destroying large-winner capture.
