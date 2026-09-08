# AlphaPilot Regime Controller V2 — Preregistered Research Contract

## Status and isolation

- Research-only branch. The formal R10 MAX branch `r10-no-trail-formal-fix-20260907` and commit `3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9` are immutable.
- Phase-1 run `34238218590` is `EXPLORATORY_ONLY`; its five-state thresholds and allocation search cannot be promoted.
- Existing audited OHLCV, institutional and official corporate-action data must be reused. No large historical refetch.
- 2015 is warm-up; evaluation is continuous 2016–2025 with one common cash pool.

## Compared controllers

A. `always_on`: no market-state exposure controller; diagnostic only.
B. `locked_five`: the existing R10 five-state classifier, unchanged; diagnostic only.
C. `simple_3`: the only V2 candidate.

The preregistered three states are:

- `Risk-On`: 0050 close > MA120 AND 20-session return > 0 AND 60-session return > 0.
- `Risk-Off`: 0050 close < MA120 AND 60-session return < 0.
- `Transition`: all other observable cases.
- `Unknown`: insufficient warm-up data.

Fixed monotone allocation:

- Risk-On: R7 exposure 100%, up to five R7 slots, up to one R0.5 slot.
- Transition: R7 exposure 40%, up to two R7 slots, up to one R0.5 slot.
- Risk-Off/Unknown: no new positions.

R7 and R0.5 selection, exits, fees, taxes, slippage and sizing are not changed.

## Sensitivity, not optimization

The following perturbations are reported but may not be selected as new parameters:

- MA100 / MA120 / MA140 with 20- and 60-session returns.
- 15/50 and 25/70 return windows with MA120.

The MA120 + 20/60 specification remains the candidate regardless of which perturbation performs best. Perturbations test fragility only.

## Time analysis

Report continuous common-pool results and causal slices for:

- 2016–2019
- 2020
- 2021–2022
- 2023–2025
- continuous 2016–2025

These are temporal robustness checks, not pristine unseen data because this project has previously inspected historical periods. Only data after the final lock date may be called forward OOS.

## Acceptance

Candidate `simple_3` requires:

- continuous Max DD no worse than -22%;
- continuous PF at least 1.50;
- no sensitivity variant worse than -25% Max DD;
- median sensitivity PF at least 1.35;
- no future information, T+1 execution, integer shares, nonnegative cash, common pool, official corporate actions, original fees/tax/slippage;
- results must disclose CAGR, return, Max DD, PF, win rate, exposure, worst trade, large-loss counts, annual results and top-five-winner dependence.

Failure must be reported as failure. Do not add thresholds or select the best sensitivity setting after seeing results.
