# AlphaPilot empirical plan-v3 research protocol

Status: research only. This branch must not modify AlphaPilot Forward, R10-MAX, or the main data layer.

## Frozen target

- Decision at T close.
- T+1 open entry.
- Cancel if T+1 has no trade data, opens at or above T close + 1 ATR20, or is locked limit-up.
- No intraday exit on T+1.
- From day 2 through day 30, success means +15% is reached before the stop.
- Stop distance = 1.5 × ATR20(T) / T close, clipped to 5%–8%.
- If stop and +15% are both touched on the same day, count failure.
- If +5% has never been reached by day 10, count failure.
- If +15% has not been reached by day 30, count failure.

## Validation discipline

- Historical source: data/history/2020-2025 only.
- 2020–2023: base model fit.
- 2024: probability calibration only.
- 2025: untouched validation.
- No feature selection or hyper-parameter tuning may use 2025 results.
- After validation is frozen, the 2026 shadow scorer may fit through 2024 and calibrate on 2025.
- Claude/LLM probability is not a model feature and cannot be used as the empirical gate.
- 2026-10-02 outcome is forbidden from training or calibration.

## Initial shared features

Only numeric fields that can be reproduced historically and are present in the 2026 Forward snapshot:

- 5-day return
- 20-day return
- ATR20 / close
- current amount / 20-day average amount
- close / MA20 - 1
- close / MA60 - 1
- 5-day three-institution net / 5-day volume
- distance from estimated foreign+trust 60-day cost, in ATR units

## Output

The first run will produce:

1. 2025 out-of-sample AUC, Brier score, base rate, and calibration buckets.
2. Empirical p30 for the 2026-10-02 finalists.
3. Edge versus the existing ATR-matched base rate.
4. A shadow-only comparison using the old +8 percentage-point edge rule.
5. No live Forward file changes.
