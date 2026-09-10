# Future Alpha Forward Contract v0.1 — frozen 2026-09-10

Purpose: discover signals that make money in the future, not optimize explanations of the past.

## Hard rules
- Forward-only. No historical parameter fitting after this contract is frozen.
- Every prediction uses information available by Taiwan market close on T and is eligible only for T+1 execution/evaluation.
- No backdating. Because the project was created during the 2026-09-10 session, 2026-09-09 or earlier may be used only as feature history, never as a newly claimed prediction date.
- First eligible prediction date is 2026-09-10 after official TWSE/TPEx end-of-day data pass the data gate; first hypothetical execution date is T+1.
- A published prediction row is append-only and must never be edited, deleted, relabelled, or re-ranked after future returns become known.
- Formal R10-MAX is isolated and untouched. Future Alpha is research until forward evidence justifies promotion.
- T+1 is mandatory. No same-day hindsight, no intraday future information.

## Frozen v0.1 hypothesis
The first hypothesis is structural rather than a historical winner search:
1. Institutional accumulation persistence — repeated foreign/investment-trust net buying relative to trading volume.
2. Absorption — capital enters while price resists selling pressure instead of collapsing.
3. Relative-strength confirmation — price begins to outperform the current cross-section; price is a confirmation, not the original causal story.
4. Liquidity/participation expansion — trading value rises enough that the move can support real money.

Weights are frozen before the first valid prediction:
- institutional persistence 35%
- absorption 25%
- relative-strength confirmation 20%
- liquidity/participation expansion 20%

These weights must not be altered using the results of predictions issued under v0.1. A genuinely different hypothesis requires a new model version and a new forward ledger; old rows remain immutable.

## Prediction output
Each valid T close produces at most 10 ranked research candidates. For each candidate record: model_version, prediction_date, next_trade_date flag, code/name/market, close, frozen component scores, total score, data-history length, confidence tier, input fingerprint.

This is NOT automatically a BUY order. It is a forward prediction ledger used to determine whether a new strategy deserves to exist.

## Evaluation
For every prediction, evaluate future close-to-close returns after 20, 60, and 120 trading days, plus future maximum adverse excursion and maximum favourable excursion when enough data arrive. Evaluation is mechanical and cannot change original ranks/scores.

Primary research question: does the frozen signal produce repeatable positive forward returns with tolerable drawdown? 0050 is reference context, not a mandatory pass/fail benchmark.
