# Independent Buy Selector V2 — preregistration

Purpose: improve absolute buy-candidate quality without adding any holding-period or sell rule.

Hard independence:
- use only immutable 2020–2025 OHLCV + institutional data;
- no R10/R7/R0.5 candidate, score, execution or portfolio logic;
- 5/20/60/120-day future outcomes are diagnostics/labels only, never holding limits.

Universe/features/OOS folds:
- identical to Independent Buy Selector V1;
- 2021–2025 expanding annual OOS with 120-session purge;
- TOPN fixed at 10; no post-result threshold tuning.

Framework change versus V1:
- V1 predicts cross-sectional multi-horizon relative quality only.
- V2 adds a separate classifier for persistent absolute strength: fwd20 > 0 AND fwd60 > 0 AND fwd120 > 0.
- Daily ranking is lexicographic in effect: persistent-positive probability is primary; predicted V1 relative quality is only a tiny deterministic tie-break (score = p_persistent + 1e-3 * predicted_relative_quality).
- This is deliberately not a tuned blend weight.

Research gate, fixed before run:
1. overall mean absolute returns at 20/60/120 days > 0;
2. at least 4 of 5 OOS years have positive mean absolute returns at each of 20/60/120 days;
3. overall mean alpha versus same-day executable-universe median > 0 at 5/20/60/120 days;
4. selected persistent-positive rate > executable-universe persistent-positive rate.

Passing this gate means only that the independent selector is promising. It is not a complete trading strategy and does not define exits, holding periods, CAGR, PF or drawdown.
