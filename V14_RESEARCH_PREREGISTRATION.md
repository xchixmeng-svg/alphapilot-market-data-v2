# AlphaPilot AI V14 — Decision-Level OOS Policy Gate

## Reason for V14
V13 was a valid research failure: contextual neighbor shadow evidence looked positive but the actual 2025 portfolio return fell from 32.52% to 26.42%, reducing CAGR by 1.063 pp. The failure shows that standalone strategy-path pair uplift is not sufficiently aligned with portfolio economics.

## Locked invariants
- R10 formal engine SHA256 and locked commit unchanged.
- T+1 causal execution only.
- Full fees/tax/slippage, integer shares, corporate actions, common cash, caps, exits unchanged.
- AI can only perform one-rank strategy-slot frontier substitution (rank N vs N+1); it cannot add candidates or delete arbitrary R10 candidates.
- Expanding OOS only; no current/future-year outcomes in policy decisions.
- Success remains strict portfolio gate: CAGR > R10, PF >= R10, Max DD no worse than R10 by more than 1 pp.

## V14 research change
Replace V13's repeated application of a route-level shadow statistic with a decision-level causal policy gate.

For every historical OOS frontier opportunity, construct a counterfactual intervention record tied to the exact date, strategy, market state and current candidate margins. Historical evidence is admitted only after its outcome is fully observable before the new decision date. The router uses local prior evidence and must abstain when support is weak.

### Anti-overfit constraints
- No tuning against 2025 portfolio result.
- No retrospective selection of thresholds after seeing V14 output.
- Context hierarchy remains causal: strategy+state -> strategy -> global.
- A context must have adequate prior OOS support and positive realized intervention evidence; otherwise abstain.
- Add a cooldown/deduplication rule so the same small historical evidence pool cannot authorize hundreds of near-identical daily substitutions. One evidence regime may authorize at most one live intervention until the frontier/context materially changes or new fully-observed OOS evidence becomes available.

## Primary diagnosis being tested
V13 issued 395 substitutions in 2025 from only 44 total historical path-training examples, including many decisions repeatedly justified by the same nine R7/NORMAL neighbors. V14 tests whether preventing this evidence reuse and calibrating at the actual decision level restores economic alignment.

CI success is not research success.