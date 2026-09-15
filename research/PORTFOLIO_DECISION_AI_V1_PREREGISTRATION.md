# AlphaPilot Portfolio Decision AI V1 — Preregistration

## Research rule
This is a framework change, not a parameter tune. Full-Market AI V1–V4 are closed as a fixed-20D stock-forecast family. No V4 threshold/weight/Top-N tuning is allowed unless a later framework gets close to the locked R10 portfolio gate.

## Question
At a real portfolio slot decision, should the system KEEP the R10 candidate, REPLACE it with a full-market candidate, or HOLD CASH?

The AI is judged on portfolio P&L, not model score.

## Immutable baseline
- Locked R10 branch/commit: r10-no-trail-formal-fix-20260907 / 3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9
- Locked engine SHA256: 2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0
- Initial capital: NT$1,300,000
- T+1 only, raw fills, full fees/tax, integer shares, common cash, corporate actions, max 5 positions, single cap 25%, total cap 95%.

## Full-market candidate universe
R10 hard flags are NOT AI eligibility gates. Candidate rows require only:
- ordinary 4-digit Taiwan stock code, excluding KY;
- 20-day average traded value >= NT$30m;
- enough history for causal features;
- T+1 execution feasibility under the precommitted AI limit convention.

## AI replacement policy used for labels and execution
A replacement is its own strategy policy, so labels match executable behavior:
- signal at T close;
- T+1 fixed limit = 99.5% of T close, with the locked adverse-buy fill model;
- minimum hold 3 sessions;
- hard stop signal at adjusted return <= -10%, executed T+1 with locked adverse sell/slippage/fees/tax;
- otherwise max-hold signal at 20 sessions, executed T+1;
- no hindsight carry to T+2 when an order cannot execute.

## Decision-context training
The training target is the realized net return of the above slot policy, not a generic forward close-to-close return.
- 2021 bootstrap: causal 2020 executable rows.
- 2022–2025: 2020 bootstrap plus prior-year rows occurring on actual R10 BUY signal dates. R10 signal dates come from submitted T-close orders, not future fill status.
- 60-session purge before each test year.
- expanding OOS only; no use of test-year outcomes for thresholds or retraining.

## Runtime action rule
At each actual runtime R10 entry frontier:
1. score the current R10 candidate with the OOS action-value model;
2. compare it with the best unused full-market AI candidate available that day;
3. REPLACE only when the AI candidate has predicted net slot return > 0, strictly exceeds the R10 candidate prediction, and has no higher predicted loss probability;
4. HOLD CASH only when the R10 candidate prediction is <= 0 and no eligible replacement dominates it;
5. otherwise KEEP R10.

No fitted margin, no tuned Top-N threshold, no portfolio-result feedback into V1.

## Model
Fixed causal tabular models:
- HistGradientBoostingRegressor for policy net return;
- HistGradientBoostingClassifier for probability(policy net return <= 0).
Hyperparameters are fixed before observing V1 portfolio results.

## Portfolio evaluation
2021–2025 full replay. Success requires ALL:
- CAGR > 13.11026701635103%
- PF >= 1.8627309459518986
- Max DD >= -20.515438907459803%
- baseline and variant contract audits PASS.

If far from the gate, the next version must change framework/target/action design rather than micro-tune weights, cutoffs, Top-N, or model hyperparameters.