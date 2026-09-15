# AlphaPilot Full-Market AI V1 — Preregistration

## Goal
Test whether a causal AI selector that can see the full executable Taiwan equity universe can beat locked R10 MAX without using R10 hard eligibility as a candidate gate.

## Locked baseline
R10 MAX formal no-trailing-profit fix remains immutable and is used only as the benchmark plus the source of the already-audited causal market/corporate-action data pipeline and portfolio execution mechanics.

Baseline fingerprint:
- branch: `r10-no-trail-formal-fix-20260907`
- commit: `3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9`
- engine SHA256: `2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0`

## Full-market universe
AI does **not** require `r7_hard` or `r05_hard`.
The daily candidate universe is only constrained by execution/data feasibility:
- ordinary 4-digit Taiwan equity code beginning 1–9,
- exclude names containing KY,
- 20-session average traded value >= NT$30m,
- 120-session history available,
- required causal features available.

Thus a stock omitted by both R10 candidate engines remains visible to AI.

## Causality / OOS
- 2020 is training/warm-up only, never counted as portfolio evaluation.
- Evaluation: 2021-01-04 through 2025-12-31.
- Yearly expanding OOS folds: 2021, 2022, 2023, 2024, 2025.
- Each fold uses only observations strictly before a 60-market-session purge boundary ahead of the test year.
- No current-year or future labels may affect model fitting, context discovery, thresholds, or candidate ranking.
- No hyperparameter search on 2021–2025.

## Automatic context identification
For each OOS fold, daily market context is learned only from prior data with:
- StandardScaler,
- KMeans(n_clusters=3, n_init=20, random_state=42),
- context inputs: 0050 20/60-session return, gaps to MA60/MA120, market breadth above MA60, 10-session advancing breadth.

The current test date is assigned to one of the prior-data clusters. A separate model is used for the matched context when historical support is sufficient; otherwise the fold-global model is used. Context labels are never hand-tuned to outcomes.

## Stock features
Only T-close causal information:
- R7/R0.5 composite scores as continuous factors only (never eligibility gates),
- amount ratio / acceleration,
- 5/10/20/60-session returns,
- distance from 120-session high,
- gaps to MA20/60/120,
- MA20/MA60 and MA60/MA120 structure,
- log 20-session traded value and volume,
- low-base flag,
- the same causal market context inputs listed above.

## Target and model
Single preregistered 20-session target aligned to the portfolio holding horizon.
Historical labels assume:
- signal at T close,
- fixed T+1 buy limit = 99.5% of T close,
- locked R10 buy adverse slippage and fees,
- 20 sessions after T+1 entry, exit using next target-session open-equivalent with locked sell adverse slippage/fees/tax.

Models per context/fold:
- HistGradientBoostingRegressor for net 20-session return,
- HistGradientBoostingClassifier for probability of return <= -5%,
- fixed model parameters; no search.

Training computation uses every fifth prior market date only; this is deterministic sampling of the entire executable universe, not a stock-selection filter. All executable stocks on every OOS test date are scored.

## AI ranking / acceptance
- `pred_return > 0`
- `pred_fail_prob < 0.50`
- Daily score is equal-weight cross-sectional percentile rank of predicted return and predicted safety (`1 - pred_fail_prob`).
- The execution engine receives the top 50 accepted names per date only for storage efficiency; ranking is computed after scoring the full daily executable universe.

## Portfolio / execution
R10 selection rules are removed for AI entries.
R10 execution/risk mechanics remain:
- T+1 only,
- fixed precommitted limit,
- raw execution prices,
- buy/sell adverse slippage,
- full fees and sell tax,
- integer shares,
- common cash pool,
- max five positions,
- max 25% NAV per stock,
- max 95% total exposure,
- 2% of 20-session average volume sizing cap,
- corporate actions only on official effective dates,
- portfolio drawdown de-risking / no-entry cooldown unchanged.

AI position sizing is fixed at 20% NAV per slot before existing drawdown multiplier and caps.
AI positions use a fixed 20-session maximum holding period with no profit trailing rule. Global R10 portfolio drawdown controls may still force an earlier exit.

## Success gate
Research SUCCESS only if the full 2021–2025 portfolio simultaneously satisfies:
1. CAGR > locked R10 CAGR 13.11026701635103%, strictly,
2. profit factor >= 1.8627309459518986,
3. Max DD >= R10 Max DD - 1 percentage point (about -20.5154% or better),
4. data SHA / locked-engine fingerprint / contract audit all PASS.

CI success, AUC, Spearman correlation, top-decile return, or any model-only metric is not research success.

## Anti-overfit rule
After the first valid 2021–2025 run, parameters above are not changed merely to improve that same sample. A subsequent version must have a new preregistered structural hypothesis and version name.
