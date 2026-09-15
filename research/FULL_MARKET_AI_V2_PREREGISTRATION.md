# AlphaPilot Full-Market AI V2 — Preregistration

## Diagnosis from V1
V1 proved that full-market selection is technically causal and executable, but its absolute-return accept gate was far too broad: roughly half of the executable universe was accepted each day. That is structurally misaligned with a five-position portfolio. V1 is therefore a research FAIL, not evidence that full-market AI is invalid.

## V2 hypothesis
For stock selection, the model should learn **cross-sectional relative alpha** rather than absolute 20-session return. The portfolio needs only a few scarce winners, not hundreds of names with slightly positive forecasts.

## Locked changes vs V1
1. Universe remains full executable TWSE/TPEx market; R10 `r7_hard` / `r05_hard` are diagnostic only and never eligibility gates.
2. Same immutable data, corporate-action handling, T+1 fill model, fees/tax/slippage, integer shares, common cash, 5-position cap, 25% single-name cap, 95% total cap and drawdown controls.
3. Same expanding yearly OOS and 60-session purge.
4. Market context remains automatically identified from causal market features using KMeans fit only on prior data.
5. Training target becomes 20-session **relative alpha**: stock net 20-session return minus the same signal-date executable-universe median net return.
6. Failure classifier remains absolute: realized 20-session net return <= -5%.
7. Model family and hyperparameters remain fixed HistGradientBoosting; no grid search.
8. Selection is intentionally sparse and portfolio-capacity-aligned: rank every executable stock each day by predicted relative alpha and predicted failure safety; retain only the daily top 10 by the fixed composite score. This constant is structural (2x the maximum 5 holdings), not tuned from V1 performance.
9. A daily candidate must be in that top 10 and have predicted relative alpha > 0. No broad absolute-return accept gate.
10. Entry/exit remains a 20-session AI strategy for target alignment; no use of future/current-year realized labels for selection.

## Validation
Final success is based only on full portfolio replay vs frozen R10:
- CAGR strictly > R10 (13.110267%)
- PF >= R10 (1.862730946)
- Max DD no more than 1 percentage point worse than R10 (~ -20.5154% floor)
- data SHA / locked R10 fingerprint / contract audit all PASS

Model diagnostics, top-decile spreads, win rate, or CI success alone are not success.
