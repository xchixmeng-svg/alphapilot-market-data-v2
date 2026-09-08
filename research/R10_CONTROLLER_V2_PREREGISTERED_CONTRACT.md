# AlphaPilot R10 Controller V2 — Pre-registered Research Contract

Status: **PRE-REGISTERED / RESEARCH_ONLY**  
Registration branch: `r10-controller-v2-preregistered-20260908`  
Formal R10 branch and locked commit are immutable: `r10-no-trail-formal-fix-20260907` / `3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9`.

## Purpose and non-promotion rule

This study tests whether a simpler, causal market controller improves risk-adjusted behavior across multiple market periods. The prior five-state search (Run 34238218590) is **EXPLORATORY_ONLY** because its state thresholds were manually encoded and 2016–2019 was used to choose exposure profiles. Neither that run nor 2021–2025 may be described as pristine unseen OOS.

No R7 or R0.5 signal, ranking, entry, exit, sizing, fee, tax, slippage, T+1, fill, common-pool, integer-share, ADV, or corporate-action rule may be changed. Controller results never overwrite formal R10 exits.

## Fixed arms (declared before execution)

- **A — No classification:** constant controller capacity on every valid day: total R7 exposure cap 0.95, five R7 slots, two R0.5 slots. The existing common-pool and per-name caps remain binding.
- **B — Locked R10 five-state:** byte-identical existing R10 controller and allocations; no profile selected from the exploratory matrix.
- **C — Simple three-state:** uses only 0050 adjusted close and past-only rolling data.
  - Risk-On: 0050 > MA120 AND 20-day return > 0 AND 60-day return > 0.
  - Risk-Off: 0050 < MA120 AND 60-day return < 0.
  - Transition: all other fully observed cases.
  - Unknown warm-up: no new positions.
  - Allocation is monotonic and fixed: Risk-On = (0.95 exposure, 5 R7 slots, 2 R0.5 slots); Transition = (0.50, 3, 1); Risk-Off/Unknown = (0.00, 0, 0).
- Breadth, volatility, and institutional flow are diagnostic continuous features only; they do not create additional discrete states.

## Fixed sensitivity grid (robustness only)

The primary C specification is MA120 with 20/60-day return windows. Sensitivity runs are the Cartesian grid:

- MA: 100, 120, 140
- short/long return windows: 15/45, 20/60, 25/75

The primary parameters stay 120 and 20/60 regardless of sensitivity results. No cell may be selected as a replacement or used to tune allocations.

## Data and causality

- Reuse preserved, audited 2015–2025 OHLCV, institutional, and official corporate-action inputs. No bulk refetch.
- 2015 is warm-up; evaluation begins 2016-01-04.
- Decisions use information available at T close and execute only at T+1 under the locked fixed precommitted buy-price/fill convention.
- Integer shares, common cash pool, full fees/taxes, adverse sell execution, and real observed prices remain enforced.
- Missing controller inputs produce Unknown/no-new-position; no backward fill and no future data.
- 2026 may be labeled YTD/forward only through the actually complete range, and only observations after the formal lock date count as forward OOS.

## Reporting slices

For A/B/C report continuous 2016–2025 and the fixed slices 2016–2019, 2020, 2021–2022, and 2023–2025. For the primary C rule also report expanding endpoints 2016, 2016–2017, …, 2016–2025 and the full sensitivity grid.

Each result must include End NAV, total return, CAGR, Max DD, PF, win rate, completed trades, average/maximum exposure, worst trade, counts <=-12%, <=-20%, <=-25%, annual returns, and top-five-winner P&L dependence.

## Candidate gate

C can be listed only as a research candidate if:

1. continuous 2016–2025 Max DD <= 22% and never >25%;
2. continuous PF >= 1.50;
3. risk-adjusted behavior is directionally stable across the fixed time slices and sensitivity grid;
4. improvement is not dependent on a small set of winning trades; and
5. no execution or data contract assertion fails.

Otherwise the conclusion is **CANNOT_PROVE / RESEARCH_INCONCLUSIVE**. No additional parameters may be added to rescue performance.
