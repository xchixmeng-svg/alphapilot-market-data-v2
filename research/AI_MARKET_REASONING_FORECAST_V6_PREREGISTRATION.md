# AlphaPilot V6 — Independent AI Market Reasoning + Path Forecast Engine

Status: PREREGISTERED BEFORE ANY V6 OOS RESULT IS VIEWED.

## 1. Research question
V6 is not a Top-N ranker and not a fixed-factor trading rule. It must answer, from information available at the decision close only:

1. What market/industry/company state is forming now?
2. Is a new upward price leg likely to begin or continue?
3. What entry-price zone has the best conditional reward/risk and fill probability?
4. How far can this price leg plausibly travel?
5. What is the calibrated probability of reaching each target level?
6. In what future time window is the main high most likely to form?
7. At what price/time region does marginal expected upside cease to justify incremental reversal/downside risk?

The model may output NONE. No fixed daily candidate count is allowed.

## 2. Independence
V6 must not use R10/R7/R0.5 or any legacy candidate pool, score, holding rule, stop rule, or exit engine. Legacy strategies may be external benchmarks only after V6 outputs are frozen.

## 3. Causality / point-in-time rule
Every OOS feature must have been available by that historical decision close. Report/publication dates are required for fundamentals. Future reports, revised future values, future sector membership, future news, and future labels are prohibited from features.

A data source that cannot be made point-in-time clean may be used for current live explanation only and MUST NOT receive OOS credit.

## 4. Information layers
### A. Stock tape and capital flow
Raw OHLCV path, gaps, intraday range/body, turnover/amount changes, foreign/trust/dealer flow, volatility compression/expansion, drawdown/recovery and relative-strength path. These are observations, not hand-written buy rules.

### B. Market and industry/peer rotation
Daily and rolling industry/peer return, breadth, volume share, dispersion, leadership persistence, relative drawdown/recovery and capital-flow concentration. The system must learn whether money is rotating into/out of a group rather than use rules such as `industry up => buy`.

For historical OOS, industry/peer membership must be point-in-time. If historical semantic membership cannot be verified, V6 must use past-only dynamic peer clusters or disable that semantic field for OOS. Current MOPS industry labels may still be shown in live explanations.

### C. Seasonality / calendar
Calendar position is supplied as neutral cyclical/time representation (month/day-of-year/holiday-distance where auditable). The model is allowed to learn airline/travel/summer-appliance or other seasonality, but rules such as `June => buy airline` are forbidden.

### D. Fundamentals and valuation
Point-in-time monthly revenue, revenue acceleration, reported margins/profit/EPS, valuation history and valuation-vs-own-history. EPS/valuation revision proxies must be constructed only from information released by that date. No paid historical consensus is assumed unless a timestamped source is later added.

### E. Macro and industry external state
Where auditable historical series are available: policy/risk-free rates, yield curve, FX, oil/energy, volatility and relevant commodity/freight/industry-price series. The model learns stock/industry sensitivities jointly with the current state; no rule like `freight +10% => buy shipping` is allowed.

### F. News/events
Timestamped historical text/events may be added only if an auditable archive is obtained. Until then, live news can be a SHADOW explanation/event feature but cannot be credited to historical OOS performance. This prevents hindsight contamination.

## 5. Forecast outputs (not holding rules)
For each stock/date V6 must produce a probability distribution, not one opaque score:

- probability that an upward leg is beginning/continuing;
- calibrated hit probabilities for future return levels (+5%, +10%, +15%, +20%, +30% where support exists) over multiple forecast windows;
- downside probabilities (at minimum first-touch -5% and -10% where support exists);
- future return quantiles;
- MFE/MAE quantiles;
- first-hit / time-to-target distribution;
- time-to-major-high distribution;
- conservative/base/optimistic target-price zone;
- candidate entry zone with predicted fill probability and conditional upside/downside distribution;
- price/time region where incremental upside becomes inferior to incremental reversal/downside risk.

The 5/10/20/40/60/120-session windows are forecast axes only. They are NOT forced holding periods.

## 6. Entry-zone definition
V6 must not say `buy now` merely because a stock has the highest score. It must compare plausible entry prices using only forecasts available at T close. For each entry candidate it estimates:

- probability price reaches/fills the entry zone;
- conditional probability of reaching each upside target after fill;
- conditional MAE/downside distribution after fill;
- expected upside distribution and uncertainty.

The preferred entry is the region with the strongest calibrated conditional reward/risk, not necessarily the lowest price and not necessarily the current close.

## 7. Target and exit-zone definition
Target price is a distribution. The engine must show conservative/base/optimistic zones and calibrated hit probability for each.

The preferred sell/harvest region is NOT a fixed +X% take-profit and NOT a fixed N-day exit. It is the price/time region where modelled marginal expected upside falls below modelled incremental reversal/downside risk. This is a forecast diagnostic until a separate execution/exit policy is formally preregistered.

## 8. Probability calibration
Raw model probabilities are not allowed to be presented as real probabilities. Every probability shown to the user must be calibrated on past-only rolling calibration data.

Required diagnostics include calibration curve, Brier score, expected calibration error (ECE), sample count per probability bucket and Wilson confidence intervals for target-hit events.

A live `>=90%` label is only allowed if the corresponding historical calibrated bucket demonstrates high realized hit rate with sufficient sample count. Otherwise the engine must report a lower calibrated probability or NONE.

## 9. OOS protocol
Primary formal OOS: 2021–2025 walk-forward / expanding history with purge at least equal to the longest forward label horizon. 2026 is current/live inference and later forward evidence, not tuning data for 2021–2025.

No parameter change may be justified using the same OOS years after viewing their results. A principled architecture change receives a new version.

## 10. Success priorities
Precision dominates signal quantity. A week with no candidate is acceptable.

V6 is only promising if all of the following hold on untouched OOS evidence:

1. calibrated high-confidence buckets materially outperform unconditional target-hit base rates;
2. probability calibration is directionally valid (higher predicted probability => higher realized hit rate) across years;
3. 20/60/120-day selected alpha is positive overall and not dependent on one single year;
4. selected adverse excursion is no worse than the investable universe at 60/120-day diagnostics;
5. target-price forecasts show useful interval coverage rather than systematic overstatement;
6. abstention is real — the system can emit NONE;
7. current recommendations are sparse enough to be actionable, with no fixed Top-N quota.

Formal production BUY remains stricter: user-facing BUY should target calibrated probability >= 90% for the stated target/event. If no such opportunity exists, output NONE/WAIT.

## 11. Interpretation
V6 is intended to become an AI research/forecast engine that understands market rotation, business/fundamental change, macro/industry context, seasonality and price structure jointly. It is not allowed to claim certainty or `必漲`; it must quantify uncertainty and be falsifiable.