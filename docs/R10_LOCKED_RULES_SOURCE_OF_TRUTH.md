# AlphaPilot R10-MAX — Locked Rules Source of Truth

This document is the executable/audit reference for R10 historical validation. It consolidates the rules already present in `docs/R10_STRESS_BACKTEST.md`, `scripts/build_r10_scan.py`, and `scripts/backtest_r10_stress.py`. Historical validation must reproduce these rules rather than silently disabling them.

## 1. Causality / no look-ahead

- Every signal is formed using information available by T close only.
- Buy/sell decisions formed at T execute on T+1 only.
- No future bar, future institutional data, future rank, or future price may affect a T decision.
- R7 and R0.5 are independent scanners. They may compete only at the common portfolio/capital layer after each scanner has produced its own candidates.

## 2. R7 scanner

Market/breadth inputs use the locked 0050 total-return benchmark and current/past market breadth only.

Regimes:
- Bear: `mr20 <= -8%` OR (`mkt < MA120` AND `mr60 < 0` AND breadth < 40%): exposure 0%, slots 0.
- Repair: `mkt < 1.02*MA120`, `mr20 > 0`, breadth > 42%, breadth > 20D breadth mean: exposure 60%, slots 2.
- Strong Bull: `mkt > MA60`, `mkt > MA120`, `mr20 > 0`, `mr60 > 0`, breadth >= 60%, advance10 >= 52%: exposure 100%, slots 4.
- Normal Bull: `mkt > MA120`, `mr60 > 0`, breadth >= 45%: exposure 80%, slots 3.
- Weak: `mkt > 0.98*MA120`, breadth >= 38%: exposure 20%, slots 2.
- Otherwise Fallback/Bear: exposure 0%, slots 0.

R7 score:
`0.26*p10 + 0.22*pRel20 + 0.10*pRel60 + 0.14*pFlow20 + 0.12*pAmtAcc + 0.08*pCLVFlow20 + 0.08*pNearHigh`

R7 eligibility:
- 20D average traded value >= NT$30m.
- adjusted close > MA120.
- near-high ratio >= 0.78.
- valid R7 score.

R7 rebalance clock:
- initial scan, regime change, or every 15 trading days.

R7 entry:
- target size = 22% of current portfolio NAV before DD throttle and portfolio caps.
- T+1 limit = T adjusted/locked close × 0.98, rounded down to legal tick.

R7 current hard-stop implementation:
- if T adjusted close <= adjusted entry price × 0.88, create HARD sell at T close; execute T+1.
- rebalance exits also apply when regime exposure becomes zero, rank falls outside retained range, or R7 exposure exceeds regime target.

## 3. R0.5 scanner

R0.5 uses institutional and microstructure references independently of R7:
- `Foreign3D`
- `Foreign10D`
- `Trust5D`
- `clvflow10`
- `amount_ratio`
- `clvflow5`
- `ma20gap`

R0.5 score:
`0.5251*pCLVFlow10 + 0.2465*pAmountRatio + 0.0683*pCLVFlow5 + 0.0628*pForeign3D - 0.0778*pForeign10D + 0.0195*pTrust5D - 0.20*pMA20Gap`

R0.5 market risk-on requires all:
- 0050 close > MA60.
- 20D return > 0.
- 60D return > 0.

R0.5 candidate filters:
- price between NT$10 and NT$40.
- 20D average traded value >= NT$50m.
- amount ratio >= 1.
- 20D return between 0% and +20%.
- MA20 gap <= 18%.
- prior-60D-high position >= -15%.
- adjusted close > prior 10D high.
- valid R0.5 score.

R0.5 entry:
- target size = 20% of current portfolio NAV before DD throttle and portfolio caps.
- maximum 3 R0.5 slots.
- T+1 limit = T close × 0.995, rounded down to legal tick.

R0.5 current exit implementation:
- HARD: adjusted close <= adjusted entry × 0.90, decision T, execute T+1.
- NORMAL can become RUNNER at >= +40% with amount_ratio >= 2.0.
- RUNNER/MEGA/TARGET state transitions and trailing exits use only T-known values.
- RUNNER trail: 14% from peak; max hold 120 trading days.
- MEGA trail: 16% from peak; max hold 120 trading days.
- TARGET: +200% target or 20% peak trail; max hold 120 trading days.
- BASE trail: once return >= +50%, 12% peak trail.
- NORMAL time exit: 60 trading days.

## 4. Locked common portfolio layer

- Initial NAV: NT$1,300,000.
- One common capital pool; no fixed 62/38 sleeves.
- R7 new position base = 22% NAV.
- R0.5 new position base = 20% NAV.
- Maximum 5 distinct stocks.
- Same stock across strategies combined <= 25% NAV.
- Normal total exposure <= 95% NAV.
- Cash cannot be negative; no borrowing or leverage.
- If enough cash exists for a board lot, quantity must be a multiple of 1,000 shares. Odd-lot integer shares are permitted only when one full board lot itself exceeds the target capital.
- Per-order liquidity <= 2% of T-known 20D average volume.

DD throttle for new positions:
- DD > -6%: 100% base size.
- DD <= -6%: 85% base size.
- DD <= -9%: 45% base size.
- DD <= -15%: 40% base size.

Portfolio DD defense:
- at DD <= -14%, T close creates weakest-first sells for T+1 until target exposure is approximately 50%.
- no new buys for 10 trading days.
- defensive cooldown 15 trading days before another force-DD event.

These DD/ADV controls are part of the documented LOCKED portfolio layer and must not be disabled by a benchmark-validation profile.

## 5. Execution and costs

- Buy fill: if T+1 Open <= precommitted limit, fill at Open; else if T+1 Low <= limit, fill at limit; otherwise cancel/no chase.
- Sell: decision at T close, execute from T+1 Open with 0.5% adverse sell slippage, rounded down to legal tick.
- Buy fee: 0.0855%.
- Sell fee: 0.0855%.
- Sell tax: 0.3%.
- Integer shares only.

## 6. Locked five-year regression benchmark

2021–2025 benchmark used for engine-equivalence testing:
- End NAV: NT$9,888,538.413551485.
- CAGR: 50.192706%.
- Max DD: -12.258760%.
- Completed trades: 241.

Any reconstructed engine that materially differs from this benchmark is not to be presented as official-equivalent R10 until the first divergence is identified and resolved.

## 7. Prohibited changes during validation

- Do not alter R7/R0.5 scores, filters, stops, position sizes, or execution rules merely to improve backtest output.
- Do not use T+1 intraday information to revise the T order.
- Do not count untouched limits as fills.
- Do not silently disable DD defense or liquidity rules.
- Do not fabricate missing institutional history.
