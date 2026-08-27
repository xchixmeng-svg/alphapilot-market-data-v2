# AlphaPilot R10-MAX Full Stress Battery

**Regression Gate: FAIL**

| Scenario | Status | Trust | End NAV | CAGR | Max DD | Trades |
|---|---|---|---:|---:|---:|---:|
| validation2021_2025 | MISSING | NOT_AVAILABLE | — | — | — | — |
| gfc2008 | MISSING | NOT_AVAILABLE | — | — | — | — |
| euro2011 | MISSING | NOT_AVAILABLE | — | — | — | — |
| china2015 | MISSING | NOT_AVAILABLE | — | — | — | — |
| tradewar2018 | MISSING | NOT_AVAILABLE | — | — | — | — |
| covid2020 | MISSING | NOT_AVAILABLE | — | — | — | — |
| bear2022 | MISSING | NOT_AVAILABLE | — | — | — | — |
| crash2024 | MISSING | NOT_AVAILABLE | — | — | — | — |
| tariff2025 | MISSING | NOT_AVAILABLE | — | — | — | — |
| synthetic_tail | PASS | DETERMINISTIC_DIAGNOSTIC | — | — | 2LD -18.05%; 3LD -25.74% | — |

## Trust rule

- PASS regression: post-2012 FULL_R10 historical reconstructions may be used as validated reconstructions.
- FAIL regression: all reconstructed historical performance remains Research only; no parameter tuning is allowed to force a match.
- 2008/2011 remain R7-only partial tests because required TWSE daily stock-level institutional inputs do not exist.
- 2024/2025 overlap the locked development sample and are event diagnostics, not independent OOS evidence.
