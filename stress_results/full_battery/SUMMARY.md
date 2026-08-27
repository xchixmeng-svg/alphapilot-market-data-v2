# AlphaPilot R10-MAX Full Stress Battery

**Regression Gate: FAIL**

| Scenario | Status | Trust | End NAV | CAGR | Max DD | Trades |
|---|---|---|---:|---:|---:|---:|
| validation2021_2025 | ERROR | REGRESSION_GATE | — | — | — | — |
| gfc2008 | ERROR | RESEARCH_RECONSTRUCTION_ONLY | — | — | — | — |
| euro2011 | ERROR | RESEARCH_RECONSTRUCTION_ONLY | — | — | — | — |
| china2015 | PASS | RESEARCH_RECONSTRUCTION_ONLY | 1,136,443 | -12.75% | -14.63% | 24 |
| tradewar2018 | PASS | RESEARCH_RECONSTRUCTION_ONLY | 1,382,672 | 6.46% | -10.35% | 56 |
| covid2020 | PASS | RESEARCH_RECONSTRUCTION_ONLY | 1,448,055 | 11.43% | -12.94% | 57 |
| bear2022 | PASS | RESEARCH_RECONSTRUCTION_ONLY | 1,411,813 | 8.71% | -10.62% | 34 |
| crash2024 | PASS | RESEARCH_RECONSTRUCTION_ONLY | 1,940,177 | 49.45% | -16.25% | 39 |
| tariff2025 | ERROR | RESEARCH_RECONSTRUCTION_ONLY | — | — | — | — |
| synthetic_tail | PASS | DETERMINISTIC_DIAGNOSTIC | — | — | 2LD -18.05%; 3LD -25.74% | — |

## Trust rule

- PASS regression: post-2012 FULL_R10 historical reconstructions may be used as validated reconstructions.
- FAIL regression: all reconstructed historical performance remains Research only; no parameter tuning is allowed to force a match.
- 2008/2011 remain R7-only partial tests because required TWSE daily stock-level institutional inputs do not exist.
- 2024/2025 overlap the locked development sample and are event diagnostics, not independent OOS evidence.
