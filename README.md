# AlphaPilot Market Data

`main` is the canonical **data layer only**. Strategy research and formal trading logic must live on separate branches.

## Canonical branch roles

- `main` — Taiwan-market data acquisition, historical archives, and reusable data-building utilities only.
- `r10-no-trail-formal-fix-20260907` — locked formal R10-MAX. Do not modify. Locked commit: `3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9`.
- `future-alpha-forward-20260910` — forward-only Future Alpha research. Must not modify formal R10-MAX.

## Data kept on main

- `data/YYYY-MM-DD/` — 2026 daily TWSE / TPEx raw and normalized OHLCV + institutional snapshots.
- `data/history/2007-2019/` — verified yearly historical OHLCV archives + manifest.
- `data/history/2020-2025/` — reusable OHLCV parquet + institutional parquet + hashes/audit.

## Canonical acquisition / build scripts

- `scripts/fetch_today.py` — official daily TWSE / TPEx OHLCV + institutional acquisition and normalization.
- `scripts/fetch_history_2007_2019.py` — historical archive acquisition / verification tooling.
- `scripts/build_2026_fundamental_layer.py` — 2026 fundamental layer builder.
- `scripts/build_full_market_fundamental_sharded.py` — full-market fundamental builder.
- `scripts/build_2026_ytd_package.py` — reusable 2026 YTD data package builder.

## Active main workflows

- `data_only.yml` — weekday daily market-data acquisition. Scheduled 17:40 Asia/Taipei with 18:40 backup.
- `history_2007_2019.yml` — historical archive reconstruction / verification.
- `build_2026_fundamental_layer.yml`
- `build_full_market_fundamental_sharded.yml`
- `build_2026_ytd_package.yml`

## Repository rule

Do not add backtests, factor tournaments, one-off stock research, strategy tuning, formal execution rules, or temporary export helpers to `main`. Put research on a dedicated branch. Historical Git commits preserve removed research files if they ever need to be recovered.
