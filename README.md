# AlphaPilot Market Data — Clean Reset

This `main` branch is intentionally **data-only**.

Kept:
- downloaded Taiwan market data under `data/`;
- raw and normalized TWSE / TPEx OHLCV and institutional data;
- `scripts/fetch_today.py` for daily data acquisition;
- one data-only GitHub Actions workflow.

Removed from `main`:
- R7 / R10 / R0.5 / Monster strategy code;
- backtests, factories, optimizers and stress results;
- golden/reference outputs;
- trading rules, execution constraints and strategy documents;
- strategy/audit workflows and tests.

No trading strategy or capital-allocation rule is active on this branch until a new specification is explicitly defined.

Pre-reset source history is preserved in branch `backup-pre-clean-reset-20260828`.
