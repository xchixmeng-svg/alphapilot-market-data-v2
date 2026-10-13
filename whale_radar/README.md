# AlphaPilot Whale Radar v0

Goal: build a reproducible tracker for **publicly disclosed** Taiwan-market large-holder / insider activity, then measure whether specific disclosed actors historically add predictive value.

This module intentionally does **not** claim access to anyone's private brokerage account. "No public disclosure" never means "no holding".

## v0 data sources

Official TWSE OpenAPI:

| Dataset | Endpoint | v0 use |
|---|---|---|
| Listed company >10% shareholders | `t187ap02_L` | detect newly appearing disclosed names |
| Directors/supervisors holdings | `t187ap11_L` | raw evidence archive |
| Daily insider transfer pre-declarations | `t187ap12_L` | risk / exit evidence |
| Listed company material information | `t187ap04_L` | filter Article 43-1 disclosures |

The first implementation uses only the Python standard library.

## Output

`data/whale_radar/`

- `snapshots/YYYY-MM-DD/*.json`: immutable daily source snapshots
- `latest/*.json`: latest copy of each source
- `events.jsonl`: deduplicated evidence ledger
- `latest/new_events.csv`: newly seen events in the latest run
- `latest/candidates.csv`: current evidence ranking

### Initial event weights

These are **research triage weights, not trading rules**:

- Article 43-1 disclosure: `+50`
- newly appearing >10% shareholder versus last stored snapshot: `+35`
- insider transfer pre-declaration: `-20`

A score above zero means "research this first", not "buy".

## Run

```bash
python scripts/whale_radar_collect.py --self-test
python scripts/whale_radar_collect.py --allow-partial
python scripts/whale_radar_rank.py
```

The GitHub Action runs after the Taiwan close on weekdays and can also be started manually.

## Data integrity rules

1. Keep original source row inside every event.
2. Store `first_seen_at`.
3. Deduplicate with a deterministic hash.
4. Do not fabricate missing actor names, percentages, dates or motives.
5. Provider failure is recorded as degraded/partial; other sources continue.
6. Never label a person as the buyer of a stock unless a public source identifies them.

## Next phases

### Phase 1 — broaden coverage
- Add OTC / TPEx equivalents.
- Add MOPS 5% / Article 43-1 historical backfill rather than relying only on current listed-company material messages.
- Add TDCC shareholding-distribution snapshots.

### Phase 2 — actor identity graph
Canonical actor ID, aliases, natural-person vs corporation, related investment vehicles, source confidence.

### Phase 3 — historical performance
For every qualifying event, attach market price as-of the first legally/publicly available timestamp and calculate T+5/10/20/40/60/120 returns, MFE and MAE. No look-ahead.

### Phase 4 — Person Alpha Score
Only after enough historical observations. Rank actors by out-of-sample performance, sample size, drawdown and repeatability. Do not crown a "guru" from one or two winners.

### Phase 5 — entry radar
Combine high-quality actor evidence with price not-yet-extended, liquidity, fundamentals and catalysts. This is where AlphaPilot can decide whether a disclosure is actionable.

## Definition of success

The project is useful only if historical, causal testing shows that the tracked public disclosures improve forward returns or risk-adjusted outcomes versus matched controls. Fame alone is not a signal.
