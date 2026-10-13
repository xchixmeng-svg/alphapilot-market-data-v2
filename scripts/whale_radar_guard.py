#!/usr/bin/env python3
"""Safety guard for Whale Radar bootstrap and first-run semantics.

The first stored >10% shareholder snapshot is a baseline, not a newly observed
entry signal. Until we have an older snapshot to compare against, remove
`new_10pct_shareholder` events from the run output and ledger.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("data/whale_radar")
EVENT_TYPE = "new_10pct_shareholder"
CSV_FIELDS = [
    "event_id",
    "event_type",
    "stock_code",
    "company_name",
    "actor",
    "signal_score",
    "direction",
    "note",
]


def snapshot_days() -> list[Path]:
    base = ROOT / "snapshots"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def filter_event_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    kept = [x for x in data if isinstance(x, dict) and x.get("event_type") != EVENT_TYPE]
    path.write_text(json.dumps(kept, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return kept


def rewrite_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def clean_ledger(path: Path) -> int:
    if not path.exists():
        return 0
    kept_lines: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        if isinstance(obj, dict) and obj.get("event_type") == EVENT_TYPE:
            removed += 1
            continue
        kept_lines.append(line)
    suffix = "\n" if kept_lines else ""
    path.write_text("\n".join(kept_lines) + suffix, encoding="utf-8")
    return removed


def main() -> int:
    days = snapshot_days()
    if len(days) > 1:
        print(f"baseline guard: {len(days)} snapshot days, no bootstrap cleanup needed")
        return 0

    removed = clean_ledger(ROOT / "events.jsonl")
    filter_event_list(ROOT / "latest" / "events_seen_today.json")
    new_events = filter_event_list(ROOT / "latest" / "new_events.json")
    rewrite_csv(ROOT / "latest" / "new_events.csv", new_events)
    print(f"baseline guard: first snapshot only; removed {removed} false new-holder events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
