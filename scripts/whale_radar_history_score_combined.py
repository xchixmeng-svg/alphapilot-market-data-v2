#!/usr/bin/env python3
"""Score Whale Radar live + historical bullish disclosure events together."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from whale_radar_history_score import (
    aggregate,
    evaluate_event,
    load_events,
    load_price_history,
    self_test as base_self_test,
    write_csv,
)


def dedup_events(events: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for ev in events:
        key = str(ev.get("event_id") or "")
        if not key:
            key = "|".join(str(ev.get(k, "")) for k in ("event_type", "stock_code", "actor", "event_date", "first_seen_at"))
        out[key] = ev
    return list(out.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/whale_radar")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if args.self_test:
        base_self_test(root)
        print("combined-history-score self-test: OK")
        return 0

    events = dedup_events(
        load_events(root / "events.jsonl")
        + load_events(root / "historical_sec43_events.jsonl")
    )
    codes = {str(x.get("stock_code", "")) for x in events if x.get("stock_code")}
    prices = load_price_history(Path(args.data_root), codes)
    event_rows = [evaluate_event(ev, prices.get(str(ev.get("stock_code", "")), [])) for ev in events]
    people = aggregate(event_rows)

    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "event_outcomes.json").write_text(json.dumps(event_rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (latest / "person_alpha_scores.json").write_text(json.dumps(people, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(latest / "event_outcomes.csv", event_rows)
    write_csv(latest / "person_alpha_scores.csv", people)

    ok = sum(1 for x in event_rows if x.get("status") == "OK")
    qualified = sum(1 for x in people if x.get("status") == "QUALIFIED")
    print(
        f"combined history score: {len(events)} bullish events; {len(people)} named actors; "
        f"repo price codes={len(prices)}; scored_events={ok}; qualified_actors={qualified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
