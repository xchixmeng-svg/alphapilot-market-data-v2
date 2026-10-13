#!/usr/bin/env python3
"""Build a simple evidence-based Whale Radar candidate table from the event ledger."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TZ_TAIPEI = timezone(timedelta(hours=8))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            pass
    return rows


def build_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        code = str(e.get("stock_code") or "").strip()
        if code:
            by_code[code].append(e)

    out = []
    for code, items in by_code.items():
        score = sum(int(x.get("signal_score") or 0) for x in items)
        company = next((str(x.get("company_name") or "") for x in items if x.get("company_name")), "")
        actors = sorted({str(x.get("actor") or "").strip() for x in items if x.get("actor")})
        types = sorted({str(x.get("event_type") or "") for x in items if x.get("event_type")})
        positive = sum(1 for x in items if int(x.get("signal_score") or 0) > 0)
        negative = sum(1 for x in items if int(x.get("signal_score") or 0) < 0)
        out.append(
            {
                "stock_code": code,
                "company_name": company,
                "whale_score_raw": score,
                "positive_evidence": positive,
                "risk_evidence": negative,
                "actors": " | ".join(actors),
                "event_types": " | ".join(types),
                "evidence_count": len(items),
                "status": (
                    "HIGH_PRIORITY_RESEARCH"
                    if score >= 50
                    else "WATCH"
                    if score > 0
                    else "RISK_WATCH"
                    if score < 0
                    else "NEUTRAL"
                ),
            }
        )
    out.sort(key=lambda x: (x["whale_score_raw"], x["positive_evidence"]), reverse=True)
    return out


def main() -> int:
    root = Path("data/whale_radar")
    events = load_jsonl(root / "events.jsonl")
    candidates = build_candidates(events)
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(TZ_TAIPEI).isoformat(timespec="seconds"),
        "important": (
            "This score ranks public disclosure evidence only. It is not a buy signal "
            "and does not prove a private brokerage position."
        ),
        "candidates": candidates,
    }
    (latest / "candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = [
        "stock_code",
        "company_name",
        "whale_score_raw",
        "status",
        "positive_evidence",
        "risk_evidence",
        "evidence_count",
        "actors",
        "event_types",
    ]
    with (latest / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in candidates:
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
