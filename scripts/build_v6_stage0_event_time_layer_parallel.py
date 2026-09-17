#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "data" / "history" / "v6-layered" / "0A_MARKET" / "market_daily.parquet"
LAYER = ROOT / "data" / "history" / "v6-layered" / "0E_EVENT_TIME"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
PROGRESS_DIR = ROOT / "data" / "history" / "v6-layered" / "progress"
LAYER.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def commit() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def main() -> None:
    if not MARKET.exists():
        raise RuntimeError("0A market calendar evidence missing; build 0A first")
    m = pd.read_parquet(MARKET)
    if "date" not in m.columns:
        raise RuntimeError("0A market layer has no date column")
    d = pd.to_datetime(m["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(d) < 2000:
        raise RuntimeError(f"market decision calendar too short: {len(d)}")

    # Raw deterministic calendar context only. No hand-coded seasonal BUY/SELL rules.
    out = pd.DataFrame({"date": d})
    out["day_of_week"] = out["date"].dt.dayofweek.astype("int16")
    out["month"] = out["date"].dt.month.astype("int16")
    out["quarter"] = out["date"].dt.quarter.astype("int16")
    out["day_of_month"] = out["date"].dt.day.astype("int16")
    out["iso_week"] = out["date"].dt.isocalendar().week.astype("int16")
    out_path = LAYER / "time_context_daily.parquet"
    out.to_parquet(out_path, index=False)

    generated = datetime.now(timezone.utc).isoformat()
    inventory = {
        "verified_historical_event_sources": [],
        "live_only_event_sources": [],
        "rejected_or_unverified": [],
        "status": "NO_HISTORICAL_EVENT_SOURCE_FROZEN_YET",
        "rule": "Only sources with reconstructible published_at/available_at may enter formal historical OOS. Otherwise keep live-only or reject."
    }
    inv_path = LAYER / "event_source_inventory.json"
    inv_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "layer_id": "0E_EVENT_TIME",
        "status": "PROVISIONAL",
        "schema_version": "v1-parallel",
        "generation_commit": commit(),
        "generated_at_utc": generated,
        "source_lineage": [
            {"dataset": "decision_calendar", "source": "0A_MARKET observed Taiwan trading dates"},
            {"dataset": "historical_events", "source": None, "status": "NOT_YET_VERIFIED"}
        ],
        "datasets": [
            {"name": "time_context_daily", "time_semantics": "decision_date_daily", "rows": int(len(out)), "manual_seasonal_rules": False},
            {"name": "historical_events", "time_semantics": "event_timestamp_asof", "rows": 0, "formal_status": "NOT_STARTED"}
        ],
        "date_start": out["date"].min().date().isoformat(),
        "date_end": out["date"].max().date().isoformat(),
        "row_count": int(len(out)),
        "file_sha256": {out_path.name: sha256(out_path), inv_path.name: sha256(inv_path)},
        "missingness": {"historical_event_rows": 0},
        "pit_rules": {
            "calendar_features_known_on_decision_date": True,
            "hand_coded_seasonal_trading_rules_allowed": False,
            "historical_event_requires_published_at_or_available_at": True,
            "historical_event_source_frozen": False,
            "formal_oos_eligible": False
        }
    }
    (MANIFEST_DIR / "0E_EVENT_TIME.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    progress = {
        "lane": "0E_EVENT_TIME",
        "status": "PROVISIONAL",
        "current_completed": 1,
        "target_total": 2,
        "completion_pct": 50.0,
        "newly_completed_this_run": 1,
        "remaining": 1,
        "current_blocker": "historical event/news source with reconstructible published_at/available_at has not yet passed source/PIT audit",
        "last_successful_unit": "raw deterministic time context",
        "artifact_name": "alphapilot-v6-stage0-0E-event-time",
        "updated_at_utc": generated
    }
    (PROGRESS_DIR / "0E_EVENT_TIME.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
