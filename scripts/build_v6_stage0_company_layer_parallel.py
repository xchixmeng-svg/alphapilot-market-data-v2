#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_v6_context_layers_resumable as ctx

ROOT = Path(__file__).resolve().parents[1]
LAYER = ROOT / "data" / "history" / "v6-layered" / "0C_COMPANY"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
PROGRESS_DIR = ROOT / "data" / "history" / "v6-layered" / "progress"
LAYER.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

HIST = ROOT / "data" / "history" / "2020-2025"
AUDIT = HIST / "history_audit.json"


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
    generated = datetime.now(timezone.utc).isoformat()
    if not AUDIT.exists():
        raise RuntimeError("missing immutable 2020-2025 history audit")
    hist = json.loads(AUDIT.read_text(encoding="utf-8"))
    if hist.get("status") != "PASS":
        raise RuntimeError(f"history audit not PASS: {hist}")

    # Local immutable company-level market inputs: no network and no duplication.
    ohlcv_files = [HIST / f"ohlcv_{y}.parquet" for y in range(2020, 2026)]
    inst_file = HIST / "institutional_2020_2025.parquet"
    missing = [str(p) for p in [*ohlcv_files, inst_file] if not p.exists()]
    if missing:
        raise RuntimeError(f"missing immutable company inputs: {missing}")

    # Current company identity/industry snapshot is LIVE-ONLY; never used as historical OOS feature.
    snap_path, snap_rows = ctx.build_company_snapshot_resumable()

    # Historical monthly revenue is checkpointed per (year, month, market) under a dedicated cache.
    before_units = len(list(ctx.REV_CACHE.glob("*.json.gz")))
    rev_path, rev_fail, rev_rows = ctx.base.build_revenue()
    after_units = len(list(ctx.REV_CACHE.glob("*.json.gz")))
    newly = max(0, after_units - before_units)

    # Copy only compact 0C outputs into the layer artifact; immutable price/flow inputs stay in-place.
    rev_df = pd.read_csv(rev_path, compression="gzip")
    rev_out = LAYER / "monthly_revenue_point_in_time.parquet"
    rev_df.to_parquet(rev_out, index=False)
    snap_df = pd.read_csv(snap_path, compression="gzip")
    snap_out = LAYER / "company_identity_live_only.parquet"
    snap_df.to_parquet(snap_out, index=False)

    # Revenue target units are month x market from 2016 through 2026-08 = 256 checkpointable units.
    target_revenue_units = 256
    completed_revenue_units = min(after_units, target_revenue_units)
    status = "PROVISIONAL" if completed_revenue_units == target_revenue_units and not rev_fail else "BUILDING"
    blocker = (
        "daily valuation PIT and publication/availability-lag audit remain; live company identity is excluded from formal OOS"
        if status == "PROVISIONAL" else
        f"monthly revenue checkpoints incomplete/failures={len(rev_fail)}; daily valuation PIT and lag audit remain"
    )

    manifest = {
        "layer_id": "0C_COMPANY",
        "status": status,
        "schema_version": "v1-parallel",
        "generation_commit": commit(),
        "generated_at_utc": generated,
        "source_lineage": [
            {"dataset": "company_ohlcv", "source": "immutable repo history", "path": "data/history/2020-2025/ohlcv_YYYY.parquet"},
            {"dataset": "institutional", "source": "immutable repo history", "path": "data/history/2020-2025/institutional_2020_2025.parquet"},
            {"dataset": "monthly_revenue", "source": "MOPS official historical monthly revenue", "transport": "public HTML", "checkpoint_unit": "year-month-market"},
            {"dataset": "company_identity", "source": "MOPS open data current snapshot", "formal_oos_use": False},
        ],
        "datasets": [
            {"name": "company_ohlcv_2020_2025", "rows": int(hist.get("ohlcv_rows", 0)), "time_semantics": "decision_date_daily", "formal_status": "PASS_INPUT"},
            {"name": "institutional_2020_2025", "rows": int(hist.get("institutional_rows", 0)), "time_semantics": "decision_date_daily", "formal_status": "PASS_INPUT"},
            {"name": "monthly_revenue_point_in_time", "rows": int(rev_rows), "time_semantics": "periodic_release_asof", "checkpoint_units": completed_revenue_units, "target_units": target_revenue_units},
            {"name": "company_identity_live_only", "rows": int(snap_rows), "time_semantics": "static_live_only", "formal_oos_use": False},
        ],
        "date_start": str(rev_df["available_date"].min()) if not rev_df.empty else None,
        "date_end": str(rev_df["available_date"].max()) if not rev_df.empty else None,
        "row_count": int(rev_rows),
        "file_sha256": {rev_out.name: sha256(rev_out), snap_out.name: sha256(snap_out)},
        "missingness": {"revenue_failures_this_run": len(rev_fail)},
        "pit_rules": {
            "monthly_revenue_has_available_date": True,
            "current_company_identity_historical_oos_allowed": False,
            "daily_valuation_pit_complete": False,
            "publication_availability_lag_audit_complete": False,
            "formal_oos_eligible": False,
        },
    }
    (MANIFEST_DIR / "0C_COMPANY.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    progress = {
        "lane": "0C_COMPANY",
        "status": status,
        "current_completed": completed_revenue_units,
        "target_total": target_revenue_units,
        "completion_pct": round(completed_revenue_units / target_revenue_units * 100, 2),
        "newly_completed_this_run": newly,
        "remaining": max(0, target_revenue_units - completed_revenue_units),
        "current_blocker": blocker,
        "last_successful_unit": max((p.stem.replace(".json", "") for p in ctx.REV_CACHE.glob("*.json.gz")), default=None),
        "artifact_name": "alphapilot-v6-stage0-0C-company",
        "updated_at_utc": generated,
        "next_stage": "daily valuation PIT after/without competing with 0B TWSE throttle",
    }
    (PROGRESS_DIR / "0C_COMPANY.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
