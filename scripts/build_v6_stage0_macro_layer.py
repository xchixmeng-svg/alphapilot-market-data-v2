#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LAYER_DIR = ROOT / "data" / "history" / "v6-layered" / "0D_MACRO"
MANIFEST_DIR = ROOT / "data" / "history" / "v6-layered" / "manifests"
LAYER_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-09-15")
DBNOMICS_BASE = "https://api.db.nomics.world/v22/series/FED/H15"
SERIES = {
    "fed_funds": "RIFSPFF_N.B",
    "us2y": "RIFLGFCY02_N.B",
    "us10y": "RIFLGFCY10_N.B",
}

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Macro/1.0",
    "Accept": "application/json,*/*",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generation_commit() -> str:
    env = os.getenv("GITHUB_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def fetch_series(name: str, code: str) -> pd.DataFrame:
    url = f"{DBNOMICS_BASE}/{code}"
    r = S.get(url, params={"observations": "1"}, timeout=30)
    r.raise_for_status()
    obj = r.json()
    docs = (((obj.get("series") or {}).get("docs")) or [])
    if len(docs) != 1:
        raise RuntimeError(f"{name}: expected one series doc, got {len(docs)}")
    doc = docs[0]
    periods = doc.get("period") or []
    values = doc.get("value") or []
    if len(periods) != len(values) or not periods:
        raise RuntimeError(f"{name}: invalid period/value lengths {len(periods)}/{len(values)}")
    df = pd.DataFrame({"date": pd.to_datetime(periods, errors="coerce"), name: pd.to_numeric(values, errors="coerce")})
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    if df.empty:
        raise RuntimeError(f"{name}: empty after date filter")
    return df.reset_index(drop=True)


def main():
    frames = {name: fetch_series(name, code) for name, code in SERIES.items()}
    merged = None
    for name in SERIES:
        merged = frames[name] if merged is None else merged.merge(frames[name], on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["us10y2y"] = merged["us10y"] - merged["us2y"]

    # Deliberately no forward-fill here. Stage 0F performs causal as-of joins later.
    out_path = LAYER_DIR / "macro_daily.parquet"
    merged.to_parquet(out_path, index=False)

    missingness = {
        c: {
            "missing_rows": int(merged[c].isna().sum()),
            "missing_rate": float(merged[c].isna().mean()),
        }
        for c in ["fed_funds", "us2y", "us10y", "us10y2y"]
    }

    manifest = {
        "layer_id": "0D_MACRO",
        "status": "PROVISIONAL",
        "schema_version": "v1",
        "generation_commit": generation_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_lineage": [
            {
                "transport": "DBnomics API",
                "provider": "FED",
                "dataset": "H15",
                "upstream": "Federal Reserve Board H.15 series mirrored by DBnomics",
                "url_pattern": DBNOMICS_BASE + "/{series_code}",
                "formal_status": "PROVISIONAL_UNTIL_FALLBACK_EQUIVALENCE_PASS",
            }
        ],
        "datasets": [
            {"name": name, "series_code": code, "time_semantics": "decision_date_daily"}
            for name, code in SERIES.items()
        ] + [
            {
                "name": "us10y2y",
                "derived_from": ["us10y", "us2y"],
                "formula": "us10y - us2y on same observation date only",
                "time_semantics": "decision_date_daily",
            }
        ],
        "date_start": merged["date"].min().date().isoformat(),
        "date_end": merged["date"].max().date().isoformat(),
        "row_count": int(len(merged)),
        "file_sha256": {out_path.name: sha256(out_path)},
        "missingness": missingness,
        "pit_rules": {
            "raw_observation_dates_preserved": True,
            "forward_fill_in_layer_builder": False,
            "stage0f_join_rule": "available_at <= decision_time; causal as-of join only after availability-lag audit",
            "formal_oos_eligible": False,
            "reason_not_yet_eligible": "fallback equivalence and publication/availability-lag evidence not yet PASS",
        },
        "fallback_sources": [
            {
                "name": name,
                "mirror_series_code": code,
                "equivalence_status": "PROVISIONAL",
                "required_before_freeze": True,
            }
            for name, code in SERIES.items()
        ],
    }

    manifest_path = MANIFEST_DIR / "0D_MACRO.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "layer": "0D_MACRO",
        "status": manifest["status"],
        "rows": manifest["row_count"],
        "date_start": manifest["date_start"],
        "date_end": manifest["date_end"],
        "sha256": manifest["file_sha256"],
        "formal_oos_eligible": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
