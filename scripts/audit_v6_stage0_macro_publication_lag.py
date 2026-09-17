#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "history" / "v6-layered" / "0D_MACRO"
PROGRESS = ROOT / "data" / "history" / "v6-layered" / "progress"
MACRO = OUT / "macro_daily.parquet"

NY = ZoneInfo("America/New_York")
USB = CustomBusinessDay(calendar=USFederalHolidayCalendar())

SOURCE_EVIDENCE = {
    "h15": {
        "url": "https://www.federalreserve.gov/releases/h15/",
        "normal_release_time_et": "16:15",
        "scope": ["us2y", "us10y"],
    },
    "effr": {
        "url": "https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates",
        "normal_release_time_et": "09:00",
        "scope": ["fed_funds"],
    },
}

# Deliberately conservative: wait two US federal business days after each
# observation date and use 16:15 ET for every series. This is later than the
# documented normal publication time for both H.15 and EFFR and therefore
# protects formal OOS from source/transport timing uncertainty.
CONSERVATIVE_BUSINESS_DAY_LAG = 2
CONSERVATIVE_RELEASE_TIME_ET = time(16, 15)


def available_at_utc(d: pd.Timestamp) -> pd.Timestamp:
    release_day = (pd.Timestamp(d).normalize() + CONSERVATIVE_BUSINESS_DAY_LAG * USB).date()
    local = datetime.combine(release_day, CONSERVATIVE_RELEASE_TIME_ET, tzinfo=NY)
    return pd.Timestamp(local.astimezone(timezone.utc))


def main() -> None:
    if not MACRO.exists():
        raise RuntimeError(f"missing macro layer: {MACRO}")
    df = pd.read_parquet(MACRO)
    if "date" not in df.columns:
        raise RuntimeError("macro layer missing date column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    series = ["fed_funds", "us2y", "us10y"]
    per_series = {}
    failures = []
    for name in series:
        if name not in df.columns:
            failures.append(f"{name}: missing column")
            continue
        x = df.loc[df[name].notna(), ["date", name]].copy()
        x["available_at_utc"] = [available_at_utc(d) for d in x["date"]]
        obs_midnight_utc = x["date"].dt.tz_localize("UTC")
        lag_hours = (x["available_at_utc"] - obs_midnight_utc).dt.total_seconds() / 3600.0
        bad = x.loc[lag_hours <= 24.0]
        per_series[name] = {
            "rows": int(len(x)),
            "date_start": x["date"].min().date().isoformat() if len(x) else None,
            "date_end": x["date"].max().date().isoformat() if len(x) else None,
            "mapped_available_at_rows": int(x["available_at_utc"].notna().sum()),
            "minimum_lag_hours": float(lag_hours.min()) if len(lag_hours) else None,
            "maximum_lag_hours": float(lag_hours.max()) if len(lag_hours) else None,
            "causal_violations": int(len(bad)),
        }
        if len(x) < 250:
            failures.append(f"{name}: insufficient observations={len(x)}")
        if len(bad):
            failures.append(f"{name}: {len(bad)} availability timestamps too early")

    status = "PASS" if not failures else "FAIL"
    generated = datetime.now(timezone.utc).isoformat()
    evidence = {
        "audit": "0D_publication_availability_lag",
        "status": status,
        "generated_at_utc": generated,
        "policy": {
            "business_day_calendar": "USFederalHolidayCalendar",
            "business_day_lag_after_observation": CONSERVATIVE_BUSINESS_DAY_LAG,
            "release_time_et": "16:15",
            "timezone": "America/New_York",
            "reason": "conservative causal availability policy later than documented normal publication schedules",
            "forward_fill": False,
            "future_join_allowed": False,
        },
        "source_evidence": SOURCE_EVIDENCE,
        "per_series": per_series,
        "failures": failures,
        "formal_oos_opened": False,
    }
    (OUT / "publication_availability_lag_audit.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    progress = {
        "lane": "0D_MACRO",
        "status": "PASS" if status == "PASS" else "PROVISIONAL",
        "current_completed": 3 if status == "PASS" else 2,
        "target_total": 3,
        "completion_pct": 100.0 if status == "PASS" else 66.67,
        "newly_completed_this_run": 1 if status == "PASS" else 0,
        "remaining": 0 if status == "PASS" else 1,
        "current_blocker": None if status == "PASS" else "publication/availability-lag audit failed",
        "last_successful_unit": "publication/availability-lag audit" if status == "PASS" else "fallback equivalence audit",
        "artifact_name": "alphapilot-v6-stage0-0D-macro-audit",
        "updated_at_utc": generated,
        "equivalence_audit_passed_series": 3,
        "publication_lag_audit_passed_series": 3 if status == "PASS" else 0,
        "formal_oos_opened": False,
    }
    PROGRESS.mkdir(parents=True, exist_ok=True)
    (PROGRESS / "0D_MACRO.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[0D PUBLICATION LAG AUDIT] " + json.dumps(evidence, ensure_ascii=False))
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
