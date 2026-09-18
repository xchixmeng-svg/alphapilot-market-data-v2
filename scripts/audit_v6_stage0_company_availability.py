#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAYER = ROOT / "data" / "history" / "v6-layered" / "0C_COMPANY"
MANIFEST = ROOT / "data" / "history" / "v6-layered" / "manifests" / "0C_COMPANY.json"
AUDIT_DIR = ROOT / "data" / "history" / "v6-layered" / "audits"
PROGRESS = ROOT / "data" / "history" / "v6-layered" / "progress" / "0C_COMPANY.json"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TAIPEI = ZoneInfo("Asia/Taipei")
LAW_URL = "https://twse-regulation.twse.com.tw/TW/law/DOC01.aspx?FLCODE=FL007009&FLNO=36"
TWSE_VAL_URL = "https://eshop.twse.com.tw/zh/product/detail/8a82e9e697fc5f620198abeec9830097"
TPEX_VAL_URL = "https://www.tpex.org.tw/zh-tw/mainboard/trading/info/daily-pe.html"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def second_month_first(y: int, m: int) -> str:
    if m == 11:
        return f"{y+1:04d}-01-01"
    if m == 12:
        return f"{y+1:04d}-02-01"
    return f"{y:04d}-{m+2:02d}-01"


def main() -> int:
    if not MANIFEST.exists():
        raise RuntimeError("0C manifest missing; rebuild from immutable checkpoints first")
    mp = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rev_path = LAYER / "monthly_revenue_point_in_time.parquet"
    val_path = LAYER / "daily_valuation_point_in_time.parquet"
    if not rev_path.exists() or not val_path.exists():
        raise RuntimeError("0C PIT parquet missing")

    rev = pd.read_parquet(rev_path)
    val = pd.read_parquet(val_path)
    required_rev = {"period_year", "period_month", "code", "available_date"}
    required_val = {"date", "market", "code", "available_at_utc"}
    if not required_rev.issubset(rev.columns):
        raise RuntimeError(f"revenue columns missing: {sorted(required_rev-set(rev.columns))}")
    if not required_val.issubset(val.columns):
        raise RuntimeError(f"valuation columns missing: {sorted(required_val-set(val.columns))}")

    if "source_policy_available_date" not in rev.columns:
        rev["source_policy_available_date"] = rev["available_date"].astype(str)
    rev["available_date"] = [
        second_month_first(int(y), int(m))
        for y, m in zip(rev["period_year"], rev["period_month"])
    ]

    if "source_policy_available_at_utc" not in val.columns:
        val["source_policy_available_at_utc"] = val["available_at_utc"].astype(str)
    vd = pd.to_datetime(val["date"], errors="coerce")
    if vd.isna().any():
        raise RuntimeError("valuation contains invalid dates")
    safe_local = [
        datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=TAIPEI) + timedelta(days=1)
        for d in vd.dt.date
    ]
    val["available_at_utc"] = [
        x.astimezone(timezone.utc).isoformat() for x in safe_local
    ]

    rev_safe = pd.to_datetime(rev["available_date"], errors="coerce")
    period_end = pd.to_datetime(
        [f"{int(y):04d}-{int(m):02d}-01" for y, m in zip(rev["period_year"], rev["period_month"])],
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    rev_viol = int((rev_safe <= period_end).sum() + rev_safe.isna().sum())

    val_avail = pd.to_datetime(val["available_at_utc"], utc=True, errors="coerce")
    val_obs_end = pd.to_datetime(vd.dt.date.astype(str), utc=True) + pd.Timedelta(days=1)
    val_viol = int((val_avail < val_obs_end).sum() + val_avail.isna().sum())

    datasets = mp.get("datasets") or []
    revenue_units = int((datasets[0] if len(datasets) > 0 else {}).get("checkpoint_units", 0))
    complete_units = int((datasets[1] if len(datasets) > 1 else {}).get("checkpoint_units", 0))
    target_units = int((datasets[1] if len(datasets) > 1 else {}).get("target_units", 0))
    structural_complete = (
        revenue_units >= 256 and target_units > 0 and complete_units == target_units
        and len(rev) >= 100000 and len(val) > 100000
    )
    audit_pass = structural_complete and rev_viol == 0 and val_viol == 0

    rev.to_parquet(rev_path, index=False)
    val.to_parquet(val_path, index=False)

    evidence = {
        "lane": "0C_COMPANY",
        "audit": "formal_oos_publication_availability",
        "status": "PASS" if audit_pass else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structural_complete": structural_complete,
        "revenue_checkpoint_units": revenue_units,
        "valuation_checkpoint_units": complete_units,
        "valuation_target_units": target_units,
        "monthly_revenue_rows": int(len(rev)),
        "valuation_rows": int(len(val)),
        "monthly_revenue_causal_violations": rev_viol,
        "valuation_causal_violations": val_viol,
        "evidence": [
            {
                "source": "Securities and Exchange Act Article 36",
                "url": LAW_URL,
                "documented_rule": "monthly operating status is announced/filed by the 10th of the following month",
                "formal_oos_policy": "delay monthly revenue to the first calendar day of the second month after the revenue period",
                "reason": "intentionally later than the statutory deadline; avoids holiday/deadline ambiguity"
            },
            {
                "source": "TWSE Data E-Shop BWIBBU_CLO",
                "url": TWSE_VAL_URL,
                "documented_rule": "TWSE daily PE/dividend-yield/PB file is produced once each trading day at 18:00",
                "formal_oos_policy": "delay all daily valuation data to 23:59:59 Asia/Taipei on the next calendar day"
            },
            {
                "source": "TPEx after-trading daily PE/dividend-yield/PB query",
                "url": TPEX_VAL_URL,
                "documented_rule": "official after-trading daily valuation dataset; exact publication clock time not relied upon",
                "formal_oos_policy": "same delayed-next-calendar-day policy as TWSE to avoid assuming a TPEx clock time"
            }
        ],
        "policy_is_deliberately_conservative": True,
        "formal_oos_opened": False
    }
    ep = AUDIT_DIR / "0C_COMPANY_AVAILABILITY.json"
    ep.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mp["status"] = "FROZEN_PASS" if audit_pass else "PROVISIONAL"
    mp["schema_version"] = "v3-conservative-formal-oos-availability"
    mp["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    mp["file_sha256"] = {
        rev_path.name: sha256(rev_path),
        val_path.name: sha256(val_path)
    }
    mp.setdefault("pit_rules", {})
    mp["pit_rules"].update({
        "monthly_revenue_formal_oos_available_date_policy": "first day of second month after revenue period",
        "daily_valuation_formal_oos_available_at_policy": "next calendar day 23:59:59 Asia/Taipei",
        "publication_availability_lag_audit_complete": audit_pass,
        "monthly_revenue_causal_violations": rev_viol,
        "valuation_causal_violations": val_viol,
        "formal_oos_eligible": audit_pass,
        "availability_evidence_path": str(ep.relative_to(ROOT))
    })
    MANIFEST.write_text(json.dumps(mp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pp = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {"lane": "0C_COMPANY"}
    pp.update({
        "status": "FROZEN_PASS" if audit_pass else "PROVISIONAL",
        "current_completed": 4 if audit_pass else 3,
        "target_total": 4,
        "completion_pct": 100.0 if audit_pass else 75.0,
        "remaining": 0 if audit_pass else 1,
        "current_blocker": None if audit_pass else "formal OOS availability audit failed",
        "formal_oos_opened": False,
        "availability_audit_path": str(ep.relative_to(ROOT)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat()
    })
    PROGRESS.write_text(json.dumps(pp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    return 0 if audit_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
