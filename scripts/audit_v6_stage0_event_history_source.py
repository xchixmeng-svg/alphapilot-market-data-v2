#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "history" / "v6-layered" / "0E_EVENT_TIME"
PROGRESS = ROOT / "data" / "history" / "v6-layered" / "progress"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS.mkdir(parents=True, exist_ok=True)

URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"
SAMPLES = [
    {"code": "2330", "roc_year": 105, "label": "2016"},
    {"code": "2330", "roc_year": 109, "label": "2020"},
    {"code": "2330", "roc_year": 113, "label": "2024"},
]
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Event-PIT-Audit/1.0",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Referer": "https://mopsov.twse.com.tw/mops/web/t05st01",
})


def fetch_history(code: str, roc_year: int) -> str:
    r = S.post(
        URL,
        data={
            "encodeURIComponent": "1",
            "firstin": "1",
            "TYPEK": "all",
            "co_id": code,
            "year": str(roc_year),
            "month": "all",
            "b_date": "",
            "e_date": "",
        },
        timeout=(15, 60),
    )
    r.raise_for_status()
    text = r.text
    if len(text) < 500:
        raise RuntimeError(f"suspiciously short MOPS response bytes={len(text)}")
    if "查詢過於頻繁" in text or "Service Unavailable" in text:
        raise RuntimeError("MOPS throttled historical-event query")
    return text


def _flat(c) -> str:
    if isinstance(c, tuple):
        return "|".join(str(v).strip() for v in c if str(v).strip() not in ("", "nan"))
    return str(c).strip()


def _roc_date_to_ad(s: str):
    s = str(s).strip().replace("-", "/")
    m = re.search(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", s)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return datetime(y + 1911, mo, d)


def extract_timestamp_evidence(html: str) -> dict:
    markers = {
        "has_speech_date_marker": "發言日期" in html,
        "has_speech_time_marker": "發言時間" in html,
        "has_fact_date_marker": "事實發生日" in html,
    }
    valid = []
    table_rows = 0
    tables_with_timestamp_schema = 0
    parse_errors = []
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as e:
        tables = []
        parse_errors.append(f"read_html={type(e).__name__}:{e}")

    for df in tables:
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [_flat(x) for x in df.columns]
        cols = list(df.columns)
        date_col = next((x for x in cols if "發言日期" in x), None)
        time_col = next((x for x in cols if "發言時間" in x), None)
        if date_col is None or time_col is None:
            continue
        tables_with_timestamp_schema += 1
        table_rows += len(df)
        for _, row in df.iterrows():
            dd = _roc_date_to_ad(row.get(date_col))
            tt = str(row.get(time_col, "")).strip()
            mt = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", tt)
            if dd is None or mt is None:
                continue
            h, mi, sec = map(int, mt.groups())
            try:
                valid.append(dd.replace(hour=h, minute=mi, second=sec).isoformat())
            except Exception:
                pass

    return {
        **markers,
        "tables_with_timestamp_schema": tables_with_timestamp_schema,
        "timestamp_table_rows": table_rows,
        "valid_timestamp_pairs": len(valid),
        "first_valid_taipei_local": valid[0] if valid else None,
        "parse_errors": parse_errors,
        "note": "MOPS 發言日期 + 發言時間 are Taiwan local disclosure timestamps; 事實發生日 is not used as availability.",
    }

def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    results = []
    for s in SAMPLES:
        try:
            html = fetch_history(s["code"], s["roc_year"])
            ev = extract_timestamp_evidence(html)
            passed = (
                ev["has_speech_date_marker"]
                and ev["has_speech_time_marker"]
                and ev["tables_with_timestamp_schema"] > 0
                and ev["valid_timestamp_pairs"] > 0
            )
            results.append({
                **s,
                "status": "PASS" if passed else "FAIL",
                "response_chars": len(html),
                **ev,
            })
        except Exception as e:
            results.append({
                **s,
                "status": "BLOCKED_TRANSPORT",
                "reason": f"{type(e).__name__}: {e}",
            })

    passed = sum(r.get("status") == "PASS" for r in results)
    audit_pass = passed == len(SAMPLES)
    evidence = {
        "audit": "0E_MOPS_historical_material_information_PIT_source",
        "generated_at_utc": generated,
        "source": URL,
        "source_owner": "MOPS/TWSE official disclosure system",
        "samples_total": len(SAMPLES),
        "samples_passed": passed,
        "results": results,
        "pit_semantics": {
            "available_at_source_fields": ["發言日期", "發言時間"],
            "fact_date_not_used_as_availability": True,
            "timezone": "Asia/Taipei",
            "future_join_allowed": False,
        },
        "status": "PASS" if audit_pass else "FAIL",
        "formal_oos_opened": False,
    }
    (OUT / "historical_event_source_pit_audit.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Three honest gates: time context, historical source/PIT audit, historical
    # event ingestion/coverage. Do not call the layer finished after source audit alone.
    progress = {
        "lane": "0E_EVENT_TIME",
        "status": "PROVISIONAL",
        "current_completed": 2 if audit_pass else 1,
        "target_total": 3,
        "completion_pct": 66.67 if audit_pass else 33.33,
        "newly_completed_this_run": 1 if audit_pass else 0,
        "remaining": 1 if audit_pass else 2,
        "current_blocker": (
            "historical material-information ingestion/coverage remains"
            if audit_pass else
            "MOPS historical material-information source/PIT audit did not pass"
        ),
        "last_successful_unit": (
            "MOPS historical material-information PIT source audit"
            if audit_pass else "raw deterministic time context"
        ),
        "artifact_name": "alphapilot-v6-stage0-0E-event-time-audit",
        "updated_at_utc": generated,
        "historical_source_audit_pass": audit_pass,
        "formal_oos_opened": False,
    }
    (PROGRESS / "0E_EVENT_TIME.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[0E HISTORICAL SOURCE AUDIT] " + json.dumps(evidence, ensure_ascii=False))
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    if not audit_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
