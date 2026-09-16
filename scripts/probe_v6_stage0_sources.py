#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pre_oos_audit"
OUT.mkdir(exist_ok=True)

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Probe/1.0",
    "Accept": "application/json,text/csv,text/html,*/*",
})


def probe(name, url, params=None, timeout=20, json_check=None):
    t0 = time.time()
    row = {"name": name, "url": url, "ok": False}
    try:
        r = S.get(url, params=params, timeout=timeout)
        row.update({
            "status_code": r.status_code,
            "elapsed_seconds": round(time.time() - t0, 3),
            "content_type": r.headers.get("content-type", ""),
            "bytes": len(r.content),
            "final_url": r.url,
        })
        r.raise_for_status()
        if json_check is not None:
            payload = r.json()
            row["json_summary"] = json_check(payload)
        row["ok"] = True
    except Exception as e:
        row.update({
            "elapsed_seconds": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
        })
    print("[STAGE0 PROBE]", json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main():
    rows = []
    rows.append(probe(
        "TWSE_MI_INDEX_IND_20260915",
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
        params={"response": "json", "date": "20260915", "type": "IND"},
        json_check=lambda j: {
            "stat": j.get("stat"),
            "tables": len(j.get("tables") or []),
            "table_rows": sum(len(t.get("data") or []) for t in (j.get("tables") or [])),
        },
    ))
    rows.append(probe(
        "TWSE_OPENAPI_MI_INDEX_LATEST",
        "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
        json_check=lambda j: {"rows": len(j) if isinstance(j, list) else None},
    ))
    rows.append(probe(
        "TWSE_OPENAPI_HOLIDAY_SCHEDULE",
        "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule",
        json_check=lambda j: {"rows": len(j) if isinstance(j, list) else None},
    ))
    rows.append(probe(
        "FRED_DFF",
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": "DFF"},
    ))
    rows.append(probe(
        "TIP_HISTORY_PAGE",
        "https://taiwanindex.com.tw/indexes/t00/history",
    ))

    out = {
        "purpose": "PRE-OOS Stage 0 transport/source diagnosis only; not model evidence",
        "all_ok": all(x["ok"] for x in rows),
        "results": rows,
    }
    (OUT / "V6_STAGE0_SOURCE_PROBE.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
