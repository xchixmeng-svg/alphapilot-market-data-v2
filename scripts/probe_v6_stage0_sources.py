#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pre_oos_audit"
OUT.mkdir(exist_ok=True)

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Probe/1.7",
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
            row["json_summary"] = json_check(r.json())
        row["ok"] = True
    except Exception as e:
        row.update({
            "elapsed_seconds": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
        })
    print("[STAGE0 PROBE]", json.dumps(row, ensure_ascii=False), flush=True)
    return row


def twse_ind_summary(j):
    if not isinstance(j, dict):
        return {"type": type(j).__name__}
    tables = j.get("tables") or []
    rows = sum(len(t.get("data") or []) for t in tables if isinstance(t, dict))
    fields = [t.get("fields") for t in tables if isinstance(t, dict) and t.get("fields")]
    return {
        "stat": j.get("stat"),
        "tables": len(tables),
        "table_rows": rows,
        "nonempty_field_tables": len(fields),
    }


def dbnomics_summary(j):
    if not isinstance(j, dict):
        return {"type": type(j).__name__}
    docs = (((j.get("series") or {}).get("docs")) or [])
    first = docs[0] if docs else {}
    periods = first.get("period") or []
    values = first.get("value") or []
    return {
        "series_docs": len(docs),
        "series_code": first.get("series_code") or first.get("code"),
        "observations": min(len(periods), len(values)),
        "first_period": periods[0] if periods else None,
        "last_period": periods[-1] if periods else None,
    }


def tip_records_summary(j):
    if not isinstance(j, dict):
        return {"type": type(j).__name__}
    root = j.get("data") if isinstance(j.get("data"), dict) else {}
    datasets = root.get("datasets") if isinstance(root, dict) else []
    datasets = datasets if isinstance(datasets, list) else []
    summaries = []
    for ds in datasets[:8]:
        if not isinstance(ds, dict):
            continue
        points = ds.get("data")
        summaries.append({
            "value_type": ds.get("value_type"),
            "points": len(points) if isinstance(points, list) else None,
            "first": points[0] if isinstance(points, list) and points else None,
            "last": points[-1] if isinstance(points, list) and points else None,
        })
    return {
        "top_keys": sorted(j.keys()),
        "empty": j.get("empty"),
        "last_date": j.get("last_date"),
        "main_type": j.get("main_type"),
        "labels": len(root.get("labels") or []) if isinstance(root, dict) and isinstance(root.get("labels"), list) else None,
        "dataset_count": len(datasets),
        "datasets": summaries,
    }


def main():
    rows = []
    twse_url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"

    # Sparse historical samples first: prove whether the free official daily endpoint
    # still serves old industry-index dates before designing any checkpoint strategy.
    for d in ("20161230", "20171229", "20181228", "20191231", "20201231", "20260915"):
        rows.append(probe(
            f"TWSE_MI_INDEX_IND_{d}", twse_url,
            params={"response": "json", "date": d, "type": "IND"},
            json_check=twse_ind_summary,
        ))
        time.sleep(0.35)

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

    if os.getenv("V6_PROBE_FRED_PRIMARY", "0") == "1":
        rows.append(probe(
            "FRED_DFF_PRIMARY",
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "DFF"},
        ))

    for name, series in (
        ("DBNOMICS_FEDFUNDS_FALLBACK", "RIFSPFF_N.B"),
        ("DBNOMICS_US2Y_FALLBACK", "RIFLGFCY02_N.B"),
        ("DBNOMICS_US10Y_FALLBACK", "RIFLGFCY10_N.B"),
    ):
        rows.append(probe(
            name,
            f"https://api.db.nomics.world/v22/series/FED/H15/{series}",
            params={"observations": "1"},
            json_check=dbnomics_summary,
        ))

    # TIP CSV is verified as a recent-history source. This small request proves the
    # public contract without repeatedly re-discovering the Nuxt runtime host.
    rows.append(probe(
        "TIP_DOWNLOAD_T00_RECENT",
        "https://backend.taiwanindex.com.tw/api/download/history",
        params={"lang": "zh-tw", "code": "t00", "start": "2026-09-01", "end": "2026-09-05"},
    ))

    # TIP records API also enforces the recent-history window: retain explicit evidence.
    for code, start, end in (
        ("t00", "2016-01-01", "2016-12-31"),
        ("t02", "2016-01-01", "2016-12-31"),
        ("t07", "2016-01-01", "2016-12-31"),
        ("t07", "2026-01-01", "2026-09-15"),
    ):
        rows.append(probe(
            f"TIP_RECORDS_{code.upper()}_{start[:4]}",
            f"https://backend.taiwanindex.com.tw/api/indexes/{code}/records",
            params={"start": start, "end": end},
            json_check=tip_records_summary,
        ))

    historical_twse = [x for x in rows if x["name"].startswith("TWSE_MI_INDEX_IND_20") and not x["name"].endswith("20260915")]
    out = {
        "purpose": "PRE-OOS Stage 0 transport/source diagnosis only; not model evidence or fallback equivalence evidence",
        "all_ok": all(x["ok"] for x in rows),
        "historical_twse_sparse_samples_all_ok": bool(historical_twse) and all(x["ok"] and (x.get("json_summary") or {}).get("table_rows", 0) > 0 for x in historical_twse),
        "fred_primary_probe_enabled": os.getenv("V6_PROBE_FRED_PRIMARY", "0") == "1",
        "results": rows,
    }
    (OUT / "V6_STAGE0_SOURCE_PROBE.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
