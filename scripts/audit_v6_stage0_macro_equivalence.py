#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "history" / "v6-layered" / "0D_MACRO"
PROGRESS = ROOT / "data" / "history" / "v6-layered" / "progress"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS.mkdir(parents=True, exist_ok=True)

DBNOMICS_BASE = "https://api.db.nomics.world/v22/series/FED/H15"
DBNOMICS = {
    "fed_funds": "RIFSPFF_N.B",
    "us2y": "RIFLGFCY02_N.B",
    "us10y": "RIFLGFCY10_N.B",
}
START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-09-15")

# Independent official transports.
NYFED_EFFR = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
FED_DDP_OUTPUT = "https://www.federalreserve.gov/datadownload/Output.aspx"
FED_TREASURY_PACKAGE = "bf17364827e38702b42a58cf8eaa3f78"

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Macro-Audit/2.0",
    "Accept": "application/json,text/csv,text/plain,*/*",
})


def _get(url: str, *, params=None, timeout=(15, 60), tries=4) -> requests.Response:
    last = None
    for attempt in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt + 1 < tries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed {url}: {type(last).__name__}: {last}")


def dbnomics(code: str) -> pd.DataFrame:
    r = _get(f"{DBNOMICS_BASE}/{code}", params={"observations": "1"}, timeout=(15, 45))
    docs = (((r.json().get("series") or {}).get("docs")) or [])
    if len(docs) != 1:
        raise RuntimeError(f"DBnomics {code}: expected one document")
    d = docs[0]
    x = pd.DataFrame({
        "date": pd.to_datetime(d.get("period") or [], errors="coerce"),
        "primary": pd.to_numeric(d.get("value") or [], errors="coerce"),
    })
    return (
        x.dropna(subset=["date"])
         .query("@START <= date <= @END")
         .drop_duplicates("date", keep="last")
         .sort_values("date")
         .reset_index(drop=True)
    )


def nyfed_effr() -> pd.DataFrame:
    frames = []
    # Chunk by year so a large historical response cannot become a single transport blocker.
    for year in range(START.year, END.year + 1):
        lo = max(START, pd.Timestamp(f"{year}-01-01"))
        hi = min(END, pd.Timestamp(f"{year}-12-31"))
        r = _get(
            NYFED_EFFR,
            params={"startDate": lo.date().isoformat(), "endDate": hi.date().isoformat(), "type": "rate"},
            timeout=(15, 45),
        )
        obj = r.json()
        rows = obj.get("refRates") or obj.get("rates") or []
        recs = []
        for z in rows:
            if str(z.get("type", "EFFR")).upper() != "EFFR":
                continue
            recs.append({
                "date": pd.to_datetime(z.get("effectiveDate"), errors="coerce"),
                "fallback": pd.to_numeric(z.get("percentRate"), errors="coerce"),
            })
        if recs:
            frames.append(pd.DataFrame(recs))
    if not frames:
        raise RuntimeError("NY Fed EFFR API returned no historical observations")
    x = pd.concat(frames, ignore_index=True)
    return (
        x.dropna(subset=["date"])
         .query("@START <= date <= @END")
         .drop_duplicates("date", keep="last")
         .sort_values("date")
         .reset_index(drop=True)
    )


def fed_treasury_ddp() -> pd.DataFrame:
    r = _get(
        FED_DDP_OUTPUT,
        params={
            "filetype": "csv",
            "from": "",
            "label": "include",
            "lastobs": "",
            "layout": "seriescolumn",
            "rel": "H15",
            "series": FED_TREASURY_PACKAGE,
            "to": "",
            "type": "package",
        },
        timeout=(20, 90),
    )
    rows = list(csv.reader(io.StringIO(r.text)))
    target2 = "RIFLGFCY02_N.B"
    target10 = "RIFLGFCY10_N.B"
    header_i = None
    header = None
    for i, row in enumerate(rows):
        norm = [str(v).strip().replace("H15/H15/", "") for v in row]
        if target2 in norm and target10 in norm and norm and norm[0].lower() in {
            "time period", "date", "observation_date"
        }:
            header_i, header = i, norm
            break
    if header_i is None:
        # Some DDP exports put identifiers on the line before the "Time Period" row.
        for i, row in enumerate(rows):
            norm = [str(v).strip().replace("H15/H15/", "") for v in row]
            if target2 in norm and target10 in norm:
                for j in range(i + 1, min(i + 8, len(rows))):
                    probe = [str(v).strip().replace("H15/H15/", "") for v in rows[j]]
                    if probe and probe[0].lower() in {"time period", "date", "observation_date"}:
                        header_i = j
                        header = [probe[0]] + norm[1:]
                        break
            if header_i is not None:
                break
    if header_i is None or header is None:
        raise RuntimeError(f"Federal Reserve DDP CSV header not recognized; prefix={r.text[:300]!r}")

    try:
        i2 = header.index(target2)
        i10 = header.index(target10)
    except ValueError as e:
        raise RuntimeError(f"Federal Reserve DDP target columns missing: {header}") from e

    recs = []
    for row in rows[header_i + 1:]:
        if not row:
            continue
        d = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(d):
            continue
        def num(idx):
            if idx >= len(row):
                return float("nan")
            v = str(row[idx]).strip()
            if v in {"", "ND", "NA", "."}:
                return float("nan")
            return pd.to_numeric(v, errors="coerce")
        recs.append({"date": d, "us2y": num(i2), "us10y": num(i10)})
    if not recs:
        raise RuntimeError("Federal Reserve DDP CSV parsed zero observations")
    x = pd.DataFrame(recs)
    return (
        x.query("@START <= date <= @END")
         .drop_duplicates("date", keep="last")
         .sort_values("date")
         .reset_index(drop=True)
    )


_TREASURY = None


def fallback(name: str) -> tuple[pd.DataFrame, str]:
    global _TREASURY
    if name == "fed_funds":
        return nyfed_effr(), "New York Fed official EFFR API"
    if _TREASURY is None:
        _TREASURY = fed_treasury_ddp()
    col = "us2y" if name == "us2y" else "us10y"
    x = _TREASURY[["date", col]].rename(columns={col: "fallback"}).copy()
    return x, "Federal Reserve Board H.15 Data Download Program"


def audit_one(name: str) -> dict:
    a = dbnomics(DBNOMICS[name])
    b, source = fallback(name)
    m = a.merge(b, on="date", how="inner").dropna(subset=["primary", "fallback"])
    if m.empty:
        return {"series": name, "status": "FAIL", "reason": "no numeric overlap", "overlap_rows": 0}
    diff = (m["primary"] - m["fallback"]).abs()
    exact = float((diff <= 1e-12).mean())
    within_bp = float((diff <= 0.01 + 1e-12).mean())
    years = int(m["date"].dt.year.nunique())
    # Audit contract: >=250 numeric overlaps, >=3 calendar years, >=99.5% within 1bp.
    passed = len(m) >= 250 and years >= 3 and within_bp >= 0.995
    return {
        "series": name,
        "status": "PASS" if passed else "FAIL",
        "primary": f"DBnomics FED/H15/{DBNOMICS[name]}",
        "fallback": source,
        "overlap_rows": int(len(m)),
        "distinct_calendar_years": years,
        "exact_match_rate": exact,
        "within_1bp_rate": within_bp,
        "max_abs_diff": float(diff.max()),
        "first_overlap": m["date"].min().date().isoformat(),
        "last_overlap": m["date"].max().date().isoformat(),
    }


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    results = []
    for name in DBNOMICS:
        try:
            results.append(audit_one(name))
        except Exception as e:
            results.append({
                "series": name,
                "status": "BLOCKED_TRANSPORT",
                "reason": f"{type(e).__name__}: {e}",
            })

    passed = sum(r.get("status") == "PASS" for r in results)
    evidence = {
        "audit": "0D_fallback_equivalence",
        "generated_at_utc": generated,
        "series_total": len(results),
        "series_passed": passed,
        "results": results,
        "zero_fill_used": False,
        "formal_oos_opened": False,
        "publication_availability_lag_audit": "NOT_YET_PASS",
    }
    (OUT / "fallback_equivalence_audit.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    completed = 2 if passed == len(results) else 1
    unresolved = [r["series"] for r in results if r.get("status") != "PASS"]
    progress = {
        "lane": "0D_MACRO",
        "status": "PROVISIONAL",
        "current_completed": completed,
        "target_total": 3,
        "completion_pct": round(completed / 3 * 100, 2),
        "newly_completed_this_run": 1 if completed == 2 else 0,
        "remaining": 3 - completed,
        "current_blocker": (
            "publication/availability-lag audit remains"
            if completed == 2 else
            "fallback equivalence/source transport audit did not pass all macro series"
        ),
        "last_successful_unit": "fallback equivalence audit" if completed == 2 else "FED/H15 source build",
        "artifact_name": "alphapilot-v6-stage0-0D-macro-audit",
        "updated_at_utc": generated,
        "macro_series_total": len(results),
        "equivalence_audit_passed_series": passed,
        "publication_lag_audit_passed_series": 0,
        "unresolved_series": unresolved,
        "unresolved_gates": ["publication_availability_lag"] + ([] if passed == len(results) else ["fallback_equivalence"]),
    }
    (PROGRESS / "0D_MACRO.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[0D EQUIVALENCE RESULTS] " + json.dumps(evidence, ensure_ascii=False))
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    if passed != len(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
