#!/usr/bin/env python3
from __future__ import annotations

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
# Independent public fallback transport. FRED identifies these as H.15/Board of Governors series.
FRED = {"fed_funds": "DFF", "us2y": "DGS2", "us10y": "DGS10"}
START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-09-15")
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Macro-Audit/1.1",
    "Accept": "text/csv,application/json,text/plain,*/*",
})
def dbnomics(code: str) -> pd.DataFrame:
    r = S.get(f"{DBNOMICS_BASE}/{code}", params={"observations": "1"}, timeout=30)
    r.raise_for_status()
    docs = (((r.json().get("series") or {}).get("docs")) or [])
    if len(docs) != 1:
        raise RuntimeError(f"DBnomics {code}: expected one document")
    d = docs[0]
    x = pd.DataFrame({"date": pd.to_datetime(d.get("period") or [], errors="coerce"), "primary": pd.to_numeric(d.get("value") or [], errors="coerce")})
    return x.dropna(subset=["date"]).query("@START <= date <= @END").drop_duplicates("date", keep="last")


def _fred_chunk(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from io import StringIO
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    last = None
    for attempt in range(4):
        try:
            r = S.get(
                url,
                params={"id": code, "cosd": start.date().isoformat(), "coed": end.date().isoformat()},
                timeout=(15, 45),
            )
            r.raise_for_status()
            if "DATE" not in r.text[:200].upper():
                raise RuntimeError(f"unexpected FRED CSV prefix={r.text[:80]!r}")
            x = pd.read_csv(StringIO(r.text))
            if x.shape[1] < 2:
                raise RuntimeError(f"invalid CSV columns={x.columns.tolist()}")
            x = x.iloc[:, :2].copy()
            x.columns = ["date", "fallback"]
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x["fallback"] = pd.to_numeric(x["fallback"], errors="coerce")
            return x.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        except Exception as e:
            last = e
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"FRED {code} transport failed for {start.date()}..{end.date()} after retries: "
        f"{type(last).__name__}: {last}"
    )


def fred(code: str) -> pd.DataFrame:
    # FRED's graph endpoint intermittently times out on decade-long GitHub-hosted
    # requests. Fetch independent year chunks so a transport stall cannot be
    # misclassified as source inequivalence.
    frames = []
    for year in range(START.year, END.year + 1):
        lo = max(START, pd.Timestamp(f"{year}-01-01"))
        hi = min(END, pd.Timestamp(f"{year}-12-31"))
        frames.append(_fred_chunk(code, lo, hi))
    x = pd.concat(frames, ignore_index=True)
    return (
        x.dropna(subset=["date"])
         .query("@START <= date <= @END")
         .drop_duplicates("date", keep="last")
         .sort_values("date")
         .reset_index(drop=True)
    )


def audit_one(name: str) -> dict:
    a = dbnomics(DBNOMICS[name])
    b = fred(FRED[name])
    m = a.merge(b, on="date", how="inner").dropna(subset=["primary", "fallback"])
    if m.empty:
        return {"series": name, "status": "FAIL", "reason": "no numeric overlap", "overlap_rows": 0}
    diff = (m["primary"] - m["fallback"]).abs()
    exact = float((diff <= 1e-12).mean())
    within_bp = float((diff <= 0.01).mean())
    # 1bp tolerance permits representation/rounding only; it is not a source substitution rule.
    passed = len(m) >= 250 and within_bp >= 0.995
    return {
        "series": name,
        "status": "PASS" if passed else "FAIL",
        "primary": f"DBnomics FED/H15/{DBNOMICS[name]}",
        "fallback": f"FRED/{FRED[name]}",
        "overlap_rows": int(len(m)),
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
    (OUT / "fallback_equivalence_audit.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    # Gate 1 = source build (already established); Gate 2 = fallback equivalence; Gate 3 = publication lag.
    completed = 2 if passed == len(results) else 1
    progress = {
        "lane": "0D_MACRO",
        "status": "PROVISIONAL" if completed < 3 else "PASS",
        "current_completed": completed,
        "target_total": 3,
        "completion_pct": round(completed / 3 * 100, 2),
        "newly_completed_this_run": 1 if completed == 2 else 0,
        "remaining": 3 - completed,
        "current_blocker": "publication/availability-lag audit remains" if completed == 2 else "fallback equivalence audit did not pass all macro series",
        "last_successful_unit": "fallback equivalence audit" if completed == 2 else "FED/H15 source build",
        "artifact_name": "alphapilot-v6-stage0-0D-macro-audit",
        "updated_at_utc": generated,
        "macro_series_total": len(results),
        "equivalence_audit_passed_series": passed,
        "publication_lag_audit_passed_series": 0,
        "unresolved_series": [r["series"] for r in results if r.get("status") != "PASS"],
        "unresolved_gates": ["publication_availability_lag"] + ([] if passed == len(results) else ["fallback_equivalence"]),
    }
    (PROGRESS / "0D_MACRO.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[0D EQUIVALENCE RESULTS] " + json.dumps(evidence, ensure_ascii=False))
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    if passed != len(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
