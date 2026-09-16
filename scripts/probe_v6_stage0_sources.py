#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pre_oos_audit"
OUT.mkdir(exist_ok=True)

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Stage0-Probe/1.3",
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


def probe_tip_transport():
    name = "TIP_HISTORY_TRANSPORT_DISCOVERY"
    page = "https://taiwanindex.com.tw/indexes/t00/history"
    t0 = time.time()
    row = {"name": name, "url": page, "ok": False}
    try:
        r = S.get(page, timeout=20)
        r.raise_for_status()
        html = r.text
        scripts = [urljoin(page, s) for s in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)', html, flags=re.I)]
        same_host = [u for u in scripts if urlparse(u).netloc == urlparse(page).netloc]
        corpus = [("page", html)]
        scanned = []
        for u in same_host[:18]:
            try:
                jr = S.get(u, timeout=10)
                if jr.ok and len(jr.text) <= 5_000_000:
                    corpus.append((u, jr.text))
                    scanned.append({"url": u, "bytes": len(jr.content)})
            except Exception as e:
                scanned.append({"url": u, "error": f"{type(e).__name__}: {e}"})

        contexts = []
        host_candidates = set()
        matches = set()
        for src, text in corpus:
            for needle in ("fileDownloadHost", "/api/download/history", "/history?start="):
                pos = 0
                while True:
                    i = text.find(needle, pos)
                    if i < 0:
                        break
                    ctx = text[max(0, i - 700):min(len(text), i + 1200)]
                    contexts.append({"source": src, "needle": needle, "context": ctx})
                    for u in re.findall(r'https?://[^\"\'\\\s<>]+', ctx, flags=re.I):
                        host_candidates.add(u.rstrip("/"))
                    pos = i + len(needle)
                    if len(contexts) >= 30:
                        break
                if len(contexts) >= 30:
                    break
            for pat in (
                re.compile(r'https?://[^\"\'\s<>]+', re.I),
                re.compile(r'/[A-Za-z0-9_./?=&%-]*(?:download|history|index)[A-Za-z0-9_./?=&%-]*', re.I),
            ):
                for m in pat.findall(text):
                    low = m.lower()
                    if any(k in low for k in ("download", "history", "index", "api")):
                        matches.add(m[:500])

        # Nuxt publicRuntimeConfig is usually serialized in SSR HTML.
        for _, text in corpus:
            for pat in (
                r'fileDownloadHost[\"\']?\s*[:=]\s*[\"\'](https?://[^\"\']+)',
                r'[\"\']fileDownloadHost[\"\']\s*:\s*[\"\'](https?://[^\"\']+)',
            ):
                for m in re.findall(pat, text, flags=re.I):
                    host_candidates.add(m.rstrip("/"))

        direct_tests = []
        # Only test plausible HTTP(S) origins, not unrelated external links.
        plausible = []
        for u in sorted(host_candidates):
            p = urlparse(u)
            if p.scheme in ("http", "https") and p.netloc:
                origin = f"{p.scheme}://{p.netloc}"
                if origin not in plausible and ("taiwanindex" in p.netloc or "twse" in p.netloc):
                    plausible.append(origin)
        # If config parsing failed, keep same-host as a diagnostic control.
        if "https://taiwanindex.com.tw" not in plausible:
            plausible.append("https://taiwanindex.com.tw")

        for host in plausible[:8]:
            try:
                dr = S.get(
                    host + "/api/download/history",
                    params={
                        "lang": "zh-tw",
                        "code": "t00",
                        "start": "2026-09-01",
                        "end": "2026-09-05",
                    },
                    timeout=20,
                    allow_redirects=True,
                )
                ctype = dr.headers.get("content-type", "")
                direct_tests.append({
                    "host": host,
                    "status_code": dr.status_code,
                    "content_type": ctype,
                    "bytes": len(dr.content),
                    "final_url": dr.url,
                    "content_disposition": dr.headers.get("content-disposition", ""),
                    "looks_download": dr.ok and (
                        "csv" in ctype.lower()
                        or "excel" in ctype.lower()
                        or "octet-stream" in ctype.lower()
                        or bool(dr.headers.get("content-disposition"))
                    ),
                    "text_prefix": dr.text[:500] if ("text" in ctype.lower() or "json" in ctype.lower()) else "",
                })
            except Exception as e:
                direct_tests.append({"host": host, "error": f"{type(e).__name__}: {e}"})

        row.update({
            "ok": True,
            "status_code": r.status_code,
            "elapsed_seconds": round(time.time() - t0, 3),
            "bytes": len(r.content),
            "script_count": len(scripts),
            "same_host_script_count": len(same_host),
            "scanned_scripts": scanned,
            "candidate_transports": sorted(matches)[:120],
            "candidate_contexts": contexts,
            "file_download_host_candidates": plausible,
            "download_contract_tests": direct_tests,
        })
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
        "FRED_DFF_PRIMARY",
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": "DFF"},
    ))
    # Use DBnomics' Federal Reserve Board provider (FED), not a guessed FRED provider.
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
    rows.append(probe_tip_transport())

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
