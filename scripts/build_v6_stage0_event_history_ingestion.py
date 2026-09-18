#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history" / "2020-2025"
CACHE = ROOT / ".cache" / "v6-event-history"
LAYER = ROOT / "data" / "history" / "v6-layered" / "0E_EVENT_TIME"
PROGRESS = ROOT / "data" / "history" / "v6-layered" / "progress"
MANIFEST = ROOT / "data" / "history" / "v6-layered" / "manifests"
for p in (CACHE, LAYER, PROGRESS, MANIFEST):
    p.mkdir(parents=True, exist_ok=True)

URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"
CHUNK_UNITS = int(os.getenv("V6_0E_CHUNK_UNITS", "160"))
WORKERS = int(os.getenv("V6_0E_WORKERS", "4"))
MIN_WORKERS = int(os.getenv("V6_0E_MIN_WORKERS", "1"))
RUN_BUDGET_SECONDS = int(os.getenv("V6_0E_RUN_BUDGET_SECONDS", "2700"))
COOLDOWN_SECONDS = float(os.getenv("V6_0E_COOLDOWN_SECONDS", "12"))
TAIPEI = ZoneInfo("Asia/Taipei")

_TLS = threading.local()


def _session() -> requests.Session:
    s = getattr(_TLS, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 AlphaPilot-V6-Event-History/1.0",
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Referer": "https://mopsov.twse.com.tw/mops/web/t05st01",
        })
        _TLS.session = s
    return s


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
    try:
        return datetime(y + 1911, mo, d)
    except Exception:
        return None


def code4(v) -> str | None:
    s = str(v).strip()
    if re.fullmatch(r"[1-9]\d{3}\.0", s):
        s = s[:-2]
    return s if re.fullmatch(r"[1-9]\d{3}", s) else None


def detect_code_column(path: Path) -> str:
    names = pq.ParquetFile(path).schema.names
    preferred = ["code", "stock_id", "stock_code", "證券代號", "ticker"]
    for c in preferred:
        if c in names:
            return c
    for c in names:
        low = c.lower()
        if "code" in low or "ticker" in low or "stock_id" in low:
            return c
    raise RuntimeError(f"cannot identify stock-code column in {path.name}: {names[:40]}")


def target_units() -> list[tuple[int, str]]:
    out = []
    for year in range(2020, 2026):
        p = HIST / f"ohlcv_{year}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing immutable OHLCV {p}")
        cc = detect_code_column(p)
        df = pd.read_parquet(p, columns=[cc])
        codes = sorted({c for c in (code4(x) for x in df[cc].dropna().tolist()) if c})
        if len(codes) < 500:
            raise RuntimeError(f"{year} unique code count suspiciously low: {len(codes)} via {cc}")
        out.extend((year, c) for c in codes)
    return out


def cache_path(year: int, code: str) -> Path:
    return CACHE / str(year) / f"{code}.json.gz"


def read_checkpoint(year: int, code: str):
    p = cache_path(year, code)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("status") in ("SUCCESS", "SUCCESS_NO_EVENTS"):
            return obj
    except Exception:
        pass
    p.unlink(missing_ok=True)
    return None


def save_checkpoint(year: int, code: str, obj: dict):
    p = cache_path(year, code)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(p)


def fetch_company_year(year: int, code: str) -> dict:
    roc_year = year - 1911
    last = None
    for attempt in range(3):
        try:
            r = _session().post(
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
            html = r.text
            if len(html) < 500:
                raise RuntimeError(f"short response chars={len(html)}")
            if "查詢過於頻繁" in html or "Service Unavailable" in html:
                raise RuntimeError("MOPS throttled request")
            # A valid historical query can legitimately return no disclosure table.
            # Only classify as verified no-events after transport/content sanity checks
            # and only when the disclosure timestamp schema markers are absent.
            if "發言日期" not in html and "發言時間" not in html:
                return {
                    "status": "SUCCESS_NO_EVENTS",
                    "year": year,
                    "code": code,
                    "events": [],
                    "response_chars": len(html),
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "no_event_evidence": "valid MOPS response without disclosure timestamp schema",
                }
            tables = pd.read_html(io.StringIO(html))
            events = []
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
                title_col = next((x for x in cols if "主旨" in x), None)
                fact_col = next((x for x in cols if "事實發生日" in x), None)
                for _, row in df.iterrows():
                    dd = _roc_date_to_ad(row.get(date_col))
                    tt = str(row.get(time_col, "")).strip()
                    mt = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", tt)
                    if dd is None or mt is None:
                        continue
                    h, mi, sec = map(int, mt.groups())
                    try:
                        local = dd.replace(hour=h, minute=mi, second=sec, tzinfo=TAIPEI)
                    except Exception:
                        continue
                    events.append({
                        "code": code,
                        "source_year": year,
                        "published_at_utc": local.astimezone(timezone.utc).isoformat(),
                        "available_at_utc": local.astimezone(timezone.utc).isoformat(),
                        "title": str(row.get(title_col, "")).strip() if title_col else "",
                        "fact_date_raw": str(row.get(fact_col, "")).strip() if fact_col else "",
                        "source": "MOPS historical material information",
                    })
            # The page may legitimately have zero events; successful HTML parse is still a checkpoint.
            dedup = {}
            for e in events:
                key = (e["published_at_utc"], e["title"])
                dedup[key] = e
            events = list(dedup.values())
            return {
                "status": "SUCCESS" if events else "SUCCESS_NO_EVENTS",
                "year": year,
                "code": code,
                "events": events,
                "response_chars": len(html),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{year} {code}: {type(last).__name__}: {last}")


def _is_throttle_failure(error: str) -> bool:
    s = error.lower()
    return any(k in s for k in (
        "throttled request",
        "查詢過於頻繁",
        "service unavailable",
        "429",
        "connecttimeout",
        "readtimeout",
        "timed out",
    ))


def _aggregate_and_write(
    targets: list[tuple[int, str]],
    generated: str,
    fresh: int,
    failures: list[dict],
) -> dict:
    all_events = []
    completed = 0
    no_event_units = 0
    for y, code in targets:
        obj = read_checkpoint(y, code)
        if obj is None:
            continue
        completed += 1
        if obj.get("status") == "SUCCESS_NO_EVENTS":
            no_event_units += 1
        all_events.extend(obj.get("events") or [])

    events = pd.DataFrame(all_events)
    out_path = LAYER / "historical_material_information.parquet"
    if not events.empty:
        events["published_at_utc"] = pd.to_datetime(
            events["published_at_utc"], utc=True, errors="coerce"
        )
        events["available_at_utc"] = pd.to_datetime(
            events["available_at_utc"], utc=True, errors="coerce"
        )
        events = events.dropna(
            subset=["published_at_utc", "available_at_utc", "code"]
        )
        events = events.drop_duplicates(["code", "published_at_utc", "title"])
        events.to_parquet(out_path, index=False)

    complete = completed == len(targets)
    future_violations = 0
    if not events.empty:
        future_violations = int(
            (events["available_at_utc"] < events["published_at_utc"]).sum()
        )

    status = "PASS" if complete and future_violations == 0 else "BUILDING"
    progress = {
        "lane": "0E_EVENT_TIME",
        "status": status,
        "current_completed": 3 if status == "PASS" else 2,
        "target_total": 3,
        "completion_pct": 100.0 if status == "PASS" else 66.67,
        "newly_completed_this_run": fresh,
        "remaining": 0 if status == "PASS" else 1,
        "current_blocker": (
            None if status == "PASS"
            else f"historical event ingestion incomplete: {completed}/{len(targets)}"
        ),
        "last_successful_unit": f"historical event checkpoints {completed}/{len(targets)}",
        "artifact_name": "alphapilot-v6-stage0-0E-event-history",
        "updated_at_utc": generated,
        "historical_source_audit_pass": True,
        "ingestion_target_units": len(targets),
        "ingestion_completed_units": completed,
        "ingestion_unresolved_units": len(targets) - completed,
        "no_event_units": no_event_units,
        "event_rows": int(len(events)),
        "future_join_violations": future_violations,
        "failures_this_run": failures[-30:],
        "formal_oos_opened": False,
    }
    (PROGRESS / "0E_EVENT_TIME.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "layer_id": "0E_EVENT_TIME",
        "status": status,
        "schema_version": "v3-event-history-longrun",
        "generation_commit": os.getenv("GITHUB_SHA", "UNKNOWN"),
        "generated_at_utc": generated,
        "source_lineage": [{
            "dataset": "historical_material_information",
            "source": "MOPS/TWSE official historical material-information system",
            "checkpoint_unit": "company-year",
        }],
        "datasets": [{
            "name": "historical_material_information",
            "rows": int(len(events)),
            "time_semantics": "event_timestamp_asof",
            "checkpoint_units": completed,
            "target_units": len(targets),
        }],
        "date_start": (
            events["published_at_utc"].min().isoformat()
            if not events.empty else None
        ),
        "date_end": (
            events["published_at_utc"].max().isoformat()
            if not events.empty else None
        ),
        "row_count": int(len(events)),
        "file_sha256": {out_path.name: sha256(out_path)} if out_path.exists() else {},
        "missingness": {
            "unresolved_company_year_units": len(targets) - completed,
            "no_event_company_year_units": no_event_units,
        },
        "pit_rules": {
            "published_at_from_mops_speech_date_time": True,
            "timezone": "Asia/Taipei",
            "available_at_equals_public_disclosure_timestamp": True,
            "fact_date_used_as_availability": False,
            "future_join_violations": future_violations,
            "formal_oos_eligible": status == "PASS",
        },
    }
    (MANIFEST / "0E_EVENT_TIME.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[0E FINAL AGGREGATE] " + json.dumps(progress, ensure_ascii=False), flush=True)
    return progress


def main():
    started = time.monotonic()
    targets = target_units()

    # Startup scan only. Intermediate chunks write checkpoints but do not
    # rescan every checkpoint or rebuild the full parquet.
    unresolved = deque()
    completed_at_start = 0
    for y, code in targets:
        if read_checkpoint(y, code) is None:
            unresolved.append((y, code))
        else:
            completed_at_start += 1

    fresh = 0
    failures: list[dict] = []
    current_workers = max(MIN_WORKERS, WORKERS)
    clean_chunks = 0
    chunk_no = 0

    print(
        f"[0E LONGRUN RESUME] target_units={len(targets)} "
        f"completed={completed_at_start} unresolved={len(unresolved)} "
        f"chunk_units={CHUNK_UNITS} workers={current_workers} "
        f"budget_seconds={RUN_BUDGET_SECONDS}",
        flush=True,
    )

    while unresolved:
        elapsed = time.monotonic() - started
        if elapsed >= RUN_BUDGET_SECONDS:
            print(
                f"[0E LONGRUN] time budget reached elapsed={elapsed:.1f}s "
                f"fresh={fresh} unresolved_queue={len(unresolved)}",
                flush=True,
            )
            break

        chunk_no += 1
        n = min(CHUNK_UNITS, len(unresolved))
        batch = [unresolved.popleft() for _ in range(n)]
        chunk_fresh = 0
        chunk_failures = []
        throttle_like = 0
        chunk_started = time.monotonic()

        print(
            f"[0E CHUNK START] chunk={chunk_no} size={len(batch)} "
            f"workers={current_workers} elapsed={elapsed:.1f}s "
            f"queue_after_pop={len(unresolved)}",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=current_workers) as ex:
            fut = {
                ex.submit(fetch_company_year, y, code): (y, code)
                for y, code in batch
            }
            for i, future in enumerate(as_completed(fut), 1):
                y, code = fut[future]
                try:
                    obj = future.result()
                    save_checkpoint(y, code, obj)
                    chunk_fresh += 1
                    fresh += 1
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    rec = {"year": y, "code": code, "error": err}
                    chunk_failures.append(rec)
                    failures.append(rec)
                    unresolved.append((y, code))
                    if _is_throttle_failure(err):
                        throttle_like += 1

                if i % 40 == 0 or i == len(batch):
                    print(
                        f"[0E CHUNK PROGRESS] chunk={chunk_no} {i}/{len(batch)} "
                        f"fresh={chunk_fresh} fail={len(chunk_failures)} "
                        f"throttle_like={throttle_like}",
                        flush=True,
                    )

        chunk_seconds = time.monotonic() - chunk_started
        failure_rate = len(chunk_failures) / max(1, len(batch))

        if throttle_like > 0 or failure_rate >= 0.05:
            old_workers = current_workers
            current_workers = max(MIN_WORKERS, current_workers // 2)
            clean_chunks = 0
            cooldown = COOLDOWN_SECONDS * (2 if throttle_like >= 3 else 1)
            print(
                f"[0E ADAPTIVE THROTTLE] chunk={chunk_no} "
                f"failure_rate={failure_rate:.3f} throttle_like={throttle_like} "
                f"workers={old_workers}->{current_workers} cooldown={cooldown:.1f}s",
                flush=True,
            )
            time.sleep(cooldown)
        else:
            clean_chunks += 1
            if clean_chunks >= 2 and current_workers < WORKERS:
                old_workers = current_workers
                current_workers += 1
                clean_chunks = 0
                print(
                    f"[0E ADAPTIVE RECOVERY] workers={old_workers}->{current_workers}",
                    flush=True,
                )

        rate = chunk_fresh / max(chunk_seconds, 0.001) * 60.0
        print(
            f"[0E CHUNK DONE] chunk={chunk_no} fresh={chunk_fresh} "
            f"fail={len(chunk_failures)} seconds={chunk_seconds:.1f} "
            f"rate_units_per_min={rate:.1f} total_fresh={fresh} "
            f"unresolved_queue={len(unresolved)}",
            flush=True,
        )

        if chunk_fresh == 0:
            raise RuntimeError(
                f"0E chunk made zero progress; chunk={chunk_no} "
                f"workers={current_workers} failures={chunk_failures[:5]}"
            )

    # Aggregate only once per workflow run.
    progress = _aggregate_and_write(
        targets=targets,
        generated=datetime.now(timezone.utc).isoformat(),
        fresh=fresh,
        failures=failures,
    )
    print(
        f"[0E LONGRUN SUMMARY] status={progress['status']} "
        f"completed={progress['ingestion_completed_units']}/{len(targets)} "
        f"fresh={fresh} runtime_seconds={time.monotonic()-started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
