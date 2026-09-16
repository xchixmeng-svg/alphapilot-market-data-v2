#!/usr/bin/env python3
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import build_v6_context_layers_resumable as r


def _fetch_industry_strict(d: date):
    """Never certify an empty TWSE response as a non-trading day without independent proof."""
    p = r._industry_cache_path(d)
    cached = r._safe_cached_rows(p)
    if cached is not None:
        return cached, "cache"

    # Earlier V6 runs could create an empty marker from an ambiguous HTTP-200/empty response.
    # Such markers are not valid PIT evidence, so invalidate them and retry as unresolved.
    r._industry_empty_path(d).unlink(missing_ok=True)

    try:
        got = r._original_parse_industry(d)
        if not got:
            err = RuntimeError("unverified empty TWSE industry response; not accepted as non-trading-day evidence")
            r._mark_industry_failure(d, err)
            raise err
        r._write_json_gz(p, got)
        r._industry_fail_path(d).unlink(missing_ok=True)
        return got, "network"
    except Exception as e:
        # Avoid double-incrementing the explicit empty case above.
        if "unverified empty TWSE industry response" not in str(e):
            r._mark_industry_failure(d, e)
        raise


def build_industry_indices_strict():
    dates = list(r.base.weekdays(date(r.base.START_YEAR, 1, 1), r.base.END_DATE))

    invalidated_empty = 0
    for d in dates:
        p = r._industry_empty_path(d)
        if p.exists():
            p.unlink(missing_ok=True)
            invalidated_empty += 1
    if invalidated_empty:
        print("[IND STRICT] invalidated_unverified_empty_markers", invalidated_empty, flush=True)

    rows = []
    cached_dates = set()
    for d in dates:
        cached = r._safe_cached_rows(r._industry_cache_path(d))
        if cached is not None:
            rows.extend(cached)
            cached_dates.add(d.isoformat())

    missing = [d for d in dates if d.isoformat() not in cached_dates]
    never_attempted = [d for d in missing if r._failure_attempts(d) == 0]
    retry_dates = [d for d in missing if r._failure_attempts(d) > 0]
    never_attempted.sort()
    retry_dates.sort(key=lambda d: (r._failure_attempts(d), d))
    batch = (never_attempted + retry_dates)[: r.MAX_IND_REQUESTS]

    failures = []
    fresh_dates = 0
    print(
        "[IND STRICT RESUME] cached_dates", len(cached_dates),
        "unresolved", len(missing),
        "never_attempted", len(never_attempted),
        "retry_dates", len(retry_dates),
        "this_run_requests", len(batch),
        "workers", r.IND_WORKERS,
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=r.IND_WORKERS) as ex:
        fut = {ex.submit(_fetch_industry_strict, d): d for d in batch}
        for i, f in enumerate(as_completed(fut), 1):
            d = fut[f]
            try:
                got, source = f.result()
                rows.extend(got)
                if source == "network":
                    fresh_dates += 1
            except Exception as e:
                failures.append({"date": d.isoformat(), "attempts": r._failure_attempts(d), "error": str(e)})
            if i % 50 == 0 or i == len(fut):
                coverage = len({x["date"] for x in rows})
                print(
                    "[IND STRICT PROGRESS]", i, "/", len(fut),
                    "fresh_saved", fresh_dates,
                    "coverage_dates", coverage,
                    "rows", len(rows),
                    "hard_fail", len(failures),
                    flush=True,
                )

    rows = sorted({(x["date"], x["index_name"]): x for x in rows}.values(), key=lambda x: (x["date"], x["index_name"]))
    coverage_dates = len({x["date"] for x in rows})
    unresolved_after = sum(1 for d in dates if r._safe_cached_rows(r._industry_cache_path(d)) is None)
    print(
        "[IND STRICT CHECKPOINT] coverage_dates", coverage_dates,
        "fresh_saved", fresh_dates,
        "unresolved_total", unresolved_after,
        flush=True,
    )
    if coverage_dates < 1800:
        raise RuntimeError(
            f"industry index coverage too low: dates={coverage_dates}; "
            f"successful checkpoints preserved; unresolved={unresolved_after}; "
            "empty HTTP responses are deliberately not certified as holidays"
        )

    dest = r.base.OUT / "twse_industry_index_daily.csv.gz"
    r.base.write_gz(dest, rows)
    return dest, failures, len(rows), coverage_dates, len({x["index_name"] for x in rows})


def main():
    r.base.build_industry_indices = build_industry_indices_strict
    r.main()


if __name__ == "__main__":
    main()
