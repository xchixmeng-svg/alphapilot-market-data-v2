#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import build_v6_stage0_industry_layer as base

RETRY_META = base.CACHE_ROOT / "twse_old_retry_metadata.json"
MARKET_PATH = base.ROOT / "data" / "history" / "v6-layered" / "0A_MARKET" / "market_daily.parquet"


def _load_retry() -> dict:
    if not RETRY_META.exists():
        return {"schema": "v1", "dates": {}}
    try:
        x = json.loads(RETRY_META.read_text(encoding="utf-8"))
        if isinstance(x, dict) and isinstance(x.get("dates"), dict):
            return x
    except Exception:
        pass
    return {"schema": "v1", "dates": {}}


def _save_retry(x: dict) -> None:
    tmp = RETRY_META.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(RETRY_META)


def actual_old_trading_days() -> list[pd.Timestamp]:
    if not MARKET_PATH.exists():
        raise RuntimeError("0A market_daily.parquet is required before 0B checkpoint v3")
    m = pd.read_parquet(MARKET_PATH, columns=["date"])
    raw = m["date"]
    if pd.api.types.is_numeric_dtype(raw):
        d = pd.to_datetime(raw.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        d = pd.to_datetime(raw, errors="coerce")
    d = d[(d >= base.START) & (d <= base.OLD_END)].dropna().drop_duplicates().sort_values()
    out = [pd.Timestamp(x) for x in d.tolist()]
    if len(out) < 1100:
        raise RuntimeError(f"0A-derived 2016-2020 trading calendar suspiciously short: {len(out)}")
    return out


def _validated_old_checkpoint(d: pd.Timestamp) -> dict | None:
    obj = base.load_old_checkpoint(d)
    if obj is None:
        return None
    if obj.get("status") == "SUCCESS":
        names = {str(r.get("index_name")) for r in (obj.get("rows") or [])}
        # TWSE renamed the broad electronics index from 電子類指數 to
        # 電子工業類指數 starting 2019-04-29. Old cached parses before the
        # alias fix are structurally incomplete and must be selectively
        # refetched; all unaffected dates remain immutable cache hits.
        if pd.Timestamp("2019-04-29") <= d <= base.OLD_END and "電子類指數" not in names:
            return None
    return obj


def fixed_load_or_fetch_tip_series(code: str, name: str):
    p = base.tip_series_cache_path(code)
    meta = p.with_suffix(".json")
    if p.exists() and meta.exists():
        try:
            df = pd.read_parquet(p)
            m = json.loads(meta.read_text(encoding="utf-8"))
            if (not df.empty and set(df["index_name"].unique()) == {name}
                    and str(m.get("end")) == base.END.date().isoformat()
                    and m.get("name") == name and m.get("sha256") == base.sha256(p)):
                return df, {**m, "cache_hit": True, "cache_validation": "SHA256+END+NAME"}
        except Exception:
            pass
        p.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
    df, audit = base.fetch_tip_series_segmented(code, name)
    df.to_parquet(p, index=False)
    m = {
        "cache_hit": False, "cache_validation": "FETCHED_AND_HASHED",
        "name": name, "code": code,
        "start": df["date"].min().date().isoformat(), "end": base.END.date().isoformat(),
        "rows": int(len(df)), "sha256": base.sha256(p), "transport_audit": audit,
    }
    meta.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, m


def advance_old_history_v3(code_map: dict[str, str]):
    targets = actual_old_trading_days()
    raw_completed = {d.date().isoformat(): base.load_old_checkpoint(d) for d in targets}
    completed = {d.date().isoformat(): _validated_old_checkpoint(d) for d in targets}
    unresolved = [d for d in targets if completed[d.date().isoformat()] is None]
    alias_repair_at_start = sum(
        1 for d in targets
        if raw_completed[d.date().isoformat()] is not None
        and completed[d.date().isoformat()] is None
    )
    retry = _load_retry()
    now = datetime.now(timezone.utc)

    eligible = []
    cooling = 0
    for d in unresolved:
        key = d.date().isoformat()
        rec = retry["dates"].get(key, {})
        nr = rec.get("next_retry_utc")
        if nr:
            try:
                if datetime.fromisoformat(nr) > now:
                    cooling += 1
                    continue
            except Exception:
                pass
        eligible.append(d)
    # Rotate by fewest attempts first, then oldest last-attempt; never unresolved[:N] starvation.
    eligible.sort(key=lambda d: (
        int(retry["dates"].get(d.date().isoformat(), {}).get("attempts", 0)),
        retry["dates"].get(d.date().isoformat(), {}).get("last_attempt_utc", ""),
        d.date().isoformat(),
    ))
    batch = eligible[:base.MAX_OLD_DATES_PER_RUN]
    fresh_success = fresh_no_data = transient_fail = 0
    errors = []
    throttle_stop = False
    attempted = 0
    print(
        f"[0B V3 RESUME] calendar=0A_MARKET targets={len(targets)} "
        f"validated={len(targets)-len(unresolved)} unresolved={len(unresolved)} "
        f"alias_repair_at_start={alias_repair_at_start} cooling={cooling} "
        f"eligible={len(eligible)} batch={len(batch)}",
        flush=True,
    )

    for i, d in enumerate(batch, 1):
        attempted += 1
        key = d.date().isoformat()
        try:
            obj = base.fetch_old_twse_date(d, code_map)
            base.save_old_checkpoint(d, obj)
            retry["dates"].pop(key, None)
            if obj["status"] == "SUCCESS": fresh_success += 1
            else: fresh_no_data += 1
        except Exception as e:
            transient_fail += 1
            msg = f"{type(e).__name__}: {e}"
            rec = retry["dates"].get(key, {})
            attempts = int(rec.get("attempts", 0)) + 1
            # Exponential cooldown capped at 24h; rotation prevents one bad date blocking the queue.
            cooldown_min = min(24 * 60, 15 * (2 ** min(attempts - 1, 6)))
            retry["dates"][key] = {
                "attempts": attempts,
                "last_attempt_utc": now.isoformat(),
                "next_retry_utc": (now + timedelta(minutes=cooldown_min)).isoformat(),
                "last_error": msg[:500],
            }
            errors.append({"date": key, "error": msg, "cooldown_minutes": cooldown_min})
            low = msg.lower()
            if "non-json" in low or "text/html" in low or "unexpected content" in low:
                throttle_stop = True
                print(f"[0B V3 THROTTLE STOP] date={key} error={msg}", flush=True)
                _save_retry(retry)
                break
        _save_retry(retry)
        if i < len(batch): time.sleep(base.OLD_REQUEST_DELAY_SECONDS)

    completed_now = {d.date().isoformat(): _validated_old_checkpoint(d) for d in targets}
    unresolved_after = [d for d in targets if completed_now[d.date().isoformat()] is None]
    rows = []
    success_dates = no_data_dates = 0
    for d in targets:
        obj = completed_now[d.date().isoformat()]
        if obj is None: continue
        if obj["status"] == "SUCCESS":
            success_dates += 1; rows.extend(obj.get("rows") or [])
        elif obj["status"] == "VERIFIED_NO_DATA": no_data_dates += 1
    df = pd.DataFrame(rows, columns=["date","index_code","index_name","price_index","total_return_index","change","change_pct","source"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date","price_index"])
    audit = {
        "target_calendar_source": "0A_MARKET/market_daily.parquet immutable-OHLCV-derived actual trading dates",
        "target_calendar_audit": {"date_start": targets[0].date().isoformat(), "date_end": targets[-1].date().isoformat(), "target_trading_dates": len(targets), "weekdays_not_assumed": True},
        "target_weekdays": len(targets),
        "raw_checkpoint_units": sum(1 for d in targets if base.load_old_checkpoint(d) is not None),
        "alias_repair_required_units_at_start": alias_repair_at_start,
        "alias_repair_remaining_units": len(unresolved_after),
        "completed_units": len(targets)-len(unresolved_after), "success_trading_dates": success_dates,
        "verified_no_data_dates": no_data_dates, "unresolved_units": len(unresolved_after),
        "unresolved_sample": [d.date().isoformat() for d in unresolved_after[:30]],
        "this_run_requested": attempted, "this_run_success": fresh_success,
        "this_run_verified_no_data": fresh_no_data, "this_run_transient_fail": transient_fail,
        "this_run_errors": errors[:30], "throttle_early_stop": throttle_stop,
        "retry_metadata_path": str(RETRY_META.relative_to(base.ROOT)), "cooling_units_at_start": cooling,
        "complete": len(unresolved_after)==0, "checkpoint_root": str(base.OLD_CACHE.relative_to(base.ROOT)),
        "permanent_skip_statuses": ["SUCCESS","VERIFIED_NO_DATA"],
    }
    return df, audit


def diagnose_electronics_coverage() -> None:
    targets = actual_old_trading_days()
    success_dates = 0
    present = []
    missing = []
    for d in targets:
        obj = base.load_old_checkpoint(d)
        if not obj or obj.get("status") != "SUCCESS":
            continue
        success_dates += 1
        names = {str(r.get("index_name")) for r in (obj.get("rows") or [])}
        if "電子類指數" in names:
            present.append(d)
        else:
            missing.append(d)

    by_year = {}
    for d in missing:
        by_year[str(d.year)] = by_year.get(str(d.year), 0) + 1
    evidence = {
        "success_checkpoint_dates": success_dates,
        "electronics_present_dates": len(present),
        "electronics_missing_dates": len(missing),
        "missing_by_year": by_year,
        "first_present": present[0].date().isoformat() if present else None,
        "last_present": present[-1].date().isoformat() if present else None,
        "first_missing": missing[0].date().isoformat() if missing else None,
        "last_missing": missing[-1].date().isoformat() if missing else None,
        "missing_sample": [d.date().isoformat() for d in missing[:30]],
    }
    print("[0B ELECTRONICS CACHE DIAG] " + json.dumps(evidence, ensure_ascii=False), flush=True)

    if not missing:
        return
    picks = []
    for idx in (0, len(missing) // 2, len(missing) - 1):
        d = missing[idx]
        if d not in picks:
            picks.append(d)
    for d in picks:
        try:
            r = base.get(base.TWSE_MI_INDEX, params={
                "response": "json", "date": d.strftime("%Y%m%d"), "type": "IND",
            }, timeout=30, tries=2)
            ctype = (r.headers.get("content-type") or "").lower()
            if "json" not in ctype:
                print(f"[0B ELECTRONICS RAW DIAG] date={d.date()} non_json={ctype}", flush=True)
                continue
            j = r.json()
            raw_names = []
            for table in j.get("tables") or []:
                if not isinstance(table, dict):
                    continue
                fields = [str(x).strip() for x in (table.get("fields") or [])]
                data = table.get("data") or []
                name_i = next((i for i, x in enumerate(fields) if x in ("指數", "指數名稱") or "指數" in x), None)
                if name_i is None:
                    continue
                for vals in data:
                    if isinstance(vals, list) and name_i < len(vals):
                        raw = str(vals[name_i]).strip()
                        if "電子" in raw or "電機" in raw or "半導體" in raw:
                            raw_names.append(raw)
            print(
                f"[0B ELECTRONICS RAW DIAG] date={d.date()} stat={j.get('stat')} "
                f"matching_raw_names={sorted(set(raw_names))}",
                flush=True,
            )
        except Exception as e:
            print(f"[0B ELECTRONICS RAW DIAG] date={d.date()} error={type(e).__name__}: {e}", flush=True)


base.load_or_fetch_tip_series = fixed_load_or_fetch_tip_series
base.old_weekdays = actual_old_trading_days
base.advance_old_history = advance_old_history_v3

if __name__ == "__main__":
    diagnose_electronics_coverage()
    base.main()
