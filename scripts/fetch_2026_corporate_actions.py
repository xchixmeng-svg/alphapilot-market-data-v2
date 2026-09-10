#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 AlphaPilot-R10-Forward/2026.09",
    "Accept": "application/json,text/plain,*/*",
})

COLS = [
    "date", "code", "market", "event_type", "official_prev_close",
    "reference_price", "cash_dividend_per_share", "stock_shares_per_1000",
    "continuity_bridge", "source",
]

TWSE_FAMILIES = {
    "exright": "https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
    "reduction": "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU",
    "par_value": "https://www.twse.com.tw/rwd/zh/change/TWTB8U",
}
TPEX_FAMILIES = {
    "exright": "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ",
    "reduction": "https://www.tpex.org.tw/www/zh-tw/bulletin/revivt",
    "par_value": "https://www.tpex.org.tw/www/zh-tw/bulletin/pvChgRslt",
}


def get(url, params=None, timeout=90):
    last = None
    for i in range(5):
        try:
            r = S.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if i < 4:
                time.sleep(min(8, 2 ** i))
    raise RuntimeError(f"GET failed {url}: {last}")


def num(v):
    if v is None:
        return math.nan
    s = str(v).strip().replace(",", "").replace("+", "")
    if s in {"", "--", "---", "N/A", "null", "None"}:
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan


def code4(v):
    s = str(v or "").strip().strip('=\"')
    return s if re.fullmatch(r"[1-9]\d{3}|0\d{3}", s) else None


def roc_to_int(v):
    s = str(v or "").strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        return int(digits)
    if len(digits) == 7:
        return int(str(int(digits[:3]) + 1911) + digits[3:])
    m = re.search(r"(\d{2,4})\D+(\d{1,2})\D+(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        y = y + 1911 if y < 1911 else y
        return y * 10000 + mo * 100 + d
    return None


def pick(r, *candidates):
    for cand in candidates:
        if cand in r:
            return r[cand]
    norm = {re.sub(r"\s+", "", str(k)): v for k, v in r.items()}
    for cand in candidates:
        cc = re.sub(r"\s+", "", cand)
        if cc in norm:
            return norm[cc]
    return None


def greg_slash(ds):
    return f"{ds[:4]}/{ds[4:6]}/{ds[6:]}"


def extract_tables(payload):
    tables = []
    if isinstance(payload.get("fields"), list) and isinstance(payload.get("data"), list):
        tables.append((payload["fields"], payload["data"]))
    for t in payload.get("tables") or []:
        if isinstance(t, dict) and isinstance(t.get("fields"), list) and isinstance(t.get("data"), list):
            tables.append((t["fields"], t["data"]))
    return tables


def exright_type(v):
    s = str(v or "").strip()
    if "權" in s and "息" in s:
        return "EX_RIGHT_DIVIDEND"
    if "權" in s:
        return "EX_RIGHT"
    if "息" in s:
        return "EX_DIVIDEND"
    return "CORPORATE_ACTION"


def normalize_row(market, family, r):
    if family == "exright":
        d = roc_to_int(pick(r, "資料日期", "除權息日期", "日期"))
        prev = num(pick(r, "除權息前收盤價"))
        ref = num(pick(r, "除權息參考價"))
        typ_raw = pick(r, "權/息", "權息")
        event_type = exright_type(typ_raw)
        cash = num(pick(r, "現金股利", "息值"))
        if not math.isfinite(cash) and event_type == "EX_DIVIDEND" and math.isfinite(prev) and math.isfinite(ref):
            cash = max(0.0, prev - ref)
        shares = num(pick(r, "每仟股無償配股", "每千股無償配股", "無償配股"))
    else:
        d = roc_to_int(pick(r, "恢復買賣日期", "日期"))
        prev = num(pick(
            r,
            "停止買賣前收盤價格", "停止買賣前收盤價",
            "最後交易日之收盤價格", "最後交易之收盤價格",
        ))
        ref = num(pick(
            r,
            "恢復買賣參考價", "減資恢復買賣開始日參考價格",
            "恢復買賣開始參考價", "開始交易基準價", "開盤競價基準",
        ))
        if family == "reduction":
            reason = str(pick(r, "減資原因", "減資源因") or "UNKNOWN").strip()
            event_type = f"REDUCTION:{reason}"
        else:
            event_type = "PAR_VALUE_CHANGE"
        cash = math.nan
        shares = math.nan

    c = code4(pick(r, "股票代號", "證券代號", "代號", "ETF代號"))
    if not c or not d:
        return None
    if not (math.isfinite(prev) and prev > 0 and math.isfinite(ref) and ref > 0):
        return None

    return {
        "date": d,
        "code": c,
        "market": market,
        "event_type": event_type,
        "official_prev_close": prev,
        "reference_price": ref,
        "cash_dividend_per_share": cash,
        "stock_shares_per_1000": shares,
        "continuity_bridge": ref / prev,
        "source": f"{market}:{family}",
    }


def fetch_family(market, family, start, end):
    if market == "TWSE":
        url = TWSE_FAMILIES[family]
        params = {"startDate": start, "endDate": end, "response": "json"}
    else:
        url = TPEX_FAMILIES[family]
        params = {"startDate": greg_slash(start), "endDate": greg_slash(end), "response": "json"}

    payload = get(url, params).json()
    stat = str(payload.get("stat") or "").strip()
    if stat and stat.lower() != "ok" and "沒有符合" not in stat:
        raise RuntimeError(f"{market} {family} stat={stat!r}")

    tables = extract_tables(payload)
    if not tables:
        # A verified empty response is acceptable; malformed non-empty is not.
        if "沒有符合" in stat or stat.lower() == "ok":
            return [], {"count": 0, "stat": stat, "tables": 0}
        raise RuntimeError(f"{market} {family} schema keys={list(payload)[:20]}")

    out = []
    schemas = []
    for fields, data in tables:
        schemas.append(fields)
        for vals in data:
            if not isinstance(vals, list):
                continue
            row = normalize_row(market, family, dict(zip(fields, vals)))
            if row is None:
                continue
            if int(start) <= int(row["date"]) <= int(end):
                out.append(row)

    return out, {
        "count": len(out),
        "stat": stat,
        "tables": len(tables),
        "fields": schemas[:2],
    }


def fetch_all(start, end):
    rows = []
    meta = {}
    for market in ("TWSE", "TPEX"):
        for family in ("exright", "reduction", "par_value"):
            got, info = fetch_family(market, family, start, end)
            rows.extend(got)
            meta[f"{market}_{family}"] = info
    return rows, meta


def next_day(ds):
    return (datetime.strptime(ds, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/reference/official_corporate_actions_2026.csv")
    a = ap.parse_args()

    out = Path(a.out)
    meta_path = Path(str(out) + ".meta.json")
    existing = pd.DataFrame(columns=COLS)
    last_checked = None

    if out.exists() and out.stat().st_size > 0:
        existing = pd.read_csv(out, dtype={"code": str})
        for c in COLS:
            if c not in existing:
                existing[c] = math.nan
    if meta_path.exists():
        try:
            last_checked = json.loads(meta_path.read_text(encoding="utf-8")).get("last_checked_date")
        except Exception:
            last_checked = None

    start = a.start or (next_day(last_checked) if last_checked else "20260101")
    if int(start) > int(a.end):
        print(json.dumps({
            "status": "PASS", "mode": "NOOP", "last_checked_date": last_checked,
            "rows": len(existing),
        }, ensure_ascii=False))
        return

    fresh_rows, family_meta = fetch_all(start, a.end)
    fresh = pd.DataFrame(fresh_rows, columns=COLS)
    df = pd.concat([existing[COLS], fresh], ignore_index=True)

    if not df.empty:
        df["code"] = df["code"].astype(str).str.zfill(4)
        df["date"] = pd.to_numeric(df["date"], errors="raise").astype(int)
        # If multiple official families share the same effective date/code, keep
        # the most specific mechanical reset source: par value > reduction > exright.
        priority = {"exright": 1, "reduction": 2, "par_value": 3}
        df["_priority"] = df["source"].astype(str).str.split(":").str[-1].map(priority).fillna(0)
        df = (
            df.sort_values(["date", "code", "_priority", "source"])
              .drop_duplicates(["date", "code"], keep="last")
              .drop(columns=["_priority"])
        )
        if df["reference_price"].isna().any():
            raise RuntimeError("missing reference_price in corporate-action rows")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, columns=COLS)
    meta = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_checked_date": a.end,
        "increment_range": [start, a.end],
        "rows": int(len(df)),
        "new_rows": int(len(fresh)),
        "families": family_meta,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
