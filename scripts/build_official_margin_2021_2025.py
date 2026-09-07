#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "history" / "2020-2025"
OUT = ROOT / "research_output" / "margin_2021_2025"
OUT.mkdir(parents=True, exist_ok=True)
HEAD = {
    "User-Agent": "Mozilla/5.0 AlphaPilot-Margin-Audit/1.1",
    "Accept": "application/json,text/plain,*/*",
}
TWSE_URLS = (
    "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
    "https://www.twse.com.tw/exchangeReport/MI_MARGN",
)


def num(v):
    s = str(v or "").strip().replace(",", "").replace("+", "")
    if s in ("", "--", "---", "null", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def code4(v):
    s = str(v or "").strip()
    return s if re.fullmatch(r"\d{4}", s) else None


def request_json(url, params, attempts=8, base_sleep=3.0):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=60)
            r.raise_for_status()
            j = r.json()
            stat = str(j.get("stat", "")).strip().upper() if isinstance(j, dict) else ""
            if j and stat not in {"很抱歉，沒有符合條件的資料!", "查詢過於頻繁，請稍後再試"}:
                return j
            last = RuntimeError(f"empty/throttled response stat={stat!r}")
        except Exception as e:
            last = e
        time.sleep(min(90.0, base_sleep * (2 ** min(i, 4))) + random.uniform(0.2, 1.2))
    raise RuntimeError(f"GET failed: {url} {params}: {last}")


def twse(ds):
    params = {"date": ds.replace("-", ""), "selectType": "ALL", "response": "json"}
    errors = []
    for url in TWSE_URLS:
        try:
            j = request_json(url, params, attempts=5, base_sleep=4.0)
            tables = j.get("tables") or []
            table = next(
                (t for t in tables if "融資融券彙總" in str(t.get("title", ""))),
                None,
            )
            if not table:
                raise RuntimeError(f"margin table missing; stat={j.get('stat')!r}")
            out = []
            for r in table.get("data") or []:
                c = code4(r[0] if r else None)
                if not c or len(r) < 14:
                    continue
                out.append({
                    "date": ds, "market": "TWSE", "code": c, "name": str(r[1]).strip(),
                    "margin_buy": num(r[2]), "margin_sell": num(r[3]), "margin_repay": num(r[4]),
                    "margin_prev": num(r[5]), "margin_balance": num(r[6]), "margin_limit": num(r[7]),
                    "short_sell": num(r[8]), "short_buy": num(r[9]), "short_repay": num(r[10]),
                    "short_prev": num(r[11]), "short_balance": num(r[12]), "short_limit": num(r[13]),
                    "offset": num(r[14]) if len(r) > 14 else None,
                })
            if not out:
                raise RuntimeError("parsed zero rows")
            return out
        except Exception as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError(" | ".join(errors))


def roc_date(ds):
    d = datetime.strptime(ds, "%Y-%m-%d")
    return f"{d.year - 1911:03d}/{d.month:02d}/{d.day:02d}"


def tpex(ds):
    j = request_json(
        "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php",
        {"l": "zh-tw", "o": "json", "d": roc_date(ds), "s": "0,asc"},
        attempts=6,
        base_sleep=1.5,
    )
    tables = j.get("tables") or []
    if not tables:
        raise RuntimeError(f"TPEX margin table missing {ds}")
    out = []
    for r in tables[0].get("data") or []:
        c = code4(r[0] if r else None)
        if not c or len(r) < 19:
            continue
        out.append({
            "date": ds, "market": "TPEX", "code": c, "name": str(r[1]).strip(),
            "margin_prev": num(r[2]), "margin_buy": num(r[3]), "margin_sell": num(r[4]),
            "margin_repay": num(r[5]), "margin_balance": num(r[6]),
            "margin_usage_pct": num(r[8]), "margin_limit": num(r[9]),
            "short_prev": num(r[10]), "short_sell": num(r[11]), "short_buy": num(r[12]),
            "short_repay": num(r[13]), "short_balance": num(r[14]),
            "short_usage_pct": num(r[16]), "short_limit": num(r[17]), "offset": num(r[18]),
        })
    if not out:
        raise RuntimeError(f"TPEX parsed zero rows {ds}")
    return out


dates = set()
for year in range(2021, 2026):
    d = pd.read_parquet(HIST / f"ohlcv_{year}.parquet", columns=["date"])
    vals = pd.to_numeric(d["date"], errors="coerce").dropna().astype(int).astype(str)
    dates.update(datetime.strptime(v, "%Y%m%d").strftime("%Y-%m-%d") for v in vals)
dates = sorted(dates)
assert dates[0] == "2021-01-04" and dates[-1] == "2025-12-31"

rows = []
failures = []

# TPEx tolerates modest concurrency. Fetch it first so TWSE receives no competing load.
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(tpex, ds): ds for ds in dates}
    for i, fut in enumerate(as_completed(futs), 1):
        ds = futs[fut]
        try:
            rows.extend(fut.result())
        except Exception as e:
            failures.append({"date": ds, "market": "TPEX", "error": str(e)})
        if i % 100 == 0 or i == len(futs):
            print(f"[TPEX] {i}/{len(futs)} failures={sum(x['market'] == 'TPEX' for x in failures)}", flush=True)

# TWSE rate-limits bursty GitHub-hosted traffic. Use one request stream, deliberate
# pacing, periodic cool-downs, then two repair passes for isolated throttled dates.
pending = list(dates)
for round_no in range(3):
    next_pending = []
    if round_no:
        cool = 120 * round_no
        print(f"[TWSE] repair round {round_no + 1}; cooldown={cool}s pending={len(pending)}", flush=True)
        time.sleep(cool)
    for i, ds in enumerate(pending, 1):
        try:
            rows.extend(twse(ds))
        except Exception as e:
            next_pending.append((ds, str(e)))
        time.sleep(1.8 + random.uniform(0.1, 0.7))
        if i % 40 == 0:
            print(f"[TWSE] round={round_no + 1} {i}/{len(pending)} pending={len(next_pending)}", flush=True)
            time.sleep(20)
    if not next_pending:
        pending = []
        break
    pending = [ds for ds, _ in next_pending]
    final_errors = dict(next_pending)

for ds in pending:
    failures.append({"date": ds, "market": "TWSE", "error": final_errors[ds]})

rows.sort(key=lambda r: (r["date"], r["market"], r["code"]))
fields = sorted({k for r in rows for k in r})
fields = ["date", "market", "code", "name"] + [
    x for x in fields if x not in {"date", "market", "code", "name"}
]
csv_path = OUT / "official_margin_short_2021_2025.csv.gz"
pd.DataFrame(rows, columns=fields).to_csv(csv_path, index=False, compression="gzip")

coverage = {}
for market in ("TWSE", "TPEX"):
    have = {r["date"] for r in rows if r["market"] == market}
    missing = sorted(set(dates) - have)
    coverage[market] = {
        "dates_expected": len(dates),
        "dates_present": len(have),
        "ratio": len(have) / len(dates),
        "missing_dates": missing,
    }

manifest = {
    "dataset": "AlphaPilot official margin and short balances 2021-2025",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "date_start": dates[0],
    "date_end": dates[-1],
    "trading_dates": len(dates),
    "rows": len(rows),
    "coverage": coverage,
    "failures": failures,
    "sources": {"TWSE": list(TWSE_URLS), "TPEX": "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"},
    "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"rows": len(rows), "coverage": coverage, "failure_count": len(failures)}, ensure_ascii=False), flush=True)
if min(v["ratio"] for v in coverage.values()) < 0.98:
    raise RuntimeError("coverage below 98%; incomplete data is quarantined and backtest must not run")
print("[PASS] both markets have >=98% trading-date coverage", flush=True)
