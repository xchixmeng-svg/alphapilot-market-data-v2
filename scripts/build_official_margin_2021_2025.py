#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "history" / "2020-2025"
OUT = ROOT / "research_output" / "margin_2021_2025"
OUT.mkdir(parents=True, exist_ok=True)
HEAD = {"User-Agent": "Mozilla/5.0 AlphaPilot-Margin-Audit/1.0", "Accept": "application/json,text/plain,*/*"}

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

def get_json(url, params, attempts=6):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=60)
            r.raise_for_status()
            j = r.json()
            if j:
                return j
        except Exception as e:
            last = e
        time.sleep(min(12, 1.5 ** i))
    raise RuntimeError(f"GET failed: {url} {params}: {last}")

def twse(ds):
    j = get_json("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
                 {"date": ds.replace("-", ""), "selectType": "ALL", "response": "json"})
    tables = j.get("tables") or []
    table = next((t for t in tables if "融資融券彙總" in str(t.get("title", ""))), None)
    if not table:
        raise RuntimeError(f"TWSE margin table missing {ds}")
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
            "offset": num(r[14]) if len(r) > 14 else None
        })
    if not out:
        raise RuntimeError(f"TWSE parsed zero rows {ds}")
    return out

def roc_date(ds):
    d = datetime.strptime(ds, "%Y-%m-%d")
    return f"{d.year - 1911:03d}/{d.month:02d}/{d.day:02d}"

def tpex(ds):
    j = get_json("https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php",
                 {"l": "zh-tw", "o": "json", "d": roc_date(ds), "s": "0,asc"})
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
            "short_usage_pct": num(r[16]), "short_limit": num(r[17]), "offset": num(r[18])
        })
    if not out:
        raise RuntimeError(f"TPEX parsed zero rows {ds}")
    return out

dates = set()
for year in range(2021, 2026):
    p = HIST / f"ohlcv_{year}.parquet"
    d = pd.read_parquet(p, columns=["date"])
    vals = pd.to_numeric(d["date"], errors="coerce").dropna().astype(int).astype(str)
    dates.update(datetime.strptime(v, "%Y%m%d").strftime("%Y-%m-%d") for v in vals)
dates = sorted(dates)
assert dates[0] == "2021-01-04" and dates[-1] == "2025-12-31"

rows, failures = [], []
tasks = [(ds, "TWSE", twse) for ds in dates] + [(ds, "TPEX", tpex) for ds in dates]
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(fn, ds): (ds, market) for ds, market, fn in tasks}
    for i, fut in enumerate(as_completed(futs), 1):
        ds, market = futs[fut]
        try:
            rows.extend(fut.result())
        except Exception as e:
            failures.append({"date": ds, "market": market, "error": str(e)})
        if i % 100 == 0 or i == len(futs):
            print(f"[PROGRESS] {i}/{len(futs)} rows={len(rows)} failures={len(failures)}", flush=True)

rows.sort(key=lambda r: (r["date"], r["market"], r["code"]))
fields = sorted({k for r in rows for k in r})
fields = ["date", "market", "code", "name"] + [x for x in fields if x not in {"date", "market", "code", "name"}]
csv_path = OUT / "official_margin_short_2021_2025.csv.gz"
pd.DataFrame(rows, columns=fields).to_csv(csv_path, index=False, compression="gzip")

coverage = {}
for market in ("TWSE", "TPEX"):
    have = {r["date"] for r in rows if r["market"] == market}
    missing = sorted(set(dates) - have)
    coverage[market] = {
        "dates_expected": len(dates), "dates_present": len(have),
        "ratio": len(have) / len(dates), "missing_dates": missing
    }

manifest = {
    "dataset": "AlphaPilot official margin and short balances 2021-2025",
    "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    "date_start": dates[0], "date_end": dates[-1], "trading_dates": len(dates),
    "rows": len(rows), "coverage": coverage, "failures": failures,
    "sources": {
        "TWSE": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
        "TPEX": "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
    },
    "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
if min(v["ratio"] for v in coverage.values()) < 0.98:
    raise RuntimeError(f"coverage below 98%: {coverage}")
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
