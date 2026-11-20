#!/usr/bin/env python3
"""匯率、大盤、恐慌指數日資料，存 data/extra/macro/{name}.csv（date,close）。
來源：Yahoo Finance 日線（免金鑰；FRED 會擋 GitHub 機器，2026-11-19 實測連線逾時）。
  TWD    TWD=X   1 美元換多少台幣（數字變大 = 台幣貶值）
  TAIEX  ^TWII   加權指數
  VIX    ^VIX    美國 VIX 恐慌指數
  SOX    ^SOX    費城半導體指數
  US10Y  ^TNX    美國 10 年期公債殖利率（×1，單位 %）
每次整份重抓（Yahoo 偶爾修正舊值），舊檔有、新資料沒有的日期保留。任何一項失敗只記錄，不影響其他項。"""
import json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extra" / "macro"
SYMS = {"TWD": "TWD=X", "TAIEX": "^TWII", "VIX": "^VIX", "SOX": "^SOX", "US10Y": "^TNX"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}

def yahoo(sym):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
           f"?period1=946684800&period2={int(time.time())}&interval=1d&includeAdjustedClose=true")
    for host in ("query1", "query2"):
        try:
            req = urllib.request.Request(url.replace("query1", host), headers=UA)
            j = json.loads(urllib.request.urlopen(req, timeout=30).read())
            r = j["chart"]["result"][0]
            ts = r["timestamp"]; q = r["indicators"]["quote"][0]
            off = r["meta"].get("gmtoffset", 0)
            df = pd.DataFrame({"date": [datetime.fromtimestamp(t + off, timezone.utc).strftime("%Y-%m-%d") for t in ts],
                               "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]})
            return df.dropna(subset=["close"]).drop_duplicates("date", keep="last")
        except Exception as e:  # noqa
            err = e
    raise err

def main():
    import urllib.parse  # noqa
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, sym in SYMS.items():
        try:
            df = yahoo(sym)
            if len(df) < 1000:
                raise RuntimeError(f"只有 {len(df)} 筆")
            f = OUT / f"{name}.csv"
            if f.exists():   # Yahoo 跨日時偶爾少給最後一天：保留舊檔裡新資料沒有的日期
                old = pd.read_csv(f, dtype={"date": str})
                df = pd.concat([df, old[~old.date.isin(df.date)]]).sort_values("date").reset_index(drop=True)
            df.to_csv(f, index=False)
            ok += 1
            print(f"✅ {name} ({sym}): {len(df)} 筆 {df.date.iloc[0]}～{df.date.iloc[-1]} 最新 {df.close.iloc[-1]:.3f}")
        except Exception as e:  # noqa
            print(f"❌ {name} ({sym}): {e}")
        time.sleep(2)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    import urllib.parse
    main()
