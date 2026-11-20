#!/usr/bin/env python3
"""一次性補資料：期交所 2008～今
1) 台指期（TXF）三大法人未平倉（外資台指期淨未平倉）
2) 選擇權 Put/Call 比
原始 CSV 存 data/history/taifex/raw/{fut|pc}_{YYYYMM}.csv，已存在的跳過。"""
import sys, time
from datetime import date
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "history" / "taifex" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
     "Referer": "https://www.taifex.com.tw/cht/3/futContractsDate"}
y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2008
today = date.today()
months = [(y, m) for y in range(y0, today.year + 1) for m in range(1, 13) if date(y, m, 1) <= today]
URL = {"fut": "https://www.taifex.com.tw/cht/3/futContractsDateDown",
       "pc": "https://www.taifex.com.tw/cht/3/pcRatioDown"}
shown = set()
for y, m in months:
    end = date(y + (m == 12), m % 12 + 1, 1)
    last = min(today, date.fromordinal(end.toordinal() - 1))
    for kind, url in URL.items():
        f = OUT / f"{kind}_{y}{m:02d}.csv"
        if f.exists() and f.stat().st_size > 200 and (y, m) != (today.year, today.month):
            continue
        data = {"queryStartDate": f"{y}/{m:02d}/01", "queryEndDate": last.strftime("%Y/%m/%d")}
        if kind == "fut": data["commodityId"] = "TXF"
        for k in range(4):
            try:
                r = requests.post(url, data=data, headers=H, timeout=(15, 60))
                if r.status_code == 200 and len(r.content) > 200:
                    f.write_bytes(r.content)
                    if kind not in shown:
                        shown.add(kind); print(kind, "sample:\n", r.content.decode("big5", "replace")[:600])
                    break
                print(kind, y, m, "status", r.status_code, len(r.content))
            except Exception as e:  # noqa
                print(kind, y, m, "retry", e)
            time.sleep(5 * (k + 1))
        time.sleep(2)
    print(y, m, "ok", flush=True)
print("done")
