#!/usr/bin/env python3
"""一次性補資料：2025-12 ～ 2026-08-25 每天上市＋上櫃收盤價與成交值（補「站上 20 日線比例」的缺口）。
輸出 data/history/prices_2026/{YYYYMMDD}.csv（trade_date, stock_id, close, trading_value），已存在的跳過。"""
import csv, io, sys, time
from pathlib import Path
import pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "history" / "prices_2026"
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2025-12-01", "2026-08-25")
tx = pd.read_csv(ROOT / "data/extra/macro/TAIEX.csv", parse_dates=["date"])
days = [d.strftime("%Y-%m-%d") for d in tx.date if start <= d.strftime("%Y-%m-%d") <= end]


def get(url, **kw):
    for k in range(4):
        try:
            return requests.get(url, headers=H, timeout=(15, 60), **kw)
        except Exception as e:  # noqa
            print("retry", url, e); time.sleep(10 * (k + 1))
    return None


ok = 0
for day in days:
    ymd = day.replace("-", ""); f = OUT / f"{ymd}.csv"
    if f.exists(): ok += 1; continue
    rows = []
    r = get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX", params={"date": ymd, "type": "ALLBUT0999", "response": "json"})
    try:
        for t in (r.json().get("tables") or []) if r is not None else []:
            fl = t.get("fields") or []
            if "證券代號" in fl and "收盤價" in fl and "成交金額" in fl:
                a, b, v = fl.index("證券代號"), fl.index("收盤價"), fl.index("成交金額")
                for x in t.get("data") or []:
                    try: rows.append((day, str(x[a]).strip(), float(str(x[b]).replace(",", "")), float(str(x[v]).replace(",", ""))))
                    except ValueError: pass
    except Exception as e:  # noqa
        print(day, "twse parse", e)
    time.sleep(3)
    roc = f"{int(ymd[:4]) - 1911:03d}/{ymd[4:6]}/{ymd[6:]}"
    r = get("https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes", params={"date": roc, "response": "csv"})
    try:
        lines = r.content.decode("cp950").splitlines() if r is not None else []
        if len(lines) > 3 and roc in lines[1]:
            for x in csv.DictReader(io.StringIO("\n".join(lines[2:]))):
                try:
                    rows.append((day, str(x.get("代號", "")).strip(), float(str(x.get("收盤", "")).replace(",", "")),
                                 float(str(x.get("成交金額(元)", x.get("成交金額", ""))).replace(",", ""))))
                except (ValueError, TypeError): pass
    except Exception as e:  # noqa
        print(day, "tpex parse", e)
    time.sleep(3)
    if len(rows) >= 1500:
        pd.DataFrame(rows, columns=["trade_date", "stock_id", "close", "trading_value"]).to_csv(f, index=False); ok += 1
        print(day, len(rows))
    else:
        print(day, "不足", len(rows))
print(f"完成 {ok}/{len(days)}")
