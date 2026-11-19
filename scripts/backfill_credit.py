#!/usr/bin/env python3
"""一次性補抓上市「融資融券」與「三大法人買賣金額」歷史（2008 起），給「找真底」研究用。

來源（證交所）：
  融資融券  https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=YYYYMMDD&selectType=ALL&response=json
            → 全市場信用交易統計（融資金額餘額，仟元）＋ 每檔融資／融券餘額（張）
  法人金額  https://www.twse.com.tw/rwd/zh/fund/BFI82U?type=day&dayDate=YYYYMMDD&response=json
            → 外資、投信、自營商當日買賣差額（元）
交易日：用 data/extra/macro/TAIEX.csv 的日期（有開盤才有資料，不用猜假日）。

先自我驗證（任何一項不符就中止、不寫檔）：
  2026-11-18 上市融資餘額應約 5,080.26 億元（新聞：財訊快報）
  2026-11-19 外資及陸資應賣超約 243.12 億元（新聞：證交所統計）

輸出（可中斷續跑，已抓過的日期跳過）：
  data/history/credit/market_{年}.csv     date, margin_bal_k(仟元), short_bal(張), foreign_net, trust_net, dealer_net(元)
  data/history/credit/stock_margin_{年}.parquet  date, code, margin_bal(張), short_bal(張)
用法：python scripts/backfill_credit.py 2008 2012
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "history" / "credit"
DBG = OUT / "_debug" / (sys.argv[1] if len(sys.argv) > 1 else "x")
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}
S = requests.Session()
PAUSE = 2.2


def get(url, params):
    err = None
    for i in range(6):
        try:
            r = S.get(url, params=params, headers=H, timeout=(15, 60))
            r.raise_for_status()
            js = r.json()
            time.sleep(PAUSE)
            return js
        except Exception as e:  # noqa
            err = e
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"{params} 連續失敗：{err}")


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return float("nan")


def tables(js):
    """新版 rwd JSON 是 tables:[{title,fields,data}]；舊格式可能是 creditList／data 等，全部攤平成 (fields, rows)。"""
    out = []
    for t in js.get("tables") or []:
        out.append((t.get("fields") or [], t.get("data") or [], t.get("title") or ""))
    for fk, dk in (("creditFields", "creditList"), ("fields", "data"), ("fields1", "data1"), ("fields2", "data2")):
        if dk in js:
            out.append((js.get(fk) or [], js.get(dk) or [], dk))
    return out


def margin(day):
    js = get("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN", {"date": day, "selectType": "ALL", "response": "json"})
    bal_k = short = float("nan")
    stocks = []
    for fields, rows, title in tables(js):
        for v in rows:
            if not isinstance(v, list) or not v:
                continue
            k = str(v[0]).strip()
            if k.startswith("融資金額"):
                bal_k = num(v[-1])          # 今日餘額（仟元）
            elif k.startswith("融券") and "交易單位" in k:
                short = num(v[-1])
            elif len(v) >= 13 and str(v[0]).strip()[:1].isalnum() and str(v[0]).strip() not in ("合計",):
                code = str(v[0]).strip()
                if 4 <= len(code) <= 6:
                    stocks.append((code, num(v[6]), num(v[12])))   # 融資今日餘額、融券今日餘額（張）
    return js, bal_k, short, stocks


def inst(day):
    js = get("https://www.twse.com.tw/rwd/zh/fund/BFI82U", {"type": "day", "dayDate": day, "response": "json"})
    f = t = dl = 0.0
    hit = False
    for fields, rows, title in tables(js):
        for v in rows:
            if not isinstance(v, list) or len(v) < 4:
                continue
            k = str(v[0]).strip()
            if k.startswith("合計"):
                continue
            hit = True
            if k.startswith("外資"):
                f += num(v[3])
            elif k.startswith("投信"):
                t += num(v[3])
            elif k.startswith("自營商"):
                dl += num(v[3])
    return js, (f if hit else float("nan")), (t if hit else float("nan")), (dl if hit else float("nan"))


def selfcheck():
    DBG.mkdir(parents=True, exist_ok=True)
    js, bal_k, short, stocks = margin("20261118")
    (DBG / "MI_MARGN_20261118.json").write_text(json.dumps(js, ensure_ascii=False)[:200000], encoding="utf-8")
    js2, f, t, dl = inst("20261119")
    (DBG / "BFI82U_20261119.json").write_text(json.dumps(js2, ensure_ascii=False), encoding="utf-8")
    m = {c: (a, b) for c, a, b in stocks}
    rep = {"融資餘額_億": bal_k / 1e5, "融券_張": short, "個股筆數": len(stocks), "00631L": m.get("00631L"),
           "外資_億": f / 1e8, "投信_億": t / 1e8, "自營_億": dl / 1e8}
    (DBG / "selfcheck.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("自我驗證：", rep, flush=True)
    ok = abs(bal_k / 1e5 - 5080.26) < 5 and abs(f / 1e8 + 243.12) < 3 and len(stocks) > 800
    if not ok:
        sys.exit("自我驗證失敗：欄位位置可能不同，原始 JSON 已存在 data/history/credit/_debug/，不寫任何資料")


def main(y0, y1):
    selfcheck()
    days = pd.read_csv(ROOT / "data" / "extra" / "macro" / "TAIEX.csv", usecols=["date"]).date
    days = sorted({d.replace("-", "") for d in days if str(y0) <= d[:4] <= str(y1)})
    OUT.mkdir(parents=True, exist_ok=True)
    for y in range(y0, y1 + 1):
        fm, fs = OUT / f"market_{y}.csv", OUT / f"stock_margin_{y}.parquet"
        mk = pd.read_csv(fm, dtype={"date": str}) if fm.exists() else pd.DataFrame()
        sk = pd.read_parquet(fs) if fs.exists() else pd.DataFrame()
        done = set(mk.date) if len(mk) else set()
        todo = [d for d in days if d[:4] == str(y) and d not in done]
        print(f"{y}：要抓 {len(todo)} 天", flush=True)
        newm, news = [], []
        for n, d in enumerate(todo, 1):
            try:
                _, bal_k, short, stocks = margin(d)
                _, f, t, dl = inst(d)
            except Exception as e:  # noqa
                print(f"{d} 失敗：{e}", flush=True)
                continue
            newm.append({"date": d, "margin_bal_k": bal_k, "short_bal": short, "foreign_net": f, "trust_net": t, "dealer_net": dl})
            news += [{"date": d, "code": c, "margin_bal": a, "short_bal": b} for c, a, b in stocks]
            if n % 40 == 0 or n == len(todo):
                mk = pd.concat([mk, pd.DataFrame(newm)], ignore_index=True)
                sk = pd.concat([sk, pd.DataFrame(news)], ignore_index=True)
                mk.to_csv(fm, index=False)
                sk.to_parquet(fs, index=False)
                newm, news = [], []
                print(f"  {y} 已存 {n}/{len(todo)}（最新 {d}）", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
