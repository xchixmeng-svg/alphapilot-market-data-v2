#!/usr/bin/env python3
"""一次性補抓上櫃三大法人歷史（2020～2025），補上 history/2020-2025 裡上櫃法人幾乎全缺的洞。

來源：櫃買中心 https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date=民國/月/日&response=json
欄位位置跟每日下載 scripts/fetch_today.py 完全相同：外資合計 8～10、投信 11～13、自營商合計 20～22（買進、賣出、買賣超股數）。
先自我驗證：用同一套解析抓 repo 已有的最新一天，和 data/YYYY-MM-DD/normalized/tpex_institutional.csv 逐檔比對，
吻合率低於 98% 就中止、不寫任何檔案。
輸出：data/history/tpex_institutional_2020_2025/tpex_inst_{年}.parquet
  欄位同 institutional_2020_2025.parquet：date(int64 YYYYMMDD), market, code, foreign_net, trust_net, dealer_net
可中斷續跑：已存在的年份檔會讀回來，已抓的日期跳過。
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import urllib3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "history" / "tpex_institutional_2020_2025"
URL = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}
S = requests.Session()


def get(day):
    p = {"type": "Daily", "sect": "EW", "date": f"{day.year - 1911:03d}/{day.month:02d}/{day.day:02d}", "id": "", "response": "json"}
    for i in range(5):
        try:
            try:
                r = S.get(URL, params=p, headers=HEADERS, timeout=(15, 60))
            except requests.exceptions.SSLError:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                r = S.get(URL, params=p, headers=HEADERS, timeout=(15, 60), verify=False)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa
            err = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"{day} 連續失敗：{err}")


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return float("nan")


def parse(js, day):
    t = (js.get("tables") or [{}])[0]
    rows = t.get("data") or []
    out = [{"date": int(day.strftime("%Y%m%d")), "market": "TPEX", "code": str(v[0]).strip(),
            "foreign_net": num(v[10]), "trust_net": num(v[13]), "dealer_net": num(v[22])}
           for v in rows if isinstance(v, list) and len(v) >= 24]
    return pd.DataFrame(out)


def selfcheck():
    days = sorted(p.parent.parent.name for p in (ROOT / "data").glob("20??-??-??/normalized/tpex_institutional.csv"))
    d = days[-1]
    ref = pd.read_csv(ROOT / "data" / d / "normalized" / "tpex_institutional.csv", dtype={"stock_id": str}, encoding="utf-8-sig")
    got = parse(get(date.fromisoformat(d)), date.fromisoformat(d))
    m = ref.merge(got, left_on="stock_id", right_on="code", suffixes=("_ref", ""))
    rate = {c: float((m[c + "_ref"].astype(float) == m[c]).mean()) for c in ("foreign_net", "trust_net", "dealer_net")}
    print(f"自我驗證 {d}：比對 {len(m)}/{len(ref)} 檔，吻合率 {rate}", flush=True)
    if len(m) < 0.95 * len(ref) or min(rate.values()) < 0.98:
        raise SystemExit(f"自我驗證失敗，中止不寫檔：{rate}")
    return {"date": d, "matched": len(m), "rate": rate}


def main(y0=2020, y1=2025, sleep=2.5):
    chk = selfcheck()
    o = pd.concat([pd.read_parquet(f, columns=["date", "code"]) for f in sorted((ROOT / "data/history/2020-2025").glob("ohlcv_*.parquet"))])
    tdays = sorted(o["date"].unique())
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for y in range(y0, y1 + 1):
        f = OUT / f"tpex_inst_{y}.parquet"
        have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        done = set(have["date"].unique()) if len(have) else set()
        frames = [have] if len(have) else []
        days = [d for d in tdays if d // 10000 == y and d not in done]
        for i, d in enumerate(days):
            day = date(d // 10000, d // 100 % 100, d % 100)
            try:
                df = parse(get(day), day)
                if len(df) < 300:
                    log.append(f"{d}: 只有 {len(df)} 筆")
                frames.append(df)
            except Exception as e:  # noqa
                log.append(f"{d}: {e}"[:200])
            time.sleep(sleep)
            if (i + 1) % 50 == 0 or i == len(days) - 1:
                pd.concat(frames, ignore_index=True).to_parquet(f, index=False)
                print(f"{y}: {i + 1}/{len(days)} 天", flush=True)
    st = ROOT / "data" / "extra" / "_status"
    st.mkdir(parents=True, exist_ok=True)
    (st / "tpex_inst_backfill.json").write_text(json.dumps({"selfcheck": chk, "errors": log[-300:]}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
