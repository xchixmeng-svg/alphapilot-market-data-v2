#!/usr/bin/env python3
"""一次性補抓歷史月營收（公開資訊觀測站「每月營業收入彙總表」，官方、免費）。

網址：{host}/nas/t21/{sii|otc}/t21sc03_{民國年}_{月}_{0|1}.html   （0=國內公司、1=國外 KY 公司）
輸出：data/extra/revenue_history.csv
  code, period(YYYYMM), available_date, revenue_thousand, mom_pct, yoy_pct, market, source
available_date：觀測站彙總表沒有每家公司的申報日，一律用「次月 10 日」（法定申報期限）＝保守、不偷看未來。
可中斷續跑：已抓過的月份記在 data/extra/_revenue_backfill_done.json。
"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extra" / "revenue_history.csv"
DONE = ROOT / "data" / "extra" / "_revenue_backfill_done.json"
HOSTS = ["https://mopsov.twse.com.tw", "https://mops.twse.com.tw"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        b = r.read()
    for enc in ("big5", "cp950", "utf-8"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="replace")


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def parse(html):
    """每張產業表的資料列：代號, 名稱, 當月營收, 上月營收, 去年當月, 上月比較%, 去年同月%, 當月累計, 去年累計, 前期比較%, 備註
    （2026-11-19 在台灣網路打開 113 年 1 月上市頁實測：台積電一列正好 11 格，順序如上）。表頭有換行，所以用位置取值。"""
    out = []
    for t in pd.read_html(io.StringIO(html)):
        if t.shape[1] < 10:
            continue
        v = t.iloc[:, :7].copy()
        v.columns = ["code", "name", "cur", "prev", "ly", "mom", "yoy"]
        v["code"] = v["code"].astype(str).str.strip()
        v = v[v["code"].str.fullmatch(r"\d{4}")]
        if v.empty:
            continue
        out.append(pd.DataFrame({"code": v["code"], "revenue_thousand": num(v["cur"]),
                                 "mom_pct": num(v["mom"]), "yoy_pct": num(v["yoy"])}))
    df = pd.concat(out, ignore_index=True).drop_duplicates("code") if out else pd.DataFrame()
    return df


def main(y0=2013, y1=None, sleep=3.0):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=8)))
    y1 = y1 or now.year
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    frames = [pd.read_csv(OUT, dtype={"code": str, "period": str, "available_date": str})] if OUT.exists() else []
    log = []
    for y in range(y0, y1 + 1):
        if log and log[-1].startswith("前 20 次"):
            break
        for m in range(1, 13):
            if (y, m) >= (now.year, now.month):
                break
            for mk in ("sii", "otc"):
                key = f"{y}{m:02d}{mk}"
                if key in done:
                    continue
                got = None
                for host in HOSTS:
                    # _0 頁已「包含國內及國外公司」（2026-11-19 在台灣網路實測確認），不另抓 _1
                    try:
                        df = parse(get(f"{host}/nas/t21/{mk}/t21sc03_{y - 1911}_{m}_0.html"))
                    except Exception as e:  # noqa
                        log.append(f"{key} {host}: {type(e).__name__} {e}"[:200])
                        print(log[-1], flush=True)
                        time.sleep(sleep)
                        continue
                    time.sleep(sleep)
                    if len(df):
                        got = df
                        break
                if got is None or len(got) < 50:
                    log.append(f"{key}: 沒抓到（{0 if got is None else len(got)} 筆）")
                    print(log[-1], flush=True)
                    continue
                ny, nm = (y + (m == 12), 1 if m == 12 else m + 1)
                got["period"] = f"{y}{m:02d}"
                got["available_date"] = f"{ny}{nm:02d}10"
                got["market"] = "TWSE" if mk == "sii" else "TPEx"
                got["source"] = "mops_t21sc03"
                frames.append(got)
                done.add(key)
                print(f"{key}: {len(got)} 家", flush=True)
            if len(log) >= 20 and not done:  # 一開始連續失敗 20 次 → 多半是被擋，不要空等好幾小時
                log.append("前 20 次全部失敗，判定此環境連不到觀測站，提早結束")
                break
            if frames and m % 6 == 0:  # 每半年存一次進度
                save(frames, done)
    if frames:
        save(frames, done)
    (ROOT / "data" / "extra" / "_status").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "extra" / "_status" / "revenue_backfill.json").write_text(
        json.dumps({"months_done": len(done), "errors": log[-200:]}, ensure_ascii=False, indent=1), encoding="utf-8")


def save(frames, done):
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "period"], keep="last").sort_values(["period", "code"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["code", "period", "available_date", "revenue_thousand", "mom_pct", "yoy_pct", "market", "source"]].to_csv(OUT, index=False)
    DONE.write_text(json.dumps(sorted(done)))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
