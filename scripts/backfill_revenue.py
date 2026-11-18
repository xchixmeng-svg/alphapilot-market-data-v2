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
    with urllib.request.urlopen(req, timeout=60) as r:
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
    tables = pd.read_html(io.StringIO(html))
    out = []
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        t.columns = cols
        code = next((c for c in cols if "公司代號" in c), None)
        cur = next((c for c in cols if "當月營收" in c and "累計" not in c), None)
        mom = next((c for c in cols if "上月比較" in c), None)
        yoy = next((c for c in cols if "去年同月" in c), None)
        if not (code and cur):
            continue
        d = pd.DataFrame({"code": t[code].astype(str).str.strip(), "revenue_thousand": num(t[cur]),
                          "mom_pct": num(t[mom]) if mom else None, "yoy_pct": num(t[yoy]) if yoy else None})
        out.append(d[d.code.str.fullmatch(r"\d{4}")])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(y0=2013, y1=None, sleep=3.0):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=8)))
    y1 = y1 or now.year
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    frames = [pd.read_csv(OUT, dtype={"code": str, "period": str, "available_date": str})] if OUT.exists() else []
    log = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            if (y, m) >= (now.year, now.month):
                break
            for mk in ("sii", "otc"):
                key = f"{y}{m:02d}{mk}"
                if key in done:
                    continue
                got = None
                for host in HOSTS:
                    rows = []
                    try:
                        for kind in (0, 1):
                            html = get(f"{host}/nas/t21/{mk}/t21sc03_{y - 1911}_{m}_{kind}.html")
                            df = parse(html)
                            if len(df):
                                rows.append(df)
                            time.sleep(sleep)
                    except Exception as e:  # noqa
                        log.append(f"{key} {host}: {type(e).__name__} {e}"[:200])
                        continue
                    if rows:
                        got = pd.concat(rows, ignore_index=True)
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
            if frames and m % 6 == 0:  # 每半年存一次進度
                save(frames, done)
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
