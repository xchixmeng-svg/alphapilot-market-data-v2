#!/usr/bin/env python3
"""每日快照（雲端）：公告、估值、月營收、股數、除權息、產業分類、集保大戶。

存放位置：data/extra/（檔名、格式跟 AlphaPilot 本機 data/ 底下完全相同，本機會直接複製過去用）
  data/extra/events/YYYYMMDD.csv       重大訊息
  data/extra/valuation/YYYYMMDD.csv    本益比／淨值比／殖利率
  data/extra/shares/YYYYMMDD.csv       已發行股數
  data/extra/revenue.csv               月營收（逐月累積，不覆蓋舊月份）
  data/extra/corporate_actions.csv     除權息事件（上市，2026 起）
  data/extra/industry_map.csv          產業分類（30 天更新一次）
  data/extra/tdcc/YYYYMMDD.csv.gz      集保戶股權分散表（每週，日期為 TDCC 資料日期）
  data/extra/_status/YYYYMMDD.json     本次每一項成功／失敗與筆數
日期 = 資料 repo 最新一個已完成的交易日（跟 data/YYYY-MM-DD/ 同一天），不是執行當下的日期。
任何一項失敗只記錄，不中止其他項目。
"""
import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
EXTRA = ROOT / "data" / "extra"
TZ = timezone(timedelta(hours=8))


def latest_trade_date():
    days = sorted(p.parent.name for p in (ROOT / "data").glob("20??-??-??/manifest.json"))
    if not days:
        raise SystemExit("data/ 底下沒有任何交易日")
    return days[-1].replace("-", "")


# 櫃買中心偶爾給 GitHub 機器不完整的憑證鏈（fetch_today_runner.py 也有同樣處理）：
# 只對 www.tpex.org.tw、只在憑證錯誤時，重試一次不驗憑證；其他網站一律正常驗證。
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlparse

_real_urlopen = urllib.request.urlopen


def _urlopen(req, *a, **k):
    try:
        return _real_urlopen(req, *a, **k)
    except (ssl.SSLError, urllib.error.URLError) as e:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        reason = getattr(e, "reason", e)
        if (urlparse(url).hostname or "").lower() != "www.tpex.org.tw" or not isinstance(reason, ssl.SSLError):
            raise
        print("[tls-fallback] TPEx 憑證鏈驗證失敗，改用不驗憑證重試一次", flush=True)
        k["context"] = ssl._create_unverified_context()
        return _real_urlopen(req, *a, **k)


urllib.request.urlopen = _urlopen


class _Shim:  # fetch_corp.update 只需要 .data
    def __init__(self, d):
        self.data = d


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else latest_trade_date()
    EXTRA.mkdir(parents=True, exist_ok=True)
    st = {"date": date, "run_at": datetime.now(TZ).isoformat(timespec="seconds"), "items": {}}

    def item(name, fn):
        t = time.time()
        try:
            n = fn()
            st["items"][name] = {"ok": True, "rows": n, "sec": round(time.time() - t, 1)}
            print(f"✅ {name}：{n}")
        except Exception as e:  # noqa
            st["items"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500],
                                 "trace": traceback.format_exc()[-1500:]}
            print(f"❌ {name}：{e}")

    def events():
        from apfetch.fetch_events import fetch_today_events
        (EXTRA / "events").mkdir(exist_ok=True)
        df = fetch_today_events(); df.to_csv(EXTRA / "events" / f"{date}.csv", index=False, encoding="utf-8-sig")
        return len(df)

    def valuation():
        from apfetch.fetch_valuation import fetch_today_valuation
        (EXTRA / "valuation").mkdir(exist_ok=True)
        df = fetch_today_valuation(); df.to_csv(EXTRA / "valuation" / f"{date}.csv", index=False, encoding="utf-8-sig")
        return len(df)

    def revenue():
        from apfetch.fetch_revenue import fetch_latest_revenue
        new = fetch_latest_revenue()
        new["available_date"], new["period"] = new["available_date"].astype(str), new["period"].astype(str)
        f = EXTRA / "revenue.csv"
        if f.exists():
            old = pd.read_csv(f, dtype={"code": str, "period": str, "available_date": str})
            new = pd.concat([old, new], ignore_index=True).sort_values("available_date").drop_duplicates(["code", "period"], keep="last")
        new.to_csv(f, index=False)
        return len(new)

    def shares():
        from apfetch.fetch_shares import fetch_shares
        (EXTRA / "shares").mkdir(exist_ok=True)
        df = fetch_shares(); df.to_csv(EXTRA / "shares" / f"{date}.csv", index=False)
        return len(df)

    def corp():
        from apfetch.fetch_corp import update
        return update(_Shim(EXTRA), start=f"{date[:4]}0101", end=date)

    def industry():
        f = EXTRA / "industry_map.csv"
        if f.exists() and (datetime.now(TZ).date() - datetime.strptime(
                json.loads((EXTRA / "_industry_updated.json").read_text())["date"], "%Y%m%d").date()).days < 30:
            return "未到 30 天，沿用"
        from apfetch.fetch_industry import fetch_industry_map
        df = fetch_industry_map(); df.to_csv(f, index=False, encoding="utf-8-sig")
        (EXTRA / "_industry_updated.json").write_text(json.dumps({"date": date}))
        return len(df)

    def tdcc():
        from apfetch.fetch_tdcc import fetch_tdcc
        df = fetch_tdcc()
        d = df["date"].iloc[0]
        (EXTRA / "tdcc").mkdir(exist_ok=True)
        p = EXTRA / "tdcc" / f"{d}.csv.gz"
        if p.exists():
            return f"{d} 已存在"
        df.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
        return f"{d}：{len(df)} 列"

    same_day = datetime.now(TZ).strftime("%Y%m%d") == date
    daily = (("公告", events), ("估值", valuation), ("股數", shares)) if same_day or len(sys.argv) > 1 else ()
    if not daily:
        st["note"] = "不是交易日當天執行（例如隔天早上補抓），公告／估值／股數是『當下快照』，不能標成前一天，這次跳過"
        print(st["note"])
    for name, fn in daily + (("月營收", revenue), ("除權息", corp), ("產業分類", industry), ("集保", tdcc)):
        item(name, fn)
        time.sleep(1)
    (EXTRA / "_status").mkdir(exist_ok=True)
    (EXTRA / "_status" / f"{date}.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(v["ok"] for v in st["items"].values())
    print(f"完成 {ok}/{len(st['items'])} 項")


if __name__ == "__main__":
    main()
