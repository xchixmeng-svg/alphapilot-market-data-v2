"""抓 MOPS 重大訊息公告（標題+時間），用來讓 AI 判斷「為什麼是現在」。

來源：證交所／櫃買中心官方 OpenAPI，不用金鑰、不用登入、免費：
  上市 https://openapi.twse.com.tw/v1/opendata/t187ap04_L  （上市公司每日重大訊息）
  上櫃 https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O （上櫃公司每日重大訊息）

欄位對應改用「依序嘗試候選名稱」（見 ap/_pick.py），因為 2026-09-22 實測發現：
- TWSE 目前欄位已無「主旨」，公告內容改放在「說明」欄。
- TPEx 欄位是中英文混用（例如 SecuritiesCompanyCode 配中文「發言日期」「主旨」），
  不是單純的全中文或全英文組合，用固定候選組合比對很容易整組落空。

⚠ 重要限制：
1. 這是「當日滾動快照」，不是歷史資料庫——只能抓到最近的公告，抓不到你開始使用前的歷史。
   → 每天存一份快照，逐日累積成自己的歷史。
2. 「出表日期」皆為民國年格式。
3. 此腳本已依實測欄位修正，但官方格式仍可能再變動；找不到候選欄位時會清楚報錯並列出實際欄位。
"""
import json
import urllib.request

import pandas as pd

from ._pick import pick

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"
UA = {"User-Agent": "Mozilla/5.0"}


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _roc_to_date(s):  # 1150521 -> 2026-05-21（民國年轉西元年）
    s = str(s).zfill(7)
    return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:7]}"


def _hhmmss(t):
    t = str(t).zfill(6)
    return f"{t[:2]}:{t[2:4]}:{t[4:6]}"


def _parse_events(data, label):
    if not data:
        return pd.DataFrame(columns=["code", "published_at", "title"])
    df = pd.DataFrame(data)
    print(f"  [{label}] 實際欄位：{list(df.columns)}")
    code = pick(df, "公司代號", "SecuritiesCompanyCode", "Code")
    date = pick(df, "發言日期", "SpokesDate")
    time_ = pick(df, "發言時間", "SpokesTime")
    title = pick(df, "主旨", "說明", "Subject")
    out = pd.DataFrame({
        "code": code.astype(str).str.strip(),
        "published_at": [f"{_roc_to_date(d)} {_hhmmss(t)}" for d, t in zip(date, time_)],
        "title": title.astype(str).str.strip(),
    })
    print(f"  [{label}] 取得 {len(out)} 筆公告")
    return out


def fetch_twse_events():
    return _parse_events(_get_json(TWSE_URL), "TWSE")


def fetch_tpex_events():
    return _parse_events(_get_json(TPEX_URL), "TPEx")


def fetch_today_events():
    frames = []
    for name, fn in (("TWSE", fetch_twse_events), ("TPEx", fetch_tpex_events)):
        try:
            frames.append(fn())
        except Exception as e:  # noqa
            print(f"警告：{name} 抓取失敗，本次先跳過（{e}）")
    if not frames:
        raise RuntimeError("TWSE 與 TPEx 皆抓取失敗")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["code", "published_at", "title"])


if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime

    df = fetch_today_events()
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    path = f"data/events/{date}.csv"
    os.makedirs("data/events", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"寫入 {path}，共 {len(df)} 筆公告")
