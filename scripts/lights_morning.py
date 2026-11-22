#!/usr/bin/env python3
"""早上推播版：拿前一個交易日晚上的完整燈號（latest.json），加上台指期夜盤和昨晚美股，存回 latest.json（stage=morning）。
訊號不重算（規則用的是台股收盤前那一晚的美股，回測驗證過）；夜盤、昨晚美股只當參考。"""
import io, json, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extra" / "lights"
MAC = ROOT / "data" / "extra" / "macro"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
     "Referer": "https://www.taifex.com.tw/cht/3/futDailyMarketReport"}


def night_session(day):
    """台指期（TXF）近月夜盤最後成交價與參考價（日盤收盤）。來源：期交所行情網（mis.taifex.com.tw），盤後時段。"""
    body = {"MarketType": "1", "SymbolType": "F", "KindID": "1", "CID": "TXF", "ExpireMonth": "", "RowSize": "全部",
            "PageNo": "", "SortColumn": "", "AscDesc": "A"}
    r = requests.post("https://mis.taifex.com.tw/futures/api/getQuoteList", json=body,
                      headers={"Content-Type": "application/json", "Referer": "https://mis.taifex.com.tw/futures/", **H}, timeout=(15, 60))
    js = r.json()
    rows = (js.get("RtData") or {}).get("QuoteList") or []
    print("行情網筆數", len(rows), rows[:2])
    rows = [q for q in rows if str(q.get("CLastPrice", "")).strip() not in ("", "-", "0") and "/" not in str(q.get("DispCName", ""))]
    if not rows:
        raise RuntimeError("行情網沒有夜盤成交")
    q = rows[0]                                   # 第一筆是近月
    last = float(str(q["CLastPrice"]).replace(",", "")); ref = float(str(q["CRefPrice"]).replace(",", ""))
    print("夜盤", q.get("DispCName"), q.get("CDate"), q.get("CTime"), last, ref)
    return ref, last


def us_last():
    out = {}
    for k in ("VIX", "SOX", "SPX"):
        f = MAC / f"{k}.csv"
        if f.exists():
            x = pd.read_csv(f).dropna(subset=["close"])
            out[k] = (x.date.iloc[-1], float(x.close.iloc[-1]), float(x.close.iloc[-1] / x.close.iloc[-2] - 1) * 100)
    return out


def us_expected(now):
    """昨晚應有的美股收盤日（美東日期）＝台北今天的前一個平日（美股假日當天本來就不會有，06:45 後就不等）。"""
    d = now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def main():
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    lj = json.loads((OUT / "latest.json").read_text())
    exp = us_expected(now)
    vix_last = ""
    try:
        vix_last = str(pd.read_csv(MAC / "VIX.csv").dropna(subset=["close"]).date.iloc[-1])[:10]
    except Exception as e:  # noqa
        print("VIX 讀不到：", e)
    fresh = vix_last >= exp
    if lj.get("stage") == "morning" and str(lj.get("morning_at", ""))[:10] == now.strftime("%Y-%m-%d") and lj.get("us_fresh"):
        print("今天早上版已完成，不重跑"); return
    if not fresh and now.hour * 60 + now.minute < 6 * 60 + 45:
        print(f"VIX 最新 {vix_last}，還沒有 {exp} 收盤，等下一次"); return
    if lj.get("stage") == "morning":
        base = json.loads((OUT / f"{lj['date']}.json").read_text())   # 早上重跑：從前一晚的版本重做
    else:
        base = lj
    day = f"{base['date'][:4]}-{base['date'][4:6]}-{base['date'][6:]}"
    extra = []
    try:
        d, n = night_session(day)
        extra.append(f"夜盤　{n:,.0f}　比日盤 {(n / d - 1) * 100:+.1f}%")
    except Exception as e:  # noqa
        print("夜盤抓不到：", e); extra.append("夜盤　資料未取得")
    us = us_last()
    if us:
        parts = []
        if "VIX" in us: parts.append(f"VIX {us['VIX'][1]:.1f}")
        if "SOX" in us: parts.append(f"費半 {us['SOX'][2]:+.1f}%")
        if "SPX" in us: parts.append(f"S&P {us['SPX'][2]:+.1f}%")
        extra.append(f"昨晚美股（{us.get('VIX', us[list(us)[0]])[0][5:].replace('-', '/')}）　" + "　".join(parts))
    lines = base["text"].split("\n")
    at = next((k + 1 for k, x in enumerate(lines) if x.startswith("大盤 ")), 3)
    text = "\n".join(lines[:at] + extra + lines[at:])
    res = dict(base, stage="morning", notify=True, text=text, morning_at=now.strftime("%Y-%m-%d %H:%M"), us_fresh=bool(fresh))
    (OUT / "latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
