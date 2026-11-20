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
    """台指期（TX）近月：日盤、夜盤最後成交價。來源：期交所開放資料 DailyMarketReportFut（最新一個交易日）。"""
    r = requests.get("https://openapi.taifex.com.tw/v1/DailyMarketReportFut", headers={"accept": "application/json", **H}, timeout=(15, 60))
    js = r.json()
    if not js:
        raise RuntimeError("開放資料是空的")
    print("期交所開放資料欄位：", list(js[0].keys()))
    x = pd.DataFrame(js)
    col = lambda *names: next(c for c in x.columns if any(n.lower() in c.lower() for n in names))
    cc, cm, cl, cs = col("Contract"), col("ContractMonth", "Month"), col("Last", "Close"), col("TradingSession", "Session")
    x = x[(x[cc].astype(str).str.strip() == "TX") & ~x[cm].astype(str).str.contains("/|W", regex=True)].copy()
    x[cl] = pd.to_numeric(x[cl], errors="coerce")
    x = x.dropna(subset=[cl])
    near = sorted(x[cm].astype(str).str.strip().unique())[0]
    x = x[x[cm].astype(str).str.strip() == near]
    ses = x[cs].astype(str)
    dr, nr = x[ses.str.contains("一般|Regular", regex=True)], x[ses.str.contains("盤後|After", regex=True)]
    print("TX 近月", near, x[[cs, cl]].to_dict("records"))
    if dr.empty or nr.empty:
        raise RuntimeError(f"日盤 {len(dr)} 筆、夜盤 {len(nr)} 筆")
    return float(dr[cl].iloc[0]), float(nr[cl].iloc[0])


def us_last():
    out = {}
    for k in ("VIX", "SOX", "SPX"):
        f = MAC / f"{k}.csv"
        if f.exists():
            x = pd.read_csv(f).dropna(subset=["close"])
            out[k] = (x.date.iloc[-1], float(x.close.iloc[-1]), float(x.close.iloc[-1] / x.close.iloc[-2] - 1) * 100)
    return out


def main():
    lj = json.loads((OUT / "latest.json").read_text())
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
    text = "\n".join(lines[:3] + extra + lines[3:])
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    res = dict(base, stage="morning", notify=True, text=text, morning_at=now.strftime("%Y-%m-%d %H:%M"))
    (OUT / "latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
