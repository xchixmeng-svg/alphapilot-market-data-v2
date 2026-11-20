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
    """台指期（TX）近月：日盤收盤、夜盤收盤。夜盤歸在下一個交易日，所以抓 day 之後幾天的資料。"""
    s = datetime.strptime(day, "%Y-%m-%d")
    data = {"down_type": "1", "commodity_id": "TX", "commodity_id2": "",
            "queryStartDate": s.strftime("%Y/%m/%d"), "queryEndDate": (s + timedelta(days=5)).strftime("%Y/%m/%d")}
    r = requests.post("https://www.taifex.com.tw/cht/3/futDataDown", data=data, headers=H, timeout=(15, 60))
    t = r.content.decode("big5", "replace")
    if "DOCTYPE" in t[:100]:
        raise RuntimeError("期交所回傳網頁不是資料")
    x = pd.read_csv(io.StringIO(t), index_col=False)
    x.columns = [str(c).strip() for c in x.columns]
    x = x[~x["到期月份(週別)"].astype(str).str.contains("/|W", regex=True)]
    x["收盤價"] = pd.to_numeric(x["收盤價"], errors="coerce")
    x = x.dropna(subset=["收盤價"])
    near = sorted(x["到期月份(週別)"].astype(str).str.strip().unique())[0]
    x = x[x["到期月份(週別)"].astype(str).str.strip() == near]
    day_rows = x[(x["交易時段"].astype(str).str.contains("一般")) & (pd.to_datetime(x["交易日期"]) == s)]
    nt_rows = x[x["交易時段"].astype(str).str.contains("盤後")]
    nt_rows = nt_rows[pd.to_datetime(nt_rows["交易日期"]) >= s]
    if day_rows.empty or nt_rows.empty:
        raise RuntimeError(f"日盤 {len(day_rows)} 筆、夜盤 {len(nt_rows)} 筆")
    d = float(day_rows["收盤價"].iloc[-1]); n = float(nt_rows.sort_values("交易日期")["收盤價"].iloc[0])
    return d, n


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
