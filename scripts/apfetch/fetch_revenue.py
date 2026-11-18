"""抓每月營業收入（月營收），公司要求最重要的獲利先行指標。

來源：證交所／櫃買中心官方 OpenAPI，不用金鑰、不用登入、免費：
  上市 https://openapi.twse.com.tw/v1/opendata/t187ap05_L
       欄位（中文，已由實測與官方資料集頁確認）：
       出表日期, 資料年月, 公司代號, 公司名稱, 產業別,
       營業收入-當月營收, 營業收入-上月營收, 營業收入-去年當月營收,
       營業收入-上月比較增減(%), 營業收入-去年同月增減(%),
       累計營業收入-當月累計營收, 累計營業收入-去年累計營收, 累計營業收入-前期比較增減(%), 備註
  上櫃 https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O （對應你截圖「上櫃公司每月營業收入彙總表」）
       具體欄位名稱未實際連線核對過。

⚠ 重要限制：
1. 這是「當期快照」——回傳最新一期全市場資料，沒有可指定期間查詢的參數，抓不到過去月份。
   → 每月固定跑一次（例如每月 10 日後，多數公司已申報），逐月累積成自己的歷史。
2. 「出表日期」「資料年月」都是民國年格式（例如 11411 代表民國 114 年 11 月）。
3. 金額單位為新台幣千元，數值以字串回傳需自行轉型（此腳本已處理）。
4. 這是「公司申報日」的營收，不是「decision_date 當下已公開」的判斷——
   ap/features.py 用 available_date 篩選 available_date <= decision_date，
   available_date 用「出表日期」（申報公告日），這是 PIT 安全的做法。
5. 此腳本未實際連線測試過，欄位若改版會直接報錯而非默默出錯資料。
"""
import json
import urllib.request

import pandas as pd

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"
UA = {"User-Agent": "Mozilla/5.0"}


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _num(v):
    try:
        return float(str(v).replace(",", "").strip() or "nan")
    except ValueError:
        return float("nan")


def _roc_date_to_ymd(s):  # 1141217 -> 20251217（民國年轉西元年）
    s = str(s).zfill(7)
    return f"{int(s[:3]) + 1911}{s[3:5]}{s[5:7]}"


def _roc_ym_to_period(s):  # 11411 -> 202411（資料年月：民國年+月，月不一定補零）
    s = str(s)
    yy, mm = s[:-2], s[-2:]
    return f"{int(yy) + 1911}{mm.zfill(2)}"


def fetch_twse_revenue():
    data = _get_json(TWSE_URL)
    if not data:
        return pd.DataFrame(columns=["code", "period", "available_date", "revenue_thousand", "mom_pct", "yoy_pct"])
    df = pd.DataFrame(data)
    print(f"  [TWSE 月營收] 實際欄位：{list(df.columns)}")
    need = {"出表日期", "資料年月", "公司代號", "營業收入-當月營收", "營業收入-上月比較增減(%)", "營業收入-去年同月增減(%)"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"TWSE t187ap05_L 欄位已改版，缺少：{missing}，實際欄位：{list(df.columns)}")
    return pd.DataFrame({
        "code": df["公司代號"].astype(str).str.strip(),
        "period": df["資料年月"].map(_roc_ym_to_period),
        "available_date": df["出表日期"].map(_roc_date_to_ymd),
        "revenue_thousand": df["營業收入-當月營收"].map(_num),
        "mom_pct": df["營業收入-上月比較增減(%)"].map(_num),
        "yoy_pct": df["營業收入-去年同月增減(%)"].map(_num),
    })


def fetch_tpex_revenue():
    """欄位名稱未經實際連線核對，第一次執行若報錯，把『實際欄位』貼回來即可對照修正。"""
    data = _get_json(TPEX_URL)
    if not data:
        return pd.DataFrame(columns=["code", "period", "available_date", "revenue_thousand", "mom_pct", "yoy_pct"])
    df = pd.DataFrame(data)
    print(f"  [TPEx 月營收] 實際欄位：{list(df.columns)}")
    candidates = [
        {"code": "公司代號", "period": "資料年月", "date": "出表日期",
         "rev": "營業收入-當月營收", "mom": "營業收入-上月比較增減(%)", "yoy": "營業收入-去年同月增減(%)"},
    ]
    for c in candidates:
        if set(c.values()) <= set(df.columns):
            return pd.DataFrame({
                "code": df[c["code"]].astype(str).str.strip(),
                "period": df[c["period"]].map(_roc_ym_to_period),
                "available_date": df[c["date"]].map(_roc_date_to_ymd),
                "revenue_thousand": df[c["rev"]].map(_num),
                "mom_pct": df[c["mom"]].map(_num),
                "yoy_pct": df[c["yoy"]].map(_num),
            })
    raise RuntimeError(f"TPEx 月營收欄位不符合已知格式，實際欄位：{list(df.columns)}，"
                       f"請對照這份清單補一組到 fetch_tpex_revenue 的 candidates 裡")


def fetch_latest_revenue():
    frames = []
    for name, fn in (("TWSE", fetch_twse_revenue), ("TPEx", fetch_tpex_revenue)):
        try:
            frames.append(fn())
        except Exception as e:  # noqa
            print(f"警告：{name} 月營收抓取失敗，本次先跳過（{e}）")
    if not frames:
        raise RuntimeError("TWSE 與 TPEx 月營收皆抓取失敗")
    return pd.concat(frames, ignore_index=True).drop_duplicates("code")


if __name__ == "__main__":
    import os

    new = fetch_latest_revenue()
    new["available_date"] = new["available_date"].astype(str)
    new["period"] = new["period"].astype(str)
    path = "data/revenue.csv"
    if os.path.exists(path):
        # 強制欄位型態為文字，否則新舊資料合併後 available_date 一邊是數字一邊是字串，
        # sort_values 會直接報 TypeError（今天實際發生的錯誤，已修正）。
        old = pd.read_csv(path, dtype={"code": str, "period": str, "available_date": str})
        combo = pd.concat([old, new], ignore_index=True)
        combo = combo.sort_values("available_date").drop_duplicates(["code", "period"], keep="last")
    else:
        combo = new
    combo.to_csv(path, index=False)
    print(f"寫入 {path}，本次新抓 {len(new)} 檔，累積總筆數 {len(combo)}")
