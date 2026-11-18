"""集保戶股權分散表（TDCC 開放資料，每週一份，官方、免金鑰）。

來源：https://opendata.tdcc.com.tw/getOD.ashx?id=1-5
欄位：資料日期, 證券代號, 持股分級, 人數, 股數, 占集保庫存數比例%
持股分級 1～15 依持股數由小到大（15 = 1,000,001 股以上，即千張以上），16 = 差異數調整，17 = 合計。
只保留 4 位數代號（普通股）。官方只提供「最新一週」，所以每週要存一份，才會累積成歷史。
"""
import io
import urllib.request

import pandas as pd

URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch_tdcc():
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    print(f"  [TDCC] 實際欄位：{list(df.columns)}，{len(df)} 列")
    need = ["資料日期", "證券代號", "持股分級", "人數", "股數"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"TDCC 欄位改版，缺少 {miss}，實際欄位：{list(df.columns)}")
    pct = next((c for c in df.columns if "比例" in c), None)
    out = pd.DataFrame({
        "date": df["資料日期"].str.strip(),
        "code": df["證券代號"].str.strip(),
        "level": pd.to_numeric(df["持股分級"], errors="coerce"),
        "holders": pd.to_numeric(df["人數"].str.replace(",", ""), errors="coerce"),
        "shares": pd.to_numeric(df["股數"].str.replace(",", ""), errors="coerce"),
        "pct": pd.to_numeric(df[pct], errors="coerce") if pct else None,
    })
    out = out[out["code"].str.fullmatch(r"\d{4}")]
    t = out[(out.code == "2330") & (out.level == 15)]
    if t.empty or not (50 <= float(t.pct.iloc[0]) <= 100):
        raise RuntimeError(f"合理性檢查失敗：台積電千張大戶比例 {t.pct.tolist()}，預期 50～100%")
    return out
