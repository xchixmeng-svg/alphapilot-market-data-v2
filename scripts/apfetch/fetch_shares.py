"""抓已發行普通股數，用來計算市值、選出市值前 500 檔。

來源（官方、免金鑰）：
  上市 https://openapi.twse.com.tw/v1/opendata/t187ap03_L   公開發行公司基本資料，欄位「已發行普通股數或TDR原股發行股數」
  上櫃 https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O 櫃買對應資料集
每天存一份 data/shares/YYYYMMDD.csv（code, shares），排名只用判斷日當天或之前最新的一份，不看未來。

欄位名稱以關鍵字辨識（官方欄名偶爾會變）；抓不到或數值不合理就報錯並列出實際欄位，不寫檔。
合理性檢查：台積電（2330）股數應在 200 億～300 億股之間。
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse(rows, label):
    if not rows:
        raise RuntimeError(f"{label}：官方沒有回傳資料")
    cols = list(rows[0].keys())
    code = next((c for c in cols if c in ("公司代號", "SecuritiesCompanyCode", "CompanyCode", "Code")), None) \
        or next((c for c in cols if "代號" in c or "Code" in c), None)
    share = next((c for c in cols if "已發行普通股數" in c), None) \
        or next((c for c in cols if "發行" in c and "股數" in c and "特別" not in c), None) \
        or next((c for c in cols if "IssueShares" in c or "IssuedShares" in c), None)
    if not code or not share:
        raise RuntimeError(f"{label}：找不到代號或股數欄位。實際欄位：{cols}")
    df = pd.DataFrame({"code": [str(r.get(code, "")).strip() for r in rows],
                       "shares": [_num(r.get(share)) for r in rows]})
    df = df[df["code"].str.fullmatch(r"\d{4}") & df["shares"].notna() & (df["shares"] > 0)]
    print(f"  {label}：{len(df)} 檔（代號欄「{code}」、股數欄「{share}」）")
    return df


def fetch_shares():
    frames = [_parse(_get(TWSE_URL), "上市")]
    try:
        frames.append(_parse(_get(TPEX_URL), "上櫃"))
    except Exception as e:  # noqa
        print(f"  ⚠ 上櫃股數抓取失敗：{e}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("code")
    tsmc = df.loc[df["code"] == "2330", "shares"]
    if tsmc.empty or not (2e10 <= tsmc.iloc[0] <= 3e10):
        raise RuntimeError(f"合理性檢查失敗：台積電股數 {tsmc.iloc[0] if len(tsmc) else '缺'}，預期 200～300 億股。可能單位不同（例如以千股計），已中止。")
    return df


def main(data_dir="data", date=None):
    date = date or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    df = fetch_shares()
    os.makedirs(f"{data_dir}/shares", exist_ok=True)
    path = f"{data_dir}/shares/{date}.csv"
    df.to_csv(path, index=False)
    print(f"寫入 {path}，共 {len(df)} 檔")
    return path


if __name__ == "__main__":
    main(date=sys.argv[1] if len(sys.argv) > 1 else None)
