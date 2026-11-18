"""抓每日估值（本益比、股價淨值比、殖利率）。

來源：證交所／櫃買中心官方 OpenAPI，不用金鑰、不用登入、免費：
  上市 https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL
  上櫃 https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis

欄位對應改用「依序嘗試候選名稱」（見 ap/_pick.py）。2026-09-22 實測：
- TWSE：Code / PEratio / PBratio / DividendYield（已確認可用）。
- TPEx：SecuritiesCompanyCode / PriceEarningRatio / PriceBookRatio / YieldRatio
  （殖利率欄位叫 YieldRatio，不是 DividendYield，已修正）。

⚠ 限制：
1. 跟價量、法人一樣是「當日快照」，只能從你開始抓的那天起逐日累積，抓不到過去歷史。
2. 本益比採最近四季財報、殖利率採近期股利、淨值比採最近一季每股淨值，三者更新頻率不同。
"""
import json
import urllib.request

import pandas as pd

from ._pick import pick

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
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


def _parse_valuation(data, label):
    if not data:
        return pd.DataFrame(columns=["code", "pe", "pb", "dy"])
    df = pd.DataFrame(data)
    print(f"  [{label}] 實際欄位：{list(df.columns)}")
    code = pick(df, "Code", "SecuritiesCompanyCode", "公司代號", "代號")
    pe = pick(df, "PEratio", "PriceEarningRatio", "本益比")
    pb = pick(df, "PBratio", "PriceBookRatio", "股價淨值比")
    dy = pick(df, "DividendYield", "YieldRatio", "殖利率")
    out = pd.DataFrame({
        "code": code.astype(str).str.strip(),
        "pe": pe.map(_num), "pb": pb.map(_num), "dy": dy.map(_num),
    })
    print(f"  [{label}] 取得 {len(out)} 檔")
    return out


def fetch_twse_valuation():
    return _parse_valuation(_get_json(TWSE_URL), "TWSE")


def fetch_tpex_valuation():
    return _parse_valuation(_get_json(TPEX_URL), "TPEx")


def fetch_today_valuation():
    frames = []
    for name, fn in (("TWSE", fetch_twse_valuation), ("TPEx", fetch_tpex_valuation)):
        try:
            frames.append(fn())
        except Exception as e:  # noqa
            print(f"警告：{name} 估值抓取失敗，本次先跳過（{e}）")
    if not frames:
        raise RuntimeError("TWSE 與 TPEx 估值皆抓取失敗")
    return pd.concat(frames, ignore_index=True).drop_duplicates("code")


if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime

    df = fetch_today_valuation()
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    path = f"data/valuation/{date}.csv"
    os.makedirs("data/valuation", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"寫入 {path}，共 {len(df)} 檔")
