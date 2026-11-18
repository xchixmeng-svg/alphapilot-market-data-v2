"""抓官方產業分類（一次性，之後偶爾重跑更新即可，不用每天抓）。

來源：TWSE ISIN 查詢頁，這是市面上抓台股清單最常用的官方頁面之一：
  上市 https://isin.twse.com.tw/isin/C_public.jsp?strMode=2
  上櫃 https://isin.twse.com.tw/isin/C_public.jsp?strMode=4
回傳表格欄位固定為：有價證券代號及名稱, ISIN, 上市日, 市場別, 產業別, CFICode, 備註
只取「股票」區塊（表格內用整列粗體字當分節標題，例如「股票」「ETF」，需跳過這些分節列與非股票列）。
用官方頁面本身的分類文字，不對應到你 0B 資料的中文產業名（例如可能是「半導體業」而非「半導體」），
若兩邊名稱對不上，共用區塊裡的產業彙總會找不到對應股票，請人工核對一次兩邊的類別文字是否一致。
"""
import re
import time
import urllib.request
from io import StringIO

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
URLS = {"TWSE": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "TPEX": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"}


def _fetch_html(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("big5", errors="replace")


def _pick_data_table(tables, url):
    """頁面可能不只一個 <table>（例如版頭、廣告區塊），不能假設第一個就是資料表；
    改成挑列數最多的那個，並印出診斷資訊，方便第一次執行時對照真實結構。"""
    if not tables:
        raise RuntimeError(f"{url} 沒有解析到任何表格，頁面格式可能已改版")
    sizes = [(i, len(t), len(t.columns)) for i, t in enumerate(tables)]
    print(f"  [{url}] 共解析到 {len(tables)} 個表格，各表格(列數,欄數)：{[(n, c) for _, n, c in sizes]}")
    idx = max(sizes, key=lambda x: x[1])[0]
    df = tables[idx]
    print(f"  選用表格 #{idx}，欄位數 {len(df.columns)}，前 2 列原始內容：\n{df.head(2).to_string()}")
    return df


def _fetch_table(url):
    html = _fetch_html(url)
    # 用 StringIO 包起來，而不是直接傳原始字串：
    # pandas/lxml 在 Windows 上會把「很長的字串」誤判成檔案路徑去嘗試開檔，
    # 導致 OSError: Error reading file，包成緩衝區物件就不會誤判。
    tables = pd.read_html(StringIO(html))
    return _pick_data_table(tables, url)


def _split_code_name(s):
    """把「1101　台泥」這種字串切成 (代號, 名稱)。
    刻意不用 regex 的 \\s——診斷發現同一段 regex 在某些 Windows/Python 組合下，
    對全形空白（U+3000）判斷結果不一致（本機測試吃得到，使用者機器吃不到，原因不明）。
    改用 Python 內建 str.split() 沒有這個問題：它對任何 Unicode 空白字元的判斷是
    直接查字元本身的屬性，不經過 re 模組的內部規則表，更可靠。"""
    s = s.strip()
    parts = s.split(None, 1)  # 用任何一段連續空白切一次，str.split() 對全形空白一樣有效
    if len(parts) == 2 and parts[0].isdigit() and 4 <= len(parts[0]) <= 6:
        return parts[0], parts[1].strip()
    return None, None


def fetch_industry_map():
    frames = []
    for market, url in URLS.items():
        df = _fetch_table(url)
        if len(df.columns) != 7:
            raise RuntimeError(f"{url} 表格欄位數是 {len(df.columns)}，預期 7，頁面格式可能已改版，"
                               f"實際欄位：{list(df.columns)}")
        df.columns = ["code_name", "isin", "listed_date", "market_type", "industry", "cfi", "note"]
        mt = df["market_type"].astype(str).str.strip()
        print(f"  [{market}] market_type 唯一值：{sorted(mt.unique().tolist())}")
        df = df[mt.isin(["上市", "上櫃"])]
        split = df["code_name"].astype(str).map(_split_code_name)
        code = split.map(lambda t: t[0])
        name = split.map(lambda t: t[1])
        n_ok = code.notna().sum()
        print(f"  [{market}] market_type 篩選後 {len(df)} 列，代號格式符合的有 {n_ok} 列；"
             f"不符合的樣本：{df['code_name'][code.isna()].head(3).tolist()}")
        out = pd.DataFrame({"code": code, "name": name,
                            "industry": df["industry"].fillna("").astype(str).str.strip()})
        out = out[out["code"].notna() & (out["industry"] != "")]
        print(f"  [{market}] 最終取得 {len(out)} 檔")
        frames.append(out)
        time.sleep(2)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("code")
    if out.empty:
        raise RuntimeError("兩個市場都抓到 0 檔，請把上面印出的診斷內容整段貼給我，不要只看最後結果")
    return out[["code", "industry"]]


if __name__ == "__main__":
    import sys
    df = fetch_industry_map()
    path = sys.argv[1] if len(sys.argv) > 1 else "data/industry_map.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"寫入 {path}，共 {len(df)} 檔")
    print("產業類別數：", df["industry"].nunique())
    print(df["industry"].value_counts().head(10))
