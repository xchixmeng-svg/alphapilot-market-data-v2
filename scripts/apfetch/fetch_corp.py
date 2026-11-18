"""除權息事件（R10-MAX 還原價需要）。

- 2020～2025：沿用 R10-MAX 原本的官方事件檔 official_corporate_actions_2020_2025.csv（在資料 repo 或 data 資料夾找）
- 2026 起：證交所「除權除息計算結果表」
  https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate=YYYYMMDD&endDate=YYYYMMDD&response=json
  欄位：資料日期（民國）、股票代號、除權息前收盤價、除權息參考價、權值+息值、權/息
  「息」→ 現金股利 = 權值+息值；含「權」→ 現金股利記 0，配股比例由 R10-MAX 原本的方法
  以（前收盤 − 現金）÷ 參考價 推算（資金價值中性）。
上櫃事件目前沒有接入（見說明書限制）：上櫃股票除息日的還原價會有落差。
"""
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={s}&endDate={e}&response=json"
COLS = ["date", "code", "market", "event_type", "official_prev_close", "reference_price",
        "cash_dividend_per_share", "stock_shares_per_1000", "continuity_bridge", "source"]


# 分割不在除權息表內，手動補官方事件：0050 於 2025-06-18 以 1 拆 4 恢復交易（分割前收盤 188.65）
SPLITS = {"date": 20250618, "code": "0050", "market": "TWSE", "event_type": "分割", "official_prev_close": 188.65,
          "reference_price": 188.65 / 4, "cash_dividend_per_share": 0.0, "stock_shares_per_1000": 3000,
          "continuity_bridge": None, "source": "元大投信公告 1拆4"}


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _roc(s):
    m = re.match(r"(\d{2,3})年(\d{1,2})月(\d{1,2})日", str(s).strip())
    if not m:
        return None
    return int(f"{int(m.group(1)) + 1911}{int(m.group(2)):02d}{int(m.group(3)):02d}")


def fetch_twse(start, end):
    req = urllib.request.Request(URL.format(s=start, e=end), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        js = json.loads(r.read().decode("utf-8"))
    tables = [(t.get("fields"), t.get("data")) for t in js.get("tables", []) if isinstance(t, dict)]
    if js.get("fields"):
        tables.append((js["fields"], js.get("data")))
    for fields, rows in tables:
        if not fields or not rows:
            continue
        f = ["".join(str(x).split()) for x in fields]
        idx = lambda *ks: next((i for i, x in enumerate(f) if all(k in x for k in ks)), None)  # noqa
        i_d, i_c, i_p, i_r, i_v, i_t = idx("日期"), idx("代號"), idx("前收盤"), idx("參考價"), idx("權值", "息值"), idx("權/息")
        if None in (i_d, i_c, i_p, i_r, i_v, i_t):
            raise RuntimeError(f"除權息表格欄位對不上：{f}")
        out = []
        for row in rows:
            d, code, kind = _roc(row[i_d]), str(row[i_c]).strip(), str(row[i_t]).strip()
            prev, ref, val = _num(row[i_p]), _num(row[i_r]), _num(row[i_v])
            if not d or not re.fullmatch(r"\d{4}", code) or not ref:
                continue
            out.append({"date": d, "code": code, "market": "TWSE", "event_type": "息" if kind == "息" else "權" + kind,
                        "official_prev_close": prev, "reference_price": ref,
                        "cash_dividend_per_share": val if kind == "息" else 0.0,
                        "stock_shares_per_1000": None, "continuity_bridge": None, "source": "TWT49U"})
        return pd.DataFrame(out, columns=COLS)
    return pd.DataFrame(columns=COLS)


def load_events(store):
    """合併 2020～2025 官方事件檔與 2026 起抓到的事件。"""
    frames = []
    for p in (store.repo_root / "official_corporate_actions_2020_2025.csv",
              store.repo_root / "scripts" / "official_corporate_actions_2020_2025.csv",
              store.repo_root / "data" / "history" / "official_corporate_actions_2020_2025.csv",
              store.data / "official_corporate_actions_2020_2025.csv"):
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"code": str}))
            break
    alt = store.data / "official_corporate_actions_2020_2026.csv"  # research/fetch_corp_actions.py 產生（上市＋上櫃）
    if alt.exists():
        a = pd.read_csv(alt, dtype={"stock_id": str})
        a = a[(a["prev_close"] > 0) & (a["ref_price"] > 0)]
        frames.append(pd.DataFrame({
            "date": a["date"].astype(str).str.replace("-", "").astype(int), "code": a["stock_id"], "market": a["market"],
            "event_type": "息權", "official_prev_close": a["prev_close"], "reference_price": a["ref_price"],
            "cash_dividend_per_share": a["prev_close"] - a["ref_price"],  # 價值中性：配股也以等值現金計
            "stock_shares_per_1000": None, "continuity_bridge": None, "source": "TWT49U/TPEx全期"}))
    frames.append(pd.DataFrame([SPLITS]))
    f = store.data / "corporate_actions.csv"
    if f.exists():
        frames.append(pd.read_csv(f, dtype={"code": str}))
    if not frames:
        return pd.DataFrame(columns=COLS)
    ev = pd.concat(frames, ignore_index=True)
    ev["code"] = ev["code"].astype(str).str.zfill(4)
    ev["date"] = ev["date"].astype(int)
    return ev.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")


def update(store, start="20260101", end=None):
    from datetime import datetime, timedelta, timezone
    end = end or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    new = fetch_twse(start, end)
    f = store.data / "corporate_actions.csv"
    if f.exists():
        new = pd.concat([pd.read_csv(f, dtype={"code": str}), new], ignore_index=True)
    new["date"] = new["date"].astype(int)
    new = new.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
    new.to_csv(f, index=False)
    return len(new)
