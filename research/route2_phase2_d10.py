#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaPilot Forward｜路線二驗證：從資料自己學規律（route2_validate.py）

用途：回答「每天只挑 1／3／5 檔時，勝率幾成、期望值多少、有沒有贏 0050」。
執行：在 alphapilot_forward 資料夾下  python research\\route2_validate.py
不修改、不匯入正在運作的系統；只讀資料，結果寫到 research\\route2_out\\。

預測目標（對應 plan-v2）：
  訊號日 T 收盤後判斷 → T+1 開盤買（不當沖，T+2 起才可能出場）
  買進後 30 個交易日內（T+2 ~ T+31）：
    盤中最高價先碰到 進場價×1.10            → 勝（預掛限價單，以 max(開盤, 目標價) 成交）
    收盤價先跌破 進場價 − 1.5×ATR20(T)      → 敗（次一交易日開盤賣出）
    都沒發生                                 → 第 30 天收盤後，次日開盤賣出（算敗）
  T+1 取消進場：停牌/無量、開盤鎖漲停、開盤已低於 收盤(T) − 1.5×ATR20
驗證：2020~2023 訓練、2024 調整、2025~ 測試；訓練與測試之間清除標籤會跨界的樣本（≥30 交易日空檔）
成本：手續費 0.0855%（買賣各一次）、交易稅 0.3%（賣）、滑價（預設每邊 0.5%，可用 --slip 調整）
價格還原：只用官方除權息檔，以「前向總報酬指數」還原（當天才知道的事件當天才套用，不看未來）
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ───────────────────────── 固定規則 ─────────────────────────
HOLD = 30            # 觀察交易日數
TP = 0.10            # 目標 +10%
SL_ATR = 1.5         # 停損 1.5×ATR20
UNIV_TOP = 500       # 每日候選檔數上限（以 20 日均成交額排序，代理市值）
UNIV_MIN_VALUE = 3e7 # 20 日均額 ≥ 3,000 萬
FEE = 0.000855
TAX = 0.003
TRAIN_END = "2024-01-01"   # 訓練 < 2024
TEST_START = "2025-01-01"  # 測試 ≥ 2025
PURGE = HOLD + 8           # 清除空檔（交易日），確保訓練標籤不跨入下一段

ALIAS = {
    "date": ["date", "trade_date", "日期", "資料日期", "datetime", "ex_date", "除權息日期",
             "除權除息日", "ex_dividend_date"],
    "stock_id": ["stock_id", "code", "symbol", "ticker", "sid", "證券代號", "股票代號", "代號",
                 "公司代號", "stock_no", "stockid"],
    "open": ["open", "開盤價", "open_price"],
    "high": ["high", "最高價", "max", "high_price"],
    "low": ["low", "最低價", "min", "low_price"],
    "close": ["close", "收盤價", "close_price"],
    "volume": ["volume", "成交股數", "trading_volume", "vol"],
    "value": ["trading_value", "成交金額", "turnover", "amount", "trading_money", "value"],
}
CORP = {
    "prev": ["prev_close", "official_prev_close", "除權息前收盤價", "before_price", "close_before", "前收盤價"],
    "ref": ["ref_price", "reference_price", "除權息參考價", "after_price", "參考價"],
    "drop": ["權值+息值", "權息值", "drop", "dividend_total"],
    "cash": ["cash_dividend", "現金股利", "息值", "cash_div"],
    "stock": ["stock_dividend", "股票股利", "stock_div", "無償配股率"],
}
REV = {
    "ym": ["資料年月", "year_month", "ym", "revenue_month_str", "yearmonth", "年月"],
    "rev": ["營業收入-當月營收", "當月營收", "revenue", "monthly_revenue", "營收"],
    "rev_ly": ["營業收入-去年當月營收", "去年當月營收", "revenue_last_year"],
    "ry": ["revenue_year"], "rm": ["revenue_month"],
}
VALU = {
    "pe": ["本益比", "pe", "per", "pe_ratio"],
    "pb": ["股價淨值比", "pb", "pbr", "pb_ratio"],
    "dy": ["殖利率(%)", "殖利率", "dividend_yield", "yield"],
}
SKIP_DIRS = (".git", "route2_out", "__pycache__", "venv", ".venv", "site-packages", "node_modules")


def log(msg=""):
    print(msg, flush=True)


# ───────────────────────── 讀檔與欄位辨識 ─────────────────────────
def pick_col(cols, names):
    low = {c: str(c).strip().lower() for c in cols}
    names = [n.lower() for n in names]
    for n in names:
        for c, l in low.items():
            if l == n:
                return c
    return None


def norm_cols(df):
    m, used = {}, set()
    for canon, names in ALIAS.items():
        c = pick_col([x for x in df.columns if x not in m], names)
        if c is not None and canon not in used:
            m[c] = canon
            used.add(canon)
    return df.rename(columns=m)


def parse_date(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.tz_localize(None).dt.normalize() if getattr(s.dt, "tz", None) else s.dt.normalize()
    x = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    roc = x.str.match(r"^\d{2,3}[/\-.]\d{1,2}[/\-.]\d{1,2}$")
    out = pd.to_datetime(x.where(~roc), errors="coerce")
    if roc.any():
        p = x[roc].str.split(r"[/\-.]", expand=True).astype(int)
        out.loc[roc] = pd.to_datetime(dict(year=p[0] + 1911, month=p[1], day=p[2]), errors="coerce")
    return out.dt.normalize()


def norm_id(s):
    x = s.astype(str).str.strip().str.replace(r"\.(TW|TWO)$", "", regex=True).str.replace(r"\.0$", "", regex=True)
    return x.where(~x.str.fullmatch(r"\d{1,3}"), x.str.zfill(4))


def to_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    x = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(x, errors="coerce")


def read_any(p, nrows=None):
    try:
        if p.lower().endswith(".parquet"):
            return pd.read_parquet(p)
        for enc in ("utf-8-sig", "cp950", "big5", "utf-8"):
            try:
                return pd.read_csv(p, encoding=enc, dtype=str, nrows=nrows, low_memory=False)
            except UnicodeDecodeError:
                continue
    except Exception as e:
        log(f"   讀取失敗 {p}: {e}")
    return None


def header_cols(p):
    if p.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            return list(pq.read_schema(p).names)
        except Exception:
            df = read_any(p)
            return list(df.columns) if df is not None else []
    df = read_any(p, nrows=3)
    return list(df.columns) if df is not None else []


def _clean(l):
    return re.sub(r"[\(（].*?[\)）]", "", str(l).lower())


def detect_inst(cols):
    """回傳 {'foreign': ('net',col) 或 ('bs',buy,sell), 'trust':..., 'dealer':...} 或 {'long': True}"""
    low = {c: str(c).lower() for c in cols}
    if {"name", "buy", "sell"} <= set(low.values()):
        return {"long": True}
    cat_keys = {
        "foreign": (("foreign", "外資", "外陸資"), ("自營", "dealer")),
        "trust": (("trust", "投信"), ()),
        "dealer": (("dealer", "自營"), ("foreign", "外資", "外陸資")),
    }
    out = {}
    for cat, (keys, excl) in cat_keys.items():
        cands = [c for c in cols if c not in ("date", "stock_id")
                 and any(k in _clean(c) for k in keys) and not any(e in _clean(c) for e in excl)]
        nets = [c for c in cands if any(k in low[c] for k in ("net", "買賣超", "diff", "buy_sell"))]
        if nets:
            out[cat] = ("net", sorted(nets, key=lambda c: len(str(c)))[0])
            continue
        buys = [c for c in cands if any(k in low[c] for k in ("buy", "買進"))]
        sells = [c for c in cands if any(k in low[c] for k in ("sell", "賣出"))]
        if buys and sells:
            out[cat] = ("bs", sorted(buys, key=lambda c: len(str(c)))[0], sorted(sells, key=lambda c: len(str(c)))[0])
    return out


def inst_frame(df, spec):
    df = df.copy()
    df["date"] = parse_date(df["date"])
    df["stock_id"] = norm_id(df["stock_id"])
    if spec.get("long"):
        name_col = [c for c in df.columns if str(c).lower() == "name"][0]
        buy = [c for c in df.columns if str(c).lower() == "buy"][0]
        sell = [c for c in df.columns if str(c).lower() == "sell"][0]
        df["net"] = to_num(df[buy]) - to_num(df[sell])
        n = df[name_col].astype(str).str.lower()
        df["cat"] = np.where(n.str.contains("foreign") & ~n.str.contains("dealer"), "foreign",
                     np.where(n.str.contains("trust"), "trust",
                     np.where(n.str.contains("dealer") & ~n.str.contains("foreign"), "dealer", "")))
        df = df[df["cat"] != ""]
        w = df.pivot_table(index=["date", "stock_id"], columns="cat", values="net", aggfunc="sum").reset_index()
        return w
    res = df[["date", "stock_id"]].copy()
    for cat, sp in spec.items():
        if sp[0] == "net":
            res[cat] = to_num(df[sp[1]])
        else:
            res[cat] = to_num(df[sp[1]]) - to_num(df[sp[2]])
    return res


def discover(root, market_repo, extra):
    dirs = [os.path.join(root, "data")]
    if market_repo and os.path.isdir(market_repo):
        dirs.append(market_repo)
    files = []
    for d in dirs:
        for ext in ("*.parquet", "*.csv"):
            for p in glob.glob(os.path.join(d, "**", ext), recursive=True):
                if any(s in p.replace("\\", "/").split("/") for s in SKIP_DIRS):
                    continue
                files.append(p)
    for g in extra or []:
        files += glob.glob(g, recursive=True)
    # 優先序：data/history → 資料 repo → 其他
    def prio(p):
        q = p.replace("\\", "/").lower()
        if "/data/history/" in q:
            return 0
        if market_repo and os.path.abspath(p).startswith(os.path.abspath(market_repo)):
            return 1
        return 2
    files = sorted(set(files), key=lambda p: (prio(p), p))
    return files, prio


def load_all(root, market_repo, extra):
    files, prio = discover(root, market_repo, extra)
    oh, ins, rev, val, corp = [], [], [], [], None
    daily_sum, bad_foreign = {}, []
    log(f"── 掃描 {len(files)} 個資料檔 ──")
    for p in files:
        name = os.path.basename(p).lower()
        if "corporate_action" in name:
            if corp is None:
                corp = p
            continue
        cols = header_cols(p)
        ncols = norm_cols(pd.DataFrame(columns=cols)).columns
        has = set(ncols)
        mdate = re.search(r"(20\d{2}-\d{2}-\d{2})", p.replace("\\", "/"))
        path_date = mdate.group(1) if (mdate and "date" not in has) else None
        if path_date:
            has.add("date")   # 資料 repo 每日資料夾 data/YYYY-MM-DD/normalized/*.csv：日期取自資料夾名
        is_oh = {"date", "stock_id", "open", "high", "low", "close", "volume"} <= has
        ispec = detect_inst([c for c in ncols]) if {"date", "stock_id"} <= has else {}
        is_rev = "stock_id" in has and pick_col(cols, REV["rev"]) is not None and (
            pick_col(cols, REV["ym"]) is not None or pick_col(cols, REV["ry"]) is not None)
        is_val = {"date", "stock_id"} <= has and (pick_col(cols, VALU["pe"]) or pick_col(cols, VALU["pb"]))
        if not (is_oh or ispec or is_rev or is_val):
            continue
        df = read_any(p)
        if df is None or len(df) == 0:
            continue
        df = norm_cols(df)
        if path_date:
            df["date"] = path_date
        tag = []
        if is_oh:
            d = df[["date", "stock_id", "open", "high", "low", "close", "volume"]
                   + (["value"] if "value" in df.columns else [])].copy()
            d["date"] = parse_date(d["date"])
            d["stock_id"] = norm_id(d["stock_id"])
            for c in d.columns[2:]:
                d[c] = to_num(d[c])
            d["_prio"] = prio(p)
            oh.append(d.dropna(subset=["date"]))
            tag.append("價量")
        if ispec:
            try:
                d = inst_frame(df, ispec)
                for cat in ("foreign", "trust", "dealer"):
                    if cat in d.columns and d[cat].notna().sum() > 100 and (d[cat] < 0).mean() < 0.02:
                        bad_foreign.append((os.path.basename(p), cat, path_date or ""))
                        d[cat] = np.nan
                d["_prio"] = prio(p)
                ins.append(d.dropna(subset=["date"]))
                tag.append("法人(" + ",".join(k for k in ispec if k != "long") + ("long" if ispec.get("long") else "") + ")")
            except Exception as e:
                log(f"   法人欄位解析失敗 {p}: {e}")
        if is_rev:
            rev.append(df)
            tag.append("營收")
        if is_val:
            d = df.copy()
            d["date"] = parse_date(d["date"])
            d["stock_id"] = norm_id(d["stock_id"])
            for k, names in VALU.items():
                c = pick_col(d.columns, names)
                d[k] = to_num(d[c]) if c is not None else np.nan
            val.append(d[["date", "stock_id", "pe", "pb", "dy"]].dropna(subset=["date"]))
            tag.append("估值")
        dd = df["date"] if "date" in df.columns else None
        rng = ""
        if dd is not None:
            pdd = parse_date(dd)
            rng = f"{pdd.min():%Y-%m-%d}~{pdd.max():%Y-%m-%d}" if pdd.notna().any() else ""
        if path_date:
            k = os.path.basename(p) + " [" + "/".join(tag) + "]"
            a = daily_sum.setdefault(k, [0, path_date, path_date])
            a[0] += 1; a[1] = min(a[1], path_date); a[2] = max(a[2], path_date)
        else:
            log(f"   ✓ {os.path.relpath(p, root)}  [{'/'.join(tag)}] {len(df):,} 列 {rng}")
    for k, (n, a, b) in sorted(daily_sum.items()):
        log(f"   ✓ 每日資料夾 {k}：{n} 天，{a} ~ {b}")
    if bad_foreign:
        g = pd.DataFrame(bad_foreign, columns=["f", "cat", "d"]).groupby(["f", "cat"])["d"].agg(["count", "min", "max"])
        for (f, cat), r in g.iterrows():
            log(f"   ⚠ {f} 的 {cat} 欄幾乎沒有負值（不像買賣超，疑似買進量）：{r['count']} 個檔案已捨棄該欄"
                + (f"（{r['min']} ~ {r['max']}）" if r['min'] else ""))
    return oh, ins, rev, val, corp


# ───────────────────────── 除權息還原 ─────────────────────────
def corp_ratio(path, C, dates, ids):
    R = np.ones(C.shape)
    stats = {"file": path, "applied": 0, "skipped": 0, "by_year": {}, "has_0050": False}
    if not path:
        return R, stats
    df = read_any(path)
    if df is None:
        return R, stats
    df = norm_cols(df)
    if "date" not in df.columns or "stock_id" not in df.columns:
        stats["error"] = "找不到日期或代號欄"
        return R, stats
    df["date"] = parse_date(df["date"])
    df["stock_id"] = norm_id(df["stock_id"])
    g = {k: (to_num(df[c]) if (c := pick_col(df.columns, v)) is not None else None) for k, v in CORP.items()}
    id_pos = {s: j for j, s in enumerate(ids)}
    Cprev = pd.DataFrame(C).ffill().shift(1).values
    for i in range(len(df)):
        s = df["stock_id"].iat[i]
        j = id_pos.get(s)
        dt = df["date"].iat[i]
        if j is None or pd.isna(dt):
            continue
        t = int(np.searchsorted(dates, np.datetime64(dt)))
        if t >= len(dates) or t == 0:
            continue
        prev = g["prev"].iat[i] if g["prev"] is not None else np.nan
        if not (prev > 0):
            prev = Cprev[t, j]
        ref = g["ref"].iat[i] if g["ref"] is not None else np.nan
        if not (ref > 0) and g["drop"] is not None and g["drop"].iat[i] == g["drop"].iat[i]:
            ref = prev - g["drop"].iat[i]
        if not (ref > 0) and (g["cash"] is not None or g["stock"] is not None):
            cash = g["cash"].iat[i] if g["cash"] is not None else 0.0
            stk = g["stock"].iat[i] if g["stock"] is not None else 0.0
            cash = 0.0 if cash != cash else cash
            stk = 0.0 if stk != stk else stk
            stk = stk / 10.0 if stk > 0.5 else stk
            ref = (prev - cash) / (1 + stk)
        r = ref / prev if (prev > 0 and ref > 0) else np.nan
        # Official 0050 4-for-1 split on 2025-06-18 has ref/prev = 47/188 = 0.25.
        # The original generic 0.3 lower guard incorrectly rejected that valid official event,
        # creating a fake ~75% benchmark crash and corrupting 0050-derived market features.
        valid_ratio = (0.3 < r < 1.5) or (s == "0050" and abs(r - 0.25) < 1e-6)
        if not valid_ratio or abs(r - 1) < 1e-9:
            stats["skipped"] += 1
            continue
        R[t, j] *= r
        stats["applied"] += 1
        y = str(pd.Timestamp(dates[t]).year)
        stats["by_year"][y] = stats["by_year"].get(y, 0) + 1
        if s == "0050":
            stats["has_0050"] = True
    return R, stats


def build_index(C, R):
    n, m = C.shape
    idx = np.full((n, m), np.nan)
    last = np.full(m, np.nan)
    cur = np.full(m, np.nan)
    pend = np.ones(m)
    for t in range(n):
        c = C[t]
        pend *= R[t]
        ok = ~np.isnan(c) & (c > 0)
        first = ok & np.isnan(last)
        cur[first] = c[first]
        cont = ok & ~np.isnan(last)
        cur[cont] = cur[cont] * c[cont] / (last[cont] * pend[cont])
        pend[ok] = 1.0
        last[ok] = c[ok]
        idx[t] = np.where(ok, cur, np.nan)
    return idx


# ───────────────────────── 營收、估值（有歷史才會被用上） ─────────────────────────
def rev_table(rev_list):
    if not rev_list:
        return None
    rows = []
    for df in rev_list:
        d = df.copy()
        d["stock_id"] = norm_id(d["stock_id"])
        rc = pick_col(d.columns, REV["rev"])
        d["rev"] = to_num(d[rc])
        lc = pick_col(d.columns, REV["rev_ly"])
        d["rev_ly"] = to_num(d[lc]) if lc else np.nan
        yc, mc, ymc = pick_col(d.columns, REV["ry"]), pick_col(d.columns, REV["rm"]), pick_col(d.columns, REV["ym"])
        if yc and mc:
            y, m = to_num(d[yc]), to_num(d[mc])
        else:
            s = d[ymc].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            parts = s.str.extract(r"^(\d{2,4})[/\-]?(\d{1,2})$")
            y, m = to_num(parts[0]), to_num(parts[1])
        y = np.where(y < 1000, y + 1911, y)
        d["ym"] = pd.to_datetime(dict(year=pd.Series(y, index=d.index), month=m, day=1), errors="coerce")
        rows.append(d[["stock_id", "ym", "rev", "rev_ly"]].dropna(subset=["ym", "rev"]))
    r = pd.concat(rows).drop_duplicates(["stock_id", "ym"], keep="last").sort_values(["stock_id", "ym"])
    r["avail"] = r["ym"] + pd.DateOffset(months=1) + pd.Timedelta(days=10)  # 次月 11 日才視為可用
    prev_y = r[["stock_id", "ym", "rev"]].copy()
    prev_y["ym"] = prev_y["ym"] + pd.DateOffset(years=1)
    r = r.merge(prev_y.rename(columns={"rev": "rev_ly2"}), on=["stock_id", "ym"], how="left")
    r["rev_ly"] = r["rev_ly"].fillna(r["rev_ly2"])
    r["rev_yoy"] = r["rev"] / r["rev_ly"] - 1
    g = r.groupby("stock_id")
    r["rev_mom"] = r["rev"] / g["rev"].shift(1) - 1
    r["rev_yoy3"] = g["rev_yoy"].transform(lambda x: x.rolling(3, min_periods=2).mean())
    r["rev_vs12"] = r["rev"] / g["rev"].transform(lambda x: x.rolling(12, min_periods=6).mean()) - 1
    return r[["stock_id", "avail", "rev_yoy", "rev_mom", "rev_yoy3", "rev_vs12"]].replace([np.inf, -np.inf], np.nan)


# ───────────────────────── 結果計算（標籤＝交易結果） ─────────────────────────
def fshift(X, k):
    out = np.full_like(X, np.nan)
    if k < X.shape[0]:
        out[: X.shape[0] - k] = X[k:]
    return out


def outcomes(aO, aH, aC, rO, rH, rC, V, atr, slip):
    n, m = aO.shape
    entry = fshift(aO, 1)
    e_rO, e_rH, e_V = fshift(rO, 1), fshift(rH, 1), fshift(V, 1)
    locked = (e_rO >= e_rH - 1e-9) & (e_rO >= rC * 1.095)
    gapdown = entry < (aC - SL_ATR * atr)
    valid = ~np.isnan(entry) & (e_V > 0) & ~locked & ~gapdown & ~np.isnan(atr) & (atr > 0)
    target = entry * (1 + TP)
    stop = entry - SL_ATR * atr
    status = np.zeros((n, m), np.int8)      # 0 未結 1 勝 2 停損 3 到期
    pend = np.zeros((n, m), np.int8)
    exit_px = np.full((n, m), np.nan)
    exit_off = np.full((n, m), -1, np.int32)
    for k in range(2, HOLD + 12):
        O, Hh, Cc = fshift(aO, k), fshift(aH, k), fshift(aC, k)
        act = valid & (status == 0)
        ex = act & (pend > 0) & ~np.isnan(O)
        exit_px[ex], exit_off[ex], status[ex] = O[ex], k, pend[ex]
        if k <= HOLD + 1:
            a2 = act & (pend == 0)
            hit = a2 & (Hh >= target)
            exit_px[hit] = np.maximum(O[hit], target[hit]) if hit.any() else exit_px[hit]
            exit_off[hit], status[hit] = k, 1
            still = a2 & ~hit
            sl = still & (Cc < stop)
            pend[sl] = 2
            if k == HOLD + 1:
                pend[still & ~sl] = 3
    gross = exit_px / entry
    net = gross * (1 - slip) * (1 - FEE - TAX) / ((1 + slip) * (1 + FEE)) - 1
    return valid, status, exit_off, exit_px, entry, net


# ───────────────────────── 投組模擬 ─────────────────────────
def simulate(sel, N, d, j, valid, status, exit_off, exit_px, entry, Cff, t0, t1, slots, slip, cost=None):
    bs, ss, fee = cost if cost else (slip, slip, FEE)
    entries = {}
    for q in sel:
        if valid[q] and d[q] + 1 <= t1:
            entries.setdefault(d[q] + 1, []).append(q)
    cash, pos, eq, trades, expo = 1.0, {}, [], [], []
    last_eq = 1.0
    for t in range(t0, t1 + 1):
        for s in [s for s, p in pos.items() if p["xt"] == t]:
            p = pos.pop(s)
            cash += p["sh"] * p["xp"] * (1 - ss) * (1 - fee - TAX)
        size = last_eq / (N * slots)
        for q in entries.get(t, []):
            s = j[q]
            if s in pos:
                continue
            amt = min(size, cash)
            if amt < size * 0.5:
                continue
            sh = amt / (entry[q] * (1 + bs) * (1 + fee))
            cash -= amt
            xt = d[q] + exit_off[q] if status[q] > 0 else -1
            pos[s] = {"sh": sh, "xt": xt, "xp": exit_px[q]}
            trades.append(q)
        mv = sum(p["sh"] * Cff[t, s] for s, p in pos.items())
        last_eq = cash + mv
        eq.append(last_eq)
        expo.append(mv / last_eq if last_eq > 0 else 0)
    return np.array(eq), np.array(trades, dtype=np.int64), float(np.mean(expo))


def curve_stats(eq, dts):
    eq = np.asarray(eq, float)
    tot = eq[-1] / eq[0] - 1
    yrs = max((dts[-1] - dts[0]).days / 365.25, 1e-9)
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if eq[-1] > 0 else -1
    mdd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
    return {"total": float(tot), "cagr": float(cagr), "mdd": mdd}


def daily_top(d, score, N, mask=None):
    idx = np.arange(len(d)) if mask is None else np.nonzero(mask)[0]
    o = idx[np.lexsort((-score[idx], d[idx]))]
    ds = d[o]
    rank = np.arange(len(o)) - np.searchsorted(ds, ds, side="left")
    return o[rank < N]


def trade_stats(sel, valid, status, net, exit_off):
    q = sel[valid[sel] & (status[sel] > 0)]
    if len(q) == 0:
        return {"n": 0, "win": np.nan, "ev": np.nan, "med": np.nan, "avg_win": np.nan,
                "avg_loss": np.nan, "hold": np.nan, "cancel": np.nan}
    r = net[q]
    w = status[q] == 1
    return {"n": int(len(q)), "win": float(w.mean()), "ev": float(r.mean()), "med": float(np.median(r)),
            "avg_win": float(r[w].mean()) if w.any() else np.nan,
            "avg_loss": float(r[~w].mean()) if (~w).any() else np.nan,
            "hold": float(exit_off[q].mean() - 1),
            "cancel": float(1 - valid[sel].mean())}


# ───────────────────────── 主程式 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="AlphaPilot 路線二驗證")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--root", default=os.path.dirname(here) if os.path.basename(here) == "research" else here)
    ap.add_argument("--market-repo", default=None, help="預設 ..\\alphapilot-market-data-v2")
    ap.add_argument("--extra", nargs="*", default=[], help="額外資料檔 glob")
    ap.add_argument("--corp-actions", default=None, help="官方除權息檔（預設自動尋找 official_corporate_actions*.csv）")
    ap.add_argument("--r10-equity", default=None, help="R10-MAX 每日淨值 CSV（date, equity），有就一起比較")
    ap.add_argument("--top", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--slots", type=int, default=10, help="每檔推薦的資金格數：每筆投入 = 淨值 /(N×slots)")
    ap.add_argument("--slip", type=float, default=0.005, help="每邊滑價（預設 0.5%%；若 R10-MAX 是來回 0.5%% 就設 0.0025）")
    ap.add_argument("--seeds", type=int, default=20, help="隨機挑選對照的次數")
    ap.add_argument("--quick", action="store_true", help="只試 2 組參數（快速檢查流程用）")
    ap.add_argument("--check", action="store_true", help="只檢查資料載入，不訓練、不看測試期")
    args = ap.parse_args()

    T0 = time.time()
    root = os.path.abspath(args.root)
    mrepo = args.market_repo or os.path.join(os.path.dirname(root), "alphapilot-market-data-v2")
    out_root = os.path.join(root, "research", "route2_out")
    run_dir = os.path.join(out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log("═" * 64)
    log(" AlphaPilot 路線二驗證｜從資料自己學規律")
    log(f" 系統資料夾：{root}")
    log(f" 資料 repo：{mrepo}{'' if os.path.isdir(mrepo) else '（找不到，略過）'}")
    log("═" * 64)

    # 1. 讀資料
    oh, ins, rev, val, corp = load_all(root, mrepo, args.extra)
    if args.corp_actions:
        corp = args.corp_actions
    if not oh:
        log("✗ 找不到任何價量資料（需要 date/stock_id/open/high/low/close/volume）。")
        sys.exit(1)
    px = pd.concat(oh).sort_values("_prio").drop_duplicates(["date", "stock_id"], keep="first")
    px = px[px["stock_id"].str.fullmatch(r"[1-9]\d{3}|0050")]
    dates = np.array(sorted(px["date"].unique()), dtype="datetime64[ns]")
    ids = np.array(sorted(px["stock_id"].unique()))
    log(f"\n── 價量：{len(px):,} 列，{len(ids)} 檔，{pd.Timestamp(dates[0]):%Y-%m-%d} ~ {pd.Timestamp(dates[-1]):%Y-%m-%d}")
    yc = pd.Series(pd.DatetimeIndex(dates).year).value_counts().sort_index()
    log("   各年交易日數：" + "、".join(f"{y}:{c}" for y, c in yc.items()))

    def wide(df, col):
        return df.pivot(index="date", columns="stock_id", values=col).reindex(index=dates, columns=ids).values.astype(float)

    rO, rH, rL, rC, V = (wide(px, c) for c in ("open", "high", "low", "close", "volume"))
    VAL = wide(px, "value") if "value" in px.columns else np.full(rC.shape, np.nan)
    med_vol = np.nanmedian(np.nanmedian(V, axis=0))
    if med_vol < 20000:
        log(f"   ⚠ 成交量中位數 {med_vol:,.0f} 看起來是「張」，換算成股（×1000）")
        V = V * 1000
    VAL = np.where(np.isnan(VAL), rC * V, VAL)
    rO[rO <= 0] = np.nan; rH[rH <= 0] = np.nan; rL[rL <= 0] = np.nan; rC[rC <= 0] = np.nan

    # 2. 除權息還原
    if corp is None:
        for cand in glob.glob(os.path.join(root, "**", "official_corporate_actions*.csv"), recursive=True) + \
                glob.glob(os.path.join(mrepo, "**", "official_corporate_actions*.csv"), recursive=True):
            corp = cand
            break
    R, cst = corp_ratio(corp, rC, dates, ids)
    if corp is None or cst["applied"] == 0:
        log("\n⚠⚠ 沒有套用任何官方除權息資料：除息跳空會被誤判成下跌，0050 也少算股息。結果只能當參考。")
    else:
        log(f"\n── 除權息：{os.path.basename(corp)}，套用 {cst['applied']} 筆（略過 {cst['skipped']}），"
            f"各年：{cst['by_year']}，0050 {'有' if cst['has_0050'] else '沒有'}配息紀錄")
        last_y = str(pd.Timestamp(dates[-1]).year)
        if last_y not in cst["by_year"]:
            log(f"   ⚠ {last_y} 年沒有任何除權息紀錄：{last_y} 的除息會被當成下跌（對策略偏保守），"
                f"0050 的 {last_y} 股息會漏算（對 0050 不利，會讓 AI 看起來比較好）。")
    idx = build_index(rC, R)
    f = idx / rC
    aO, aH, aL, aC = rO * f, rH * f, rL * f, idx
    Cff = pd.DataFrame(aC).ffill().values

    # 3. 候選池
    D = lambda x: pd.DataFrame(x)
    val20 = D(VAL).rolling(20, min_periods=15).mean().values
    prevC = D(aC).shift(1).values
    tr = np.fmax(aH, prevC) - np.fmin(aL, prevC)
    tr = np.where(np.isnan(prevC), aH - aL, tr)
    atr = D(tr).rolling(20, min_periods=15).mean().values
    is_stock = np.array([bool(re.fullmatch(r"[1-9]\d{3}", s)) for s in ids])
    v_rank = D(np.where(is_stock, val20, np.nan)).rank(axis=1, ascending=False).values
    hist_ok = D(~np.isnan(aC)).rolling(60, min_periods=1).sum().values >= 55
    U = is_stock[None, :] & (val20 >= UNIV_MIN_VALUE) & (v_rank <= UNIV_TOP) & ~np.isnan(aC) & ~np.isnan(atr) & hist_ok
    ui, uj = np.nonzero(U)
    log(f"── 候選池：每日平均 {U.sum(1)[U.sum(1) > 0].mean():.0f} 檔，共 {len(ui):,} 筆（訊號日×股票）")

    # 4. 特徵（全部只用 T 日收盤前已知資料）
    F = {}

    def put(name, W):
        F[name] = np.asarray(W, dtype=np.float32)[ui, uj]

    def rk(name, W):
        put(name + "_rk", D(np.where(U, W, np.nan)).rank(axis=1, pct=True).values)

    A = D(aC)
    lr1 = np.log(A / A.shift(1))
    ret = {n: (A / A.shift(n) - 1).values for n in (1, 3, 5, 10, 20, 60, 120)}
    for n, W in ret.items():
        put(f"ret{n}", W)
    for n in (5, 20, 60):
        put(f"vol{n}", lr1.rolling(n, min_periods=int(n * .7)).std().values)
    put("atr_pct", atr / aC)
    put("atr_ratio", D(tr).rolling(5, min_periods=4).mean().values / atr)
    ma = {n: A.rolling(n, min_periods=int(n * .8)).mean() for n in (5, 10, 20, 60, 120)}
    for n, M in ma.items():
        put(f"ma_gap{n}", (A / M - 1).values)
    put("ma20_slope", (ma[20] / ma[20].shift(5) - 1).values)
    put("ma60_slope", (ma[60] / ma[60].shift(10) - 1).values)
    for n in (20, 60, 120, 250):
        hi = D(aH).rolling(n, min_periods=int(n * .7)).max()
        lo = D(aL).rolling(n, min_periods=int(n * .7)).min()
        put(f"pos{n}", ((A - lo) / (hi - lo)).values)
        put(f"dist_hi{n}", (A / hi - 1).values)
    Vd = D(V)
    vm20 = Vd.rolling(20, min_periods=15).mean()
    put("vr1", (Vd / vm20).values)
    put("vr5", (Vd.rolling(5, min_periods=4).mean() / vm20).values)
    put("vr60", (vm20 / Vd.rolling(60, min_periods=40).mean()).values)
    put("log_val20", np.log(val20))
    put("val_chg", (D(VAL).rolling(5, min_periods=4).mean().values / val20))
    body_den = np.where(atr > 0, atr, np.nan)
    put("body", (aC - aO) / body_den)
    put("upper", (aH - np.fmax(aO, aC)) / body_den)
    put("lower", (np.fmin(aO, aC) - aL) / body_den)
    put("gap", aO / prevC - 1)
    put("clv", (aC - aL) / (aH - aL))
    r1 = D(ret[1])
    put("lim_up20", (r1 >= 0.095).rolling(20, min_periods=1).sum().values)
    put("up_days10", (r1 > 0).rolling(10, min_periods=7).mean().values)
    put("max_ret20", r1.rolling(20, min_periods=15).max().values)
    put("min_ret20", r1.rolling(20, min_periods=15).min().values)
    put("skew20", lr1.rolling(20, min_periods=15).skew().values)
    put("price_log", np.log(rC))

    # 大盤狀態
    Um = np.where(U, 1.0, np.nan)
    mkt = {}
    for n in (1, 5, 20, 60):
        mkt[f"mkt_ret{n}"] = np.nanmean(ret[n] * Um, axis=1)
    mkt["mkt_breadth20"] = np.nanmean(np.where(U, (A > ma[20]).values, np.nan), axis=1)
    mkt["mkt_breadth60"] = np.nanmean(np.where(U, (A > ma[60]).values, np.nan), axis=1)
    mkt["mkt_vol20"] = pd.Series(mkt["mkt_ret1"]).rolling(20, min_periods=15).std().values
    mkt["mkt_disp20"] = np.nanstd(ret[20] * Um, axis=1)
    j50 = np.nonzero(ids == "0050")[0]
    if len(j50):
        s = pd.Series(aC[:, j50[0]])
        mkt["b50_ret20"] = (s / s.shift(20) - 1).values
        mkt["b50_ret60"] = (s / s.shift(60) - 1).values
        mkt["b50_gap60"] = (s / s.rolling(60, min_periods=45).mean() - 1).values
        mkt["b50_gap200"] = (s / s.rolling(200, min_periods=150).mean() - 1).values
    for k, v in mkt.items():
        F[k] = np.asarray(v, np.float32)[ui]
    for n in (5, 20, 60):
        F[f"rs{n}"] = F[f"ret{n}"] - F[f"mkt_ret{n}"]

    # 法人
    inst_used = []
    if ins:
        it = pd.concat(ins).sort_values("_prio").drop_duplicates(["date", "stock_id"], keep="first")
        for cat in ("foreign", "trust", "dealer"):
            if cat not in it.columns or it[cat].notna().sum() == 0:
                continue
            M = D(it.pivot(index="date", columns="stock_id", values=cat).reindex(index=dates, columns=ids).values.astype(float))
            if np.nanmedian(np.abs(M.values)) < 50 and med_vol >= 20000:
                M = M * 1000  # 法人檔若以張為單位
            inst_used.append(cat)
            put(f"{cat}_1", (M / Vd).values)
            for n in (5, 20):
                put(f"{cat}_{n}", (M.rolling(n, min_periods=int(n * .6)).sum() / Vd.rolling(n, min_periods=int(n * .6)).sum()).values)
            put(f"{cat}_pos10", (M > 0).where(M.notna()).rolling(10, min_periods=6).mean().values)
            rk(f"{cat}_5", (M.rolling(5, min_periods=3).sum() / Vd.rolling(5, min_periods=3).sum()).values)
            rk(f"{cat}_20", (M.rolling(20, min_periods=12).sum() / Vd.rolling(20, min_periods=12).sum()).values)
    log(f"── 法人欄位：{inst_used if inst_used else '無'}")

    for nm, W in (("ret5", ret[5]), ("ret20", ret[20]), ("ret60", ret[60]), ("ret120", ret[120]),
                  ("atr_pct", atr / aC), ("log_val20", np.log(val20)), ("vr5", (Vd.rolling(5, min_periods=4).mean() / vm20).values),
                  ("ma_gap20", (A / ma[20] - 1).values), ("dist_hi250", None)):
        if W is not None:
            rk(nm, W)

    # 營收、估值（依公告可得日 merge，歷史不足會在後面自動剔除）
    long_key = pd.DataFrame({"date": pd.DatetimeIndex(dates[ui]), "stock_id": ids[uj], "_o": np.arange(len(ui))})
    rt = rev_table(rev)
    if rt is not None and len(rt):
        mg = pd.merge_asof(long_key.sort_values("date"), rt.sort_values("avail"), left_on="date",
                           right_on="avail", by="stock_id", direction="backward",
                           tolerance=pd.Timedelta(days=75)).sort_values("_o")
        for c in ("rev_yoy", "rev_mom", "rev_yoy3", "rev_vs12"):
            F[c] = mg[c].values.astype(np.float32)
    if val:
        vt = pd.concat(val).drop_duplicates(["date", "stock_id"], keep="last").sort_values("date")
        mg = pd.merge_asof(long_key.sort_values("date"), vt, on="date", by="stock_id",
                           direction="backward", tolerance=pd.Timedelta(days=7)).sort_values("_o")
        for c in ("pe", "pb", "dy"):
            F[c] = mg[c].values.astype(np.float32)

    # 5. 交易結果
    log("── 計算每筆訊號的真實交易結果（T+1 開盤進、30 日內 +10% vs 收盤停損）")
    valid_w, status_w, xoff_w, xpx_w, entry_w, net_w = outcomes(aO, aH, aC, rO, rH, rC, V, atr, args.slip)
    valid, status, xoff = valid_w[ui, uj], status_w[ui, uj], xoff_w[ui, uj]
    xpx, entry, net = xpx_w[ui, uj], entry_w[ui, uj], net_w[ui, uj]
    d, j = ui, uj
    done = valid & (status > 0)
    y = (status == 1).astype(np.int8)

    t_train_end = int(np.searchsorted(dates, np.datetime64(TRAIN_END)))
    t_test = int(np.searchsorted(dates, np.datetime64(TEST_START)))
    t_last = len(dates) - 1
    if t_test >= t_last:
        log("✗ 資料沒有 2025 年以後的交易日，無法做測試。")
        sys.exit(1)

    names = [k for k in F if k not in ()]
    X = np.column_stack([F[k] for k in names]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    pre = d < t_test
    miss = np.isnan(X[pre]).mean(0)
    keep = (miss <= 0.5) & (np.nanstd(X[pre], 0) > 0)
    dropped = [n for n, k in zip(names, keep) if not k]
    names = [n for n, k in zip(names, keep) if k]
    X = X[:, keep]
    log(f"── 特徵 {len(names)} 個" + (f"；因訓練期缺資料剔除：{dropped}" if dropped else ""))

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    m_trA = done & (d <= t_train_end - PURGE)
    m_tune = (d >= t_train_end) & (d < t_test)
    m_trB = done & (d <= t_test - PURGE)
    m_test = d >= t_test
    log(f"── 樣本：訓練 {m_trA.sum():,}（2020~2023）｜調整 {(m_tune & done).sum():,}（2024）｜"
        f"最終訓練 {m_trB.sum():,}｜測試訊號 {m_test.sum():,}")
    log(f"   基準率（訓練期全體勝率）：{y[m_trA].mean():.1%}")
    if args.check:
        log("\n✓ 資料檢查完成（--check 模式：沒有訓練、沒有碰測試期）。確認上面的年份、檔數、除權息、法人都正常後，拿掉 --check 正式跑一次。")
        return

    grid = [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=lf, min_samples_leaf=ms)
            for lr, it in ((0.05, 150), (0.03, 400)) for lf in (15, 63) for ms in (300, 1500)]
    if args.quick:
        grid = grid[:2]
    log(f"\n── 2024 調整期：試 {len(grid)} 組參數（評分＝每日前 5 檔平均淨報酬）")
    best, best_sc, tune_rows = None, -9, []
    for g in grid:
        t1 = time.time()
        mdl = HistGradientBoostingClassifier(l2_regularization=1.0, early_stopping=False, random_state=0, **g)
        mdl.fit(X[m_trA], y[m_trA])
        p = np.full(len(d), np.nan)
        p[m_tune] = mdl.predict_proba(X[m_tune])[:, 1]
        sel = daily_top(d, np.nan_to_num(p, nan=-1), 5, m_tune)
        st = trade_stats(sel, valid, status, net, xoff)
        auc = roc_auc_score(y[m_tune & done], p[m_tune & done]) if (m_tune & done).sum() > 100 else np.nan
        sc = st["ev"] if st["ev"] == st["ev"] else -9
        tune_rows.append({**g, "auc": auc, **{f"top5_{k}": v for k, v in st.items()}})
        log(f"   {g} → AUC {auc:.3f}，前5檔 勝率 {st['win']:.1%} 期望值 {st['ev']:+.2%}（{time.time() - t1:.0f}s）")
        if sc > best_sc:
            best, best_sc, best_model = g, sc, mdl
    log(f"   ⇒ 選用 {best}")

    # 特徵重要度（調整期，置換法）
    imp = None
    try:
        from sklearn.inspection import permutation_importance
        mt = np.nonzero(m_tune & done)[0]
        rs = np.random.RandomState(0)
        mt = rs.choice(mt, min(len(mt), 40000), replace=False)
        pi = permutation_importance(best_model, X[mt], y[mt], scoring="roc_auc", n_repeats=2, random_state=0)
        imp = sorted(zip(names, pi.importances_mean), key=lambda x: -x[1])
    except Exception as e:
        log(f"   特徵重要度略過：{e}")

    # 6. 用 2020~2024 重新訓練，測試期只看一次
    log("\n── 以 2020~2024 重新訓練（清除跨界樣本），進入測試期（2025~）")
    final = HistGradientBoostingClassifier(l2_regularization=1.0, early_stopping=False, random_state=0, **best)
    final.fit(X[m_trB], y[m_trB])
    prob = np.full(len(d), -1.0)
    prob[m_test] = final.predict_proba(X[m_test])[:, 1]

    # ───────────────── Phase 2：2025 OOS 高分群價格路徑診斷（只分析，不回頭調模型） ─────────────────
    phase2_dir = os.path.join(root, "research", "route2_phase2_out")
    os.makedirs(phase2_dir, exist_ok=True)

    id_pos_phase2 = {s:i for i,s in enumerate(ids)}
    q_all = np.nonzero(m_test & done & valid & np.isfinite(entry) & (entry > 0))[0]
    p_all = prob[q_all]
    q90 = float(np.nanquantile(p_all, 0.90))
    q95 = float(np.nanquantile(p_all, 0.95))
    q98 = float(np.nanquantile(p_all, 0.98))

    horizons = [1, 3, 5, 10, 20, 30, 40, 60]
    targets = [0.05, 0.10, 0.15, 0.20]
    path_rows = []
    enough60 = 0
    for q in q_all:
        t = int(d[q]); jj = int(j[q]); ep = float(entry[q])
        max_h = min(60, t_last - (t + 1))
        if max_h <= 0:
            continue
        highs = np.array([aH[t + 1 + k, jj] for k in range(1, max_h + 1)], dtype=float)
        lows  = np.array([aL[t + 1 + k, jj] for k in range(1, max_h + 1)], dtype=float)
        closes= np.array([aC[t + 1 + k, jj] for k in range(1, max_h + 1)], dtype=float)
        hr = highs / ep - 1.0
        lr = lows / ep - 1.0
        cr = closes / ep - 1.0
        row = {
            "signal_date": pd.Timestamp(dates[t]).strftime("%Y-%m-%d"),
            "stock_id": ids[jj], "prob": float(prob[q]), "entry": ep,
            "orig_win30": int(status[q] == 1), "orig_net": float(net[q]),
            "max_h": int(max_h)
        }
        if max_h >= 60:
            enough60 += 1
        for h in horizons:
            if max_h >= h:
                hh=hr[:h]; ll=lr[:h]; cc=cr[:h]
                row[f"mfe_{h}"] = float(np.nanmax(hh)) if np.isfinite(hh).any() else np.nan
                row[f"mae_{h}"] = float(np.nanmin(ll)) if np.isfinite(ll).any() else np.nan
                row[f"close_{h}"] = float(cc[h-1]) if np.isfinite(cc[h-1]) else np.nan
            else:
                row[f"mfe_{h}"] = row[f"mae_{h}"] = row[f"close_{h}"] = np.nan
        finite_hr = np.where(np.isfinite(hr), hr, -np.inf)
        row["peak_day_60"] = int(np.argmax(finite_hr) + 1) if np.isfinite(hr).any() else np.nan
        row["peak_ret_60"] = float(np.nanmax(hr)) if np.isfinite(hr).any() else np.nan
        trough = np.where(np.isfinite(lr), lr, np.inf)
        row["trough_day_60"] = int(np.argmin(trough) + 1) if np.isfinite(lr).any() else np.nan
        row["trough_ret_60"] = float(np.nanmin(lr)) if np.isfinite(lr).any() else np.nan
        for tar in targets:
            hit = np.nonzero(hr >= tar)[0]
            tag = str(int(tar*100))
            row[f"hit{tag}_60"] = int(len(hit) > 0)
            row[f"day_hit{tag}"] = int(hit[0] + 1) if len(hit) else np.nan
            hit30 = np.nonzero(hr[:min(30,max_h)] >= tar)[0]
            row[f"hit{tag}_30"] = int(len(hit30) > 0)
        path_rows.append(row)

    pth = pd.DataFrame(path_rows)
    pth["cohort_D10"] = pth["prob"] >= q90
    pth["cohort_top5pct"] = pth["prob"] >= q95
    pth["cohort_top2pct"] = pth["prob"] >= q98
    for th in (0.70,0.75,0.80,0.85):
        pth[f"cohort_p{int(th*100)}"] = pth["prob"] >= th

    cohorts = {
        "D10": pth["cohort_D10"],
        "Top5pct": pth["cohort_top5pct"],
        "Top2pct": pth["cohort_top2pct"],
        "P>=70": pth["cohort_p70"],
        "P>=75": pth["cohort_p75"],
        "P>=80": pth["cohort_p80"],
        "P>=85": pth["cohort_p85"],
    }
    summary_rows=[]
    for name, mask in cohorts.items():
        z=pth[mask].copy()
        if len(z)==0: continue
        rr={"cohort":name,"n":int(len(z)),"mean_prob":float(z["prob"].mean()),
            "orig_win30":float(z["orig_win30"].mean()),"orig_ev":float(z["orig_net"].mean()),
            "median_peak_day60":float(z["peak_day_60"].median()),"mean_peak_ret60":float(z["peak_ret_60"].mean()),
            "mean_trough_ret60":float(z["trough_ret_60"].mean())}
        for tar in targets:
            tag=str(int(tar*100))
            rr[f"hit{tag}_30"]=float(z[f"hit{tag}_30"].mean())
            rr[f"hit{tag}_60"]=float(z[f"hit{tag}_60"].mean())
            hitdays=z.loc[z[f"hit{tag}_60"]==1,f"day_hit{tag}"]
            rr[f"median_day_hit{tag}"]=float(hitdays.median()) if len(hitdays) else np.nan
        for h in horizons:
            rr[f"mfe{h}_mean"]=float(z[f"mfe_{h}"].mean())
            rr[f"mae{h}_mean"]=float(z[f"mae_{h}"].mean())
            rr[f"close{h}_mean"]=float(z[f"close_{h}"].mean())
        summary_rows.append(rr)
    phase2_summary=pd.DataFrame(summary_rows)

    # D10：原 30 日 +10% 成功 vs 未成功的每日平均 close path，找分化時間
    d10=pth[pth["cohort_D10"]].copy()
    div_rows=[]
    for day in range(1,61):
        c=f"close_{day}" if day in horizons else None
        # 未預先存逐日 close 的日子，直接回原矩陣重算
        vals_win=[]; vals_fail=[]
        for _, rr in d10.iterrows():
            t=int(np.searchsorted(dates, np.datetime64(rr["signal_date"])))
            jj=int(np.searchsorted(ids, rr["stock_id"])) if False else None
        # 以 signal_date/stock_id 建索引，避免依排序假設
        for idx in d10.index:
            rr=d10.loc[idx]
            t=int(np.searchsorted(dates, np.datetime64(rr["signal_date"])))
            jj=id_pos_phase2.get(str(rr["stock_id"]), -1)
            if jj < 0 or t+1+day > t_last: continue
            px=aC[t+1+day,jj]
            if not np.isfinite(px) or rr["entry"]<=0: continue
            val=float(px/rr["entry"]-1)
            (vals_win if rr["orig_win30"]==1 else vals_fail).append(val)
        div_rows.append({"day":day,
                         "win_mean_close":float(np.mean(vals_win)) if vals_win else np.nan,
                         "fail_mean_close":float(np.mean(vals_fail)) if vals_fail else np.nan,
                         "spread":float(np.mean(vals_win)-np.mean(vals_fail)) if vals_win and vals_fail else np.nan,
                         "n_win":len(vals_win),"n_fail":len(vals_fail)})

    div=pd.DataFrame(div_rows)
    pth.to_csv(os.path.join(phase2_dir,"phase2_signal_paths.csv"),index=False,encoding="utf-8-sig")
    phase2_summary.to_csv(os.path.join(phase2_dir,"phase2_cohort_summary.csv"),index=False,encoding="utf-8-sig")
    div.to_csv(os.path.join(phase2_dir,"phase2_d10_divergence.csv"),index=False,encoding="utf-8-sig")
    phase2_meta={"q90":q90,"q95":q95,"q98":q98,"n_all":int(len(pth)),"n_enough60":int(enough60),
                 "model_params":best,"model_frozen":True,"note":"2025 is diagnostic research only; not a fresh OOS after inspection."}
    json.dump(phase2_meta,open(os.path.join(phase2_dir,"phase2_meta.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    log("\n── Phase 2 D10 路徑診斷完成 ──")
    log(phase2_summary.to_string(index=False))

    td = pd.DatetimeIndex(dates[t_test:t_last + 1])
    results, curves = {}, {"date": td.strftime("%Y-%m-%d")}
    auc_test = roc_auc_score(y[m_test & done], prob[m_test & done]) if (m_test & done).sum() > 100 else np.nan
    base = trade_stats(np.nonzero(m_test)[0], valid, status, net, xoff)

    # 基準：0050
    b50 = None
    if len(j50):
        s = Cff[t_test:t_last + 1, j50[0]]
        s = s / s[0] * (1 - args.slip) * (1 - FEE) if s[0] == s[0] else None
        if s is not None:
            b50 = curve_stats(s, td)
            curves["0050_含息"] = s
    r10 = None
    if args.r10_equity and os.path.exists(args.r10_equity):
        rq = norm_cols(read_any(args.r10_equity))
        ec = pick_col(rq.columns, ["equity", "nav", "淨值", "資產", "total", "value"])
        rq["date"] = parse_date(rq["date"])
        rq = rq[(rq["date"] >= td[0]) & (rq["date"] <= td[-1])]
        if ec and len(rq) > 5:
            e = to_num(rq[ec]).values
            r10 = curve_stats(e / e[0], pd.DatetimeIndex(rq["date"]))

    pick_rows = []
    # 每日基準（同一天全體可交易候選的平均淨報酬），用來算超額並做月份區塊 bootstrap
    bq = np.nonzero(m_test & done)[0]
    day_sum = np.bincount(d[bq], weights=net[bq], minlength=len(dates))
    day_cnt = np.bincount(d[bq], minlength=len(dates))
    day_base = np.where(day_cnt > 0, day_sum / np.maximum(day_cnt, 1), np.nan)
    month_of = pd.DatetimeIndex(dates).to_period("M").astype(str).values

    def excess_ci(tr_idx, B=2000):
        q = tr_idx[done[tr_idx]]
        if len(q) < 20:
            return np.nan, np.nan
        ex = net[q] - day_base[d[q]]
        dfm = pd.DataFrame({"m": month_of[d[q]], "ex": ex}).groupby("m")["ex"].agg(["sum", "count"])
        rsb = np.random.RandomState(7)
        k = len(dfm)
        draws = rsb.randint(0, k, size=(B, k))
        boot = dfm["sum"].values[draws].sum(1) / dfm["count"].values[draws].sum(1)
        return float(ex.mean()), float(np.percentile(boot, 5))

    for N in args.top:
        sel = daily_top(d, prob, N, m_test)
        eq, ntr, expo = simulate(sel, N, d, j, valid, status, xoff, xpx, entry, Cff, t_test, t_last, args.slots, args.slip)
        st = trade_stats(ntr, valid, status, net, xoff)   # 以實際成交（同檔持有中不重複買）計算
        st["signals"] = int(len(sel))
        exm, exlo = excess_ci(ntr)
        cs = curve_stats(eq, td)
        curves[f"AI_前{N}檔"] = eq
        by_year = {}
        for yv in sorted(set(td.year)):
            ss = ntr[pd.DatetimeIndex(dates[d[ntr]]).year == yv]
            by_year[str(yv)] = trade_stats(ss, valid, status, net, xoff)
        rnd_tot, rnd_ev = [], []
        rs = np.random.RandomState(42)
        for k in range(args.seeds):
            sc = rs.rand(len(d))
            rsel = daily_top(d, sc, N, m_test)
            req, rtr, _ = simulate(rsel, N, d, j, valid, status, xoff, xpx, entry, Cff, t_test, t_last, args.slots, args.slip)
            rnd_tot.append(req[-1] / req[0] - 1)
            rnd_ev.append(trade_stats(rtr, valid, status, net, xoff)["ev"])
        pct = float(np.mean(np.array(rnd_ev) < st["ev"]))  # 以每筆期望值比較（不受持股水位影響）
        results[N] = {"trades": st, "curve": cs, "by_year": by_year, "exposure": expo, "excess_mean": exm, "excess_lo90": exlo,
                      "random_total_median": float(np.median(rnd_tot)), "random_ev_median": float(np.nanmedian(rnd_ev)),
                      "beat_random_pct": pct}
        if N == max(args.top):
            for q in sel:
                pick_rows.append({"signal_date": pd.Timestamp(dates[d[q]]).strftime("%Y-%m-%d"), "stock_id": ids[j[q]],
                                  "prob": round(float(prob[q]), 4), "tradable": bool(valid[q]),
                                  "result": {0: "未結/取消", 1: "達標", 2: "停損", 3: "到期"}[int(status[q])],
                                  "net_ret": round(float(net[q]), 4) if done[q] else None,
                                  "hold_days": int(xoff[q] - 1) if done[q] else None})

    # 壓力成本（資料 repo CONTRACT.md）：手續費 0.1425%、買進滑價 0.5%、賣出以開盤×0.98 計（所有出場一律套 2%）
    # 成本不影響模型與選股，只影響損益，所以同一次執行一起算，不會多看一次測試期
    SC = (0.005, 0.02, 0.001425)
    net_s = (xpx / entry) * (1 - SC[1]) * (1 - SC[2] - TAX) / ((1 + SC[0]) * (1 + SC[2])) - 1
    stress = {}
    for N in args.top:
        sel = daily_top(d, prob, N, m_test)
        eq_s, ntr_s, expo_s = simulate(sel, N, d, j, valid, status, xoff, xpx, entry, Cff, t_test, t_last, args.slots, 0, SC)
        stress[N] = {"trades": trade_stats(ntr_s, valid, status, net_s, xoff), "curve": curve_stats(eq_s, td), "exposure": expo_s}
        curves[f"AI_前{N}檔_壓力成本"] = eq_s

    # 機率分組（模型有沒有分辨力）
    dq = np.nonzero(m_test & done)[0]
    dec = pd.DataFrame({"p": prob[dq], "win": y[dq], "ret": net[dq]})
    dec["組"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    dect = dec.groupby("組", observed=True).agg(筆數=("win", "size"), 平均機率=("p", "mean"),
                                                 實際勝率=("win", "mean"), 期望值=("ret", "mean"))

    # 7. 報告
    P = lambda x: "—" if x is None or x != x else f"{x:+.2%}"
    Q = lambda x: "—" if x is None or x != x else f"{x:.1%}"
    L = []
    L.append(f"# AlphaPilot 路線二驗證報告（{datetime.now():%Y-%m-%d %H:%M}）\n")
    L.append(f"- 資料：{pd.Timestamp(dates[0]):%Y-%m-%d} ~ {pd.Timestamp(dates[-1]):%Y-%m-%d}；測試期 {td[0]:%Y-%m-%d} ~ {td[-1]:%Y-%m-%d}（{len(td)} 交易日）")
    L.append(f"- 目標：30 交易日內先碰 +10% 而非收盤跌破 1.5×ATR20；成本 手續費 {FEE:.4%}×2、稅 {TAX:.1%}、滑價每邊 {args.slip:.2%}")
    L.append(f"- 除權息：{'已套用 ' + str(cst['applied']) + ' 筆' if cst['applied'] else '未套用（結果僅供參考）'}；法人：{inst_used or '無'}；特徵 {len(names)} 個")
    L.append(f"- 選用參數：{best}；測試期 AUC：{auc_test:.3f}（0.5＝沒有分辨力）")
    L.append(f"- 測試期全體候選（每天約 {U[t_test:].sum(1).mean():.0f} 檔）基準：勝率 {Q(base['win'])}，每筆期望值 {P(base['ev'])}\n")
    L.append("## 測試期結果（2025~，只看這一次）\n")
    L.append("| 每天挑 | 實際成交 | 勝率 | 每筆期望值 | 中位數 | 平均持有 | 投組累積 | 年化 | 最大回撤 | 持股水位 | 隨機挑每筆期望值 | 超額(信賴下限) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N, r in results.items():
        t, c = r["trades"], r["curve"]
        L.append(f"| 前{N}檔 | {t['n']} | {Q(t['win'])} | {P(t['ev'])} | {P(t['med'])} | {t['hold']:.1f}天 | {P(c['total'])} | "
                 f"{P(c['cagr'])} | {Q(c['mdd'])} | {r['exposure']:.0%} | {P(r['random_ev_median'])} | {P(r['excess_mean'])}（{P(r['excess_lo90'])}） |")
    if b50:
        L.append(f"| 0050含息 | — | — | — | — | — | {P(b50['total'])} | {P(b50['cagr'])} | {Q(b50['mdd'])} | 100% | — | — |")
    if r10:
        L.append(f"| R10-MAX | — | — | — | — | — | {P(r10['total'])} | {P(r10['cagr'])} | {Q(r10['mdd'])} | — | — | — |")
    L.append("\n實際成交＝投組模擬中真的買進的筆數（同一檔持有中不重複買）。超額＝每筆淨報酬減去同一天全體候選的平均；"
             "括號是以「月」為單位重抽 2000 次的單邊 95% 信賴下限，下限 >0 才算挑股能力站得住。"
             "推薦常連續多天是同一檔，逐筆算顯著性會高估，所以用月份區塊。投組累積受持股水位影響，要一併看水位。")
    if not r10:
        L.append("R10-MAX：未提供每日淨值檔（--r10-equity），本次不比較。")
    L.append("\n## 分年（交易層級）\n")
    for N, r in results.items():
        L.append(f"- 前{N}檔：" + "；".join(f"{yv} 年 {t['n']} 筆、勝率 {Q(t['win'])}、期望值 {P(t['ev'])}" for yv, t in r["by_year"].items()))
    L.append("\n## 判定（自動）\n")
    for N, r in results.items():
        ok_ev = r["trades"]["ev"] > 0
        ok_50 = b50 is not None and r["curve"]["total"] > b50["total"]
        ok_rd = r["excess_lo90"] == r["excess_lo90"] and r["excess_lo90"] > 0
        verdict = "有初步證據（仍需前向驗證）" if (ok_ev and ok_50 and ok_rd) else "未證明有效"
        L.append(f"- 前{N}檔：期望值>0 {'✓' if ok_ev else '✗'}｜贏 0050 {'✓' if ok_50 else '✗'}｜"
                 f"超額信賴下限>0 {'✓' if ok_rd else '✗'} → **{verdict}**")
    L.append("\n## 壓力成本（CONTRACT.md：手續費 0.1425%、買 +0.5%、賣 −2%）\n")
    L.append("| 每天挑 | 實際成交 | 勝率 | 每筆期望值 | 投組累積 | 年化 | 最大回撤 | 持股水位 | 贏 0050 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for N, r in stress.items():
        t, c = r["trades"], r["curve"]
        w = "—" if not b50 else ("✓" if c["total"] > b50["total"] else "✗")
        L.append(f"| 前{N}檔 | {t['n']} | {Q(t['win'])} | {P(t['ev'])} | {P(c['total'])} | {P(c['cagr'])} | {Q(c['mdd'])} | {r['exposure']:.0%} | {w} |")
    L.append("\n主判定用 R10-MAX 成本；壓力成本下若期望值仍 >0 且贏 0050，才算經得起較差的成交。"
             "（壓力版對 +10% 限價出場也扣 2%，比實際更保守。）")
    L.append("\n## 機率分組（測試期，D10＝模型最看好的一成）\n")
    L.append("| 組 | 筆數 | 平均預測機率 | 實際勝率 | 每筆期望值 |")
    L.append("|---|---|---|---|---|")
    for g, rr in dect.iterrows():
        L.append(f"| {g} | {int(rr['筆數'])} | {rr['平均機率']:.1%} | {rr['實際勝率']:.1%} | {P(rr['期望值'])} |")
    L.append("\n## 2024 調整期各組參數\n")
    for tr_ in tune_rows:
        L.append(f"- lr={tr_['learning_rate']} iter={tr_['max_iter']} leaf={tr_['max_leaf_nodes']} "
                 f"minleaf={tr_['min_samples_leaf']}：AUC {tr_['auc']:.3f}，前5檔勝率 {Q(tr_['top5_win'])}，期望值 {P(tr_['top5_ev'])}")
    if imp:
        L.append("\n## 模型最依賴的特徵（2024 調整期，置換重要度，前 20）\n")
        L.append("、".join(f"{n}（{v:+.4f}）" for n, v in imp[:20]))
    L.append("\n## 已知限制\n")
    L.append("- 候選池以 20 日均成交額排名代替市值排名（歷史股本資料不足）。")
    L.append("- 出場只用「+10% 目標／收盤停損／30 天到期」，未含 plan-v2 的保本、移動停利、10 天未 +5% 出場；跌停鎖死賣不掉未模擬。")
    L.append("- 營收、估值若沒有 2020 起的歷史，會在訓練期缺值過多而被自動剔除（見特徵清單）。")
    L.append("- 資料 repo 的上櫃外資欄位若仍有錯，只影響測試期輸入（不會造成看未來，只會讓結果變差）。")
    report = "\n".join(L)

    # 封存紀錄：記錄測試期被看過幾次
    sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    summary = {N: {"win": r["trades"]["win"], "ev": r["trades"]["ev"], "total": r["curve"]["total"]} for N, r in results.items()}
    seal_path = os.path.join(out_root, "seal_log.json")
    seal = json.load(open(seal_path, encoding="utf-8")) if os.path.exists(seal_path) else []
    seal.append({"time": datetime.now().isoformat(timespec="seconds"), "script_sha": sha, "args": vars(args),
                 "data_range": [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())],
                 "result_sha": hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()[:16]})
    json.dump(seal, open(seal_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report += (f"\n\n---\n測試期已被查看 {len(seal)} 次（含本次）。若在看過結果後修改設定再跑，"
               f"測試期就不再是乾淨的樣本外；請把第一次的結果當正式結論。程式指紋 {sha}。\n")

    open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8").write(report)
    pd.DataFrame(curves).to_csv(os.path.join(run_dir, "equity_curves.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pick_rows).to_csv(os.path.join(run_dir, "test_picks.csv"), index=False, encoding="utf-8-sig")
    json.dump({"best_params": best, "features": names, "dropped": dropped, "auc_test": auc_test, "base": base,
               "results": results, "stress_contract": stress, "0050": b50, "r10": r10, "corp": cst, "tune": tune_rows},
              open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log("\n" + report)
    log(f"\n輸出資料夾：{run_dir}（report.md、equity_curves.csv、test_picks.csv、metrics.json）")
    log(f"總耗時 {(time.time() - T0) / 60:.1f} 分鐘")

if __name__ == "__main__":
    main()