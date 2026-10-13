"""AlphaPilot 結構目標價驗證（只讀，不改任何東西）。2026-10-13

問題：程式算的「主目標」（ap/momentum/targets.py，發動路線與 v1 交易計畫共用）是不是系統性偏低？
       發動路線「報酬風險比 < 1 不准買」這條規則，擋掉的是輸家還是贏家？

資料：alphapilot-market-data-v2 的 data/history/2020-2025/ohlcv_20XX.parquet（欄位 date, code, name, volume, open, high, low, close）
用法：python verify_targets.py --hist <放 ohlcv_*.parquet 的資料夾> [--step 5] [--top 500] [--out out_dir]
需要：pandas、numpy、pyarrow

方法（不看未來）：
- 每 step 個交易日抽一天 T（2021 起，T 之前要有 250 天、之後要有 20 天）
- 當天選股範圍：近 20 日平均成交值前 top 名、股價 ≥ 10、4 碼普通股（排除 0 開頭 ETF）
- 用 T 收盤以前的資料算 ATR20（近 20 日真實區間平均，和 ap/features.py 相同）與結構目標（和 targets.py 相同算法，逐行照抄）
- 結果：T+1 開盤進場，T+1～T+20 的最高／最低／收盤
"""
import argparse
import glob
import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

_TICKS = [(10, "0.01"), (50, "0.05"), (100, "0.1"), (500, "0.5"), (1000, "1")]


def _tick(p):
    for b, t in _TICKS:
        if p < b:
            return Decimal(t)
    return Decimal("5")


def round_price(x):
    if x is None or x != x:
        return None
    p = Decimal(str(float(x)))
    t = _tick(float(p))
    q = (p / t).to_integral_value(rounding=ROUND_HALF_UP) * t
    t2 = _tick(float(q))
    if t2 != t:
        q = (p / t2).to_integral_value(rounding=ROUND_HALF_UP) * t2
    return float(q)


def limit_up(c):
    p = Decimal(str(c)) * Decimal("1.10")
    t = _tick(float(p))
    return float((p / t).to_integral_value(rounding="ROUND_FLOOR") * t)


def swing_highs(h, k=5):
    return [float(h[i]) for i in range(k, len(h) - k) if h[i] == h[i - k:i + k + 1].max()]


def compute(high, low, close, atr):
    n = len(high)
    if n < 30 or not atr or atr != atr:
        return None
    base_h = high[-45:-5] if n >= 50 else high[:-5]
    base_l = low[-45:-5] if n >= 50 else low[:-5]
    base_hi, base_lo = float(base_h.max()), float(base_l.min())
    broke = close > base_hi
    h250 = high[-250:]
    hi52 = float(h250.max())
    cands = swing_highs(h250) + [float(high[-5:].max()), hi52]
    res_lv = sorted({x for x in cands if x > close * 1.02})
    if res_lv:
        t1, t1b = res_lv[0], "前高壓力"
    else:
        t1, t1b = close + 2 * atr, "收盤+2ATR"
    mm = base_hi + (base_hi - base_lo) if broke else None
    if mm and mm > t1:
        main, mb = mm, "MM突破量度"
    elif hi52 > t1 * 1.02:
        main, mb = hi52, "HI52五十二週高"
    else:
        main, mb = close + 4 * atr, "ATR4收盤+4ATR"
    ext = base_hi + 1.618 * (base_hi - base_lo) if broke else None
    stretch = max(v for v in (ext, main + 2 * atr) if v)
    struct_inv = max(float(low[-5:].min()), close - 2 * atr)
    up = main / close - 1
    down = 1 - struct_inv / close if struct_inv < close else None
    return {"t1": round_price(t1), "t1_basis": t1b, "main": round_price(main), "main_basis": mb,
            "stretch": round_price(stretch), "inv": round_price(struct_inv), "broke_out": bool(broke),
            "upside": up, "downside": down, "rr": (up / down) if down and down > 0 else None,
            "chase": round_price(min(close + 0.5 * atr, limit_up(close)))}


def load(hist_dir):
    fs = sorted(glob.glob(os.path.join(hist_dir, "ohlcv_*.parquet")))
    if not fs:
        raise SystemExit(f"{hist_dir} 底下找不到 ohlcv_*.parquet")
    df = pd.concat([pd.read_parquet(f, columns=["date", "code", "open", "high", "low", "close", "volume"]) for f in fs])
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"].astype(str)).dt.strftime("%Y%m%d")
    df = df[df["code"].str.fullmatch(r"[1-9]\d{3}")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    return df.drop_duplicates(["date", "code"]).sort_values(["code", "date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--out", default="verify_targets_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.hist)
    P = {k: df.pivot(index="date", columns="code", values=k).sort_index() for k in ("open", "high", "low", "close", "volume")}
    dates = list(P["close"].index)
    O, H, L, C, V = (P[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    codes = list(P["close"].columns)
    prevc = np.vstack([np.full(C.shape[1], np.nan), C[:-1]])
    TR = np.fmax.reduce([H - L, np.abs(H - prevc), np.abs(L - prevc)])
    ATR = pd.DataFrame(TR).rolling(20, min_periods=20).mean().to_numpy()
    VAL20 = pd.DataFrame(C * V).rolling(20, min_periods=20).mean().to_numpy()
    rows = []
    idx = [i for i in range(250, len(dates) - 21, a.step) if dates[i] >= "20210101"]
    for k, i in enumerate(idx):
        val = VAL20[i].copy()
        ok = (C[i] >= 10) & np.isfinite(val) & np.isfinite(ATR[i])
        val[~ok] = -1
        pick = [j for j in np.argsort(-val)[:a.top] if val[j] > 0]
        for j in pick:
            h, l = H[:i + 1, j], L[:i + 1, j]
            m = np.isfinite(h) & np.isfinite(l)
            h, l = h[m][-260:], l[m][-260:]
            c, atr = C[i, j], ATR[i, j]
            t = compute(h, l, c, atr)
            if t is None:
                continue
            o1 = O[i + 1, j]
            fh, fl, fc = H[i + 1:i + 21, j], L[i + 1:i + 21, j], C[i + 1:i + 21, j]
            if not (np.isfinite(o1) and np.isfinite(fh).all() and np.isfinite(fl).all()):
                continue
            if np.any(np.abs(fc[1:] / fc[:-1] - 1) > 0.105) or abs(o1 / c - 1) > 0.105:
                continue
            first = lambda lvl, inv: next((("TARGET" if hh >= lvl and not ll <= inv else "INV") for hh, ll in zip(fh, fl) if hh >= lvl or ll <= inv), "NONE")
            rows.append({"date": dates[i], "code": codes[j], "close": c, "atr_pct": atr / c, "chg1": c / C[i - 1, j] - 1,
                         **t, "entry": o1, "mfe20": fh.max() / o1 - 1, "mae20": fl.min() / o1 - 1, "ret20": fc[-1] / o1 - 1,
                         "hit_t1": fh.max() >= t["t1"], "hit_main": fh.max() >= t["main"], "hit_stretch": fh.max() >= t["stretch"],
                         "hit15": fh.max() >= o1 * 1.15, "main_first": first(t["main"], t["inv"]) == "TARGET",
                         "beyond_main": fh.max() / t["main"] - 1})
        if k % 20 == 0:
            print(f"{dates[i]}  {k + 1}/{len(idx)}  樣本 {len(rows):,}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "samples.csv"), index=False)
    R["atr_q"] = pd.qcut(R["atr_pct"], 5, labels=["Q1低波動", "Q2", "Q3", "Q4", "Q5高波動"])
    R["rr_b"] = pd.cut(R["rr"].fillna(-1), [-2, 0, 1, 2, 1e9], labels=["無法計算", "<1", "1～2", "≥2"])
    R["upside_b"] = pd.cut(R["upside"], [-1, 0.05, 0.10, 0.15, 0.25, 10], labels=["<5%", "5～10%", "10～15%", "15～25%", "≥25%"])
    R["near_lu"] = np.where(R["chg1"] >= 0.09, "當天漲≥9%", "其他")

    def tab(by):
        g = R.groupby(by, observed=True)
        return pd.DataFrame({"樣本": g.size(), "主目標漲幅中位": g["upside"].median(), "20日內碰主目標": g["hit_main"].mean(),
                          "先碰主目標再碰失效": g["main_first"].mean(), "碰延伸目標": g["hit_stretch"].mean(),
                          "碰+15%": g["hit15"].mean(), "20日最高中位": g["mfe20"].median(), "超過主目標幅度中位": g["beyond_main"].median(),
                          "20日報酬平均": g["ret20"].mean(), "20日報酬中位": g["ret20"].median()})
    out = ["# 結構目標價驗證結果", "", f"樣本 {len(R):,} 筆｜{R['date'].min()}～{R['date'].max()}｜每 {a.step} 個交易日抽一天｜成交值前 {a.top} 名", ""]
    for title, by in (("全體", lambda x: "全體"), ("依主目標算法", "main_basis"), ("依是否已突破整理區", "broke_out"),
                      ("依報酬風險比（發動路線 <1 不准買）", "rr_b"), ("依主目標漲幅", "upside_b"), ("依波動", "atr_q"),
                      ("當天漲≥9%（接近漲停）", "near_lu"), (("已突破×主目標算法"), ["broke_out", "main_basis"])):
        t = tab(R.index.map(lambda _: "全體") if callable(by) else by)
        out += [f"## {title}", "", t.to_markdown(floatfmt=".3f"), ""]
        t.to_csv(os.path.join(a.out, f"table_{title}.csv"))
    sp = R[["upside", "mfe20"]].corr(method="spearman").iloc[0, 1]
    out += [f"主目標漲幅 vs 20 日實際最高漲幅 的 Spearman 相關：{sp:.3f}", ""]
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
