"""v0.9.1 全體基準率：假設「每天隨便買」，明天開盤買進後，出場前碰到買進價 +10% 的比例（5 日內、30 日內）。

做法（和正式結算完全同一套規則，直接呼叫 plan._one）：
  期間：預設 2020-01-01 ~ 2024-12-31（2025 年已被 Route2 看過，不用來定基準）。
  每隔 step 個交易日取一個發現日 T；範圍同每日判斷：代號 4 碼、有成交、20 日均成交額 ≥ min_amount20_twd、
  取成交額前 universe_top_n 名。
  停損：T 收盤 − 1.5 × ATR20（= plan-v2 主規則的停損下限；隨便買沒有 AI 停損，所以只用下限）。目標價不設限。
  出場規則：plan-v2 主規則（災難停損、收盤停損、10 天未 +5%、30 天未 +10%）。
  碰到 +10% 的那天就是發動日；A＝發動日 ≤ 5、B＝發動日 ≤ 30。T+1 放棄買進（開盤低於停損、一字漲停）的不計。
資料防呆：價格未還原除權息（和正式結算一致），但路徑中任一天漲跌超過 ±10.5%（台股漲跌幅限制外），
  幾乎都是除權、減資或面額變更造成的假跳動，該筆排除並計數。
"""
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .plan import _one

CODE_RE = re.compile(r"^\d{4}$")


def compute(store, cfg, start="20200101", end="20241231", step=3, log=print):
    """從本機資料（Store，含 data/history 的 parquet）計算。"""
    all_days = store.list_days("prices")
    sig_days = [d for d in all_days if start <= d <= end]
    if len(sig_days) < 60:
        raise SystemExit(f"{start}～{end} 只有 {len(sig_days)} 個交易日，歷史資料不足（需要 data/history 的 parquet）")
    i0 = max(0, all_days.index(sig_days[0]) - 25)
    i1 = min(len(all_days), all_days.index(sig_days[-1]) + 32)
    days = all_days[i0:i1]
    log(f"讀取 {days[0]}～{days[-1]} 共 {len(days)} 個交易日的價量…")
    return compute_frames(store.load_days("prices", days), cfg, start, end, step, log)


def compute_frames(px, cfg, start="20200101", end="20241231", step=3, log=print):
    """核心計算。px：價量長表，欄位 date(YYYYMMDD)、code、open、high、low、close、volume，可選 amount
    （沒有 amount 時用 close×volume 近似，和本機讀歷史 parquet 的做法相同）。GitHub Actions 版也呼叫這裡。"""
    px = px.copy()
    px["date"] = px["date"].astype(str)
    px["code"] = px["code"].astype(str).str.strip()
    px = px[px["code"].str.match(r"^\d{4}$")]
    days = sorted(px["date"].unique())
    sig_days = [d for d in days if start <= d <= end]
    if len(sig_days) < 60:
        raise SystemExit(f"{start}～{end} 只有 {len(sig_days)} 個交易日")
    P = {k: px.pivot_table(index="date", columns="code", values=k, aggfunc="last").reindex(days).astype(float)
         for k in ("open", "high", "low", "close", "volume")}
    amt = (px.pivot_table(index="date", columns="code", values="amount", aggfunc="last").reindex(days).astype(float)
           if "amount" in px.columns and px["amount"].notna().any() else P["close"] * P["volume"])
    prev = P["close"].shift(1)
    tr = np.maximum.reduce([(P["high"] - P["low"]).values, (P["high"] - prev).abs().values, (P["low"] - prev).abs().values])
    atr = pd.DataFrame(tr, index=days, columns=P["close"].columns).rolling(20, min_periods=20).mean()
    amt20 = amt.rolling(20, min_periods=20).mean()
    jump = (P["close"] / prev - 1).abs() > 0.105
    floor, top_n = cfg.get("min_amount20_twd") or 0, cfg.get("universe_top_n") or 500
    O, H, L, C = (P[k].values for k in ("open", "high", "low", "close"))
    J = jump.values
    idx = {d: i for i, d in enumerate(days)}
    n = n5 = n30 = cancelled = anomalies = 0
    hit_days, by_year = [], {}
    picks = sig_days[::step]
    for t_no, d in enumerate(picks):
        i = idx[d]
        if i + 31 > len(days):
            break
        a20 = amt20.iloc[i]
        ok = a20[(a20 >= floor) & (P["volume"].iloc[i] > 0) & atr.iloc[i].notna() & P["close"].iloc[i].notna()]
        codes = ok.sort_values(ascending=False).index[:top_n]
        cols = [P["close"].columns.get_loc(c) for c in codes]
        for j in cols:
            ref, a = C[i, j], float(atr.iat[i, j])
            if J[i + 1:i + 31, j].any():
                anomalies += 1
                continue
            bars = [(days[k], O[k, j], H[k, j], L[k, j], C[k, j]) for k in range(i + 1, i + 31)]
            bars = [(dd, *(None if v != v else float(v) for v in b)) for dd, *b in bars]
            r = _one(float(ref), float("inf"), float(ref) - 1.5 * a, a, bars, 10)
            if r["status"] == "CANCELLED":
                cancelled += 1
                continue
            if r["status"] == "PENDING":
                continue
            hd = r.get("hit10_day")
            n += 1
            y = by_year.setdefault(d[:4], [0, 0, 0])
            y[0] += 1
            if hd is not None:
                hit_days.append(hd)
                n30 += 1
                y[2] += 1
                if hd <= 5:
                    n5 += 1
                    y[1] += 1
        if t_no % 40 == 0:
            log(f"  {d}｜累計 {n:,} 筆")
    if not n:
        raise SystemExit("沒有可計算的樣本")
    hd = pd.Series(hit_days)
    return {
        "base_p5": round(n5 / n, 4), "base_p30": round(n30 / n, 4), "n": n, "cancelled": cancelled, "anomalies_excluded": anomalies,
        "period": [start, end], "step": step, "universe_top_n": top_n, "min_amount20_twd": floor,
        "stop_rule": "T 收盤 − 1.5 × ATR20", "exit_rules": "plan-v2 主規則", "computed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by_year": {y: {"n": v[0], "p5": round(v[1] / v[0], 4), "p30": round(v[2] / v[0], 4)} for y, v in sorted(by_year.items())},
        "hit_day_quantiles": {q: float(hd.quantile(q)) for q in (0.25, 0.5, 0.75)} if len(hd) else {},
        "hit_day_hist": {str(k): int(v) for k, v in hd.value_counts().sort_index().items()},
    }


def report(r):
    out = [f"v0.9.1 全體基準（{r['period'][0]}～{r['period'][1]}，每 {r['step']} 個交易日抽一天，共 {r['n']:,} 筆）",
           f"  5 日內賺到 +10%：{r['base_p5'] * 100:.1f}%",
           f"  30 日內賺到 +10%：{r['base_p30'] * 100:.1f}%",
           f"  T+1 放棄買進 {r['cancelled']:,} 筆、價格異常排除 {r['anomalies_excluded']:,} 筆（不計入）"]
    for y, v in r["by_year"].items():
        out.append(f"  {y}：{v['n']:,} 筆｜5 日 {v['p5'] * 100:.1f}%｜30 日 {v['p30'] * 100:.1f}%")
    if r.get("hit_day_quantiles"):
        q = r["hit_day_quantiles"]
        out.append(f"  有賺到的，發動日：25% 在第 {q[0.25]:.0f} 天、中位第 {q[0.5]:.0f} 天、75% 在第 {q[0.75]:.0f} 天")
    return "\n".join(out)


def save(r, store, config_path):
    (store.data / "baserate_v091.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg.update(base_p5=r["base_p5"], base_p30=r["base_p30"])
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
