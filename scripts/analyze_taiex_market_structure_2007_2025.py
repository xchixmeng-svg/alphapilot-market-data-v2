#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

START_YEAR = 2007
END_YEAR = 2025
OUT = Path("taiex_structure_2007_2025")
OUT.mkdir(exist_ok=True)

URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"


def fetch_month(year: int, month: int) -> pd.DataFrame:
    params = {"date": f"{year:04d}{month:02d}01", "response": "json"}
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            obj = r.json()
            if obj.get("stat") != "OK":
                return pd.DataFrame()
            fields = obj.get("fields", [])
            data = obj.get("data", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=fields)
            # Expected fields: 日期, 開盤指數, 最高指數, 最低指數, 收盤指數
            ren = {}
            for c in df.columns:
                if "日期" in c: ren[c] = "date"
                elif "開盤" in c: ren[c] = "open"
                elif "最高" in c: ren[c] = "high"
                elif "最低" in c: ren[c] = "low"
                elif "收盤" in c: ren[c] = "close"
            df = df.rename(columns=ren)
            need = ["date", "open", "high", "low", "close"]
            if any(c not in df.columns for c in need):
                raise RuntimeError(f"unexpected fields: {fields}")
            # ROC date 096/01/02 -> Gregorian
            parts = df["date"].astype(str).str.split("/", expand=True)
            gy = parts[0].astype(int) + 1911
            df["date"] = pd.to_datetime(gy.astype(str) + "-" + parts[1] + "-" + parts[2])
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            return df[need].dropna()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed {year}-{month:02d}: {last_err}")


frames = []
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        m = fetch_month(year, month)
        if not m.empty:
            frames.append(m)
        time.sleep(0.15)

if not frames:
    raise RuntimeError("no TAIEX data fetched")

df = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)
df = df[(df.date.dt.year >= START_YEAR) & (df.date.dt.year <= END_YEAR)].copy()
if df.date.dt.year.nunique() != END_YEAR - START_YEAR + 1:
    raise RuntimeError(f"missing years: {sorted(set(range(START_YEAR, END_YEAR+1)) - set(df.date.dt.year.unique()))}")

df["ret1"] = df.close.pct_change()
df["prior60_high"] = df.close.shift(1).rolling(60, min_periods=60).max()
df["breakout60"] = df.close > df.prior60_high
df["fwd20_ret"] = df.close.shift(-20) / df.close - 1.0
# Count only first day of each breakout run to avoid counting the same trend every day.
df["breakout_event"] = df.breakout60 & ~df.breakout60.shift(1, fill_value=False)
df["breakout20_success"] = np.where(df.breakout_event & df.fwd20_ret.notna(), df.fwd20_ret > 0, np.nan)

rows = []
for year in range(START_YEAR, END_YEAR + 1):
    y = df[df.date.dt.year == year].copy()
    if y.empty:
        continue
    prev = df[df.date < y.date.iloc[0]].tail(1)
    curve = pd.concat([prev[["date", "close"]], y[["date", "close"]]], ignore_index=True) if len(prev) else y[["date", "close"]].copy()
    running_peak = curve.close.cummax()
    dd = curve.close / running_peak - 1.0
    max_dd = float(dd.min())
    daily = y.ret1.dropna()
    ann_vol = float(daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 else np.nan
    up = int((daily > 0).sum())
    down = int((daily < 0).sum())
    flat = int((daily == 0).sum())
    events = y[y.breakout_event & y.fwd20_ret.notna()].copy()
    succ = float((events.fwd20_ret > 0).mean()) if len(events) else np.nan
    avg_fwd = float(events.fwd20_ret.mean()) if len(events) else np.nan
    rows.append({
        "year": year,
        "trading_days": int(len(y)),
        "year_end_close": float(y.close.iloc[-1]),
        "annual_return": float(y.close.iloc[-1] / (prev.close.iloc[-1] if len(prev) else y.close.iloc[0]) - 1.0),
        "max_drawdown": max_dd,
        "annualized_daily_vol": ann_vol,
        "up_days": up,
        "down_days": down,
        "flat_days": flat,
        "up_day_pct_nonflat": float(up / (up + down)) if (up + down) else np.nan,
        "up_down_ratio": float(up / down) if down else np.nan,
        "breakout60_events": int(len(events)),
        "breakout20_success_rate": succ,
        "breakout20_avg_return": avg_fwd,
    })

annual = pd.DataFrame(rows)
annual.to_csv(OUT / "taiex_annual_structure_2007_2025.csv", index=False)
df.to_csv(OUT / "taiex_daily_2007_2025.csv", index=False)

summary = {
    "source": URL,
    "source_owner": "Taiwan Stock Exchange (TWSE)",
    "date_start": str(df.date.min().date()),
    "date_end": str(df.date.max().date()),
    "rows": int(len(df)),
    "definitions": {
        "max_drawdown": "Daily close peak-to-trough; previous year-end close included as the initial anchor for each year.",
        "annualized_daily_vol": "Std dev of daily close-to-close returns multiplied by sqrt(252).",
        "up_day_pct_nonflat": "Up days divided by up+down days; flat days excluded.",
        "breakout_event": "First close above the prior 60-trading-day closing high; consecutive breakout days counted once.",
        "breakout20_success_rate": "Share of breakout events whose close 20 trading days later is above breakout-day close.",
    },
}
(OUT / "methodology.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(annual.to_string(index=False))
print(json.dumps(summary, ensure_ascii=False, indent=2))
