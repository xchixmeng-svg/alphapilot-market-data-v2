"""Research-only T+1 entry invalidation study for locked R10 MAX using Fugle 1-minute bars.

Frozen strategy source is NOT modified. Study window is limited to dates where Fugle
minute history is available: 2023-05-23 through 2025-12-31.

Required GitHub secret: FUGLE_API_KEY
API: GET https://api.fugle.tw/marketdata/v1.0/stock/historical/candles/{symbol}
with timeframe=1 and X-API-KEY auth.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("research_out_fugle")
OUT.mkdir(exist_ok=True)

ORDERS = Path("formal_run/r10max_formal_orders.csv")
TRADES = Path("formal_run/r10max_formal_trades.csv")
DAILY = Path("formal_run/ohlcv_causal_2020_2025.csv.gz")
for p in (ORDERS, TRADES, DAILY):
    if not p.exists():
        raise FileNotFoundError(p)

START = 20230523
END = 20251231
WAITS = [0, 5, 10, 15, 20, 30]
THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.095]
REQUEST_SPACING_SEC = 1.25  # stay safely below Fugle basic-tier 60 requests/minute
RATE_LIMIT_BACKOFF_SEC = 65.0

orders = pd.read_csv(ORDERS, dtype={"code": str})
trades = pd.read_csv(TRADES, dtype={"code": str})
daily = pd.read_csv(DAILY, dtype={"code": str}, low_memory=False)
for x in (orders, trades, daily):
    x["code"] = x["code"].astype(str).str.zfill(4)

buys = orders[(orders.side == "BUY") & (orders.status == "FILLED")].copy()
for c in ("scheduled_date", "signal_date", "fill_date"):
    buys[c] = buys[c].astype(int)
buys = buys[(buys.scheduled_date >= START) & (buys.scheduled_date <= END)].copy()

trades["entry_date"] = trades["entry_date"].astype(int)
trades["exit_date"] = trades["exit_date"].astype(int)
keep_trade = ["code", "strategy", "entry_date", "exit_date", "exit_raw_price", "return", "reason"]
merged = buys.merge(
    trades[keep_trade],
    left_on=["code", "strategy", "fill_date"],
    right_on=["code", "strategy", "entry_date"],
    how="left",
    validate="many_to_one",
)
open_pos = merged[merged.exit_date.isna()].copy()
open_pos[["code", "strategy", "signal_date", "scheduled_date", "fill_date"]].to_csv(
    OUT / "excluded_open_positions.csv", index=False
)
buys = merged[merged.exit_date.notna()].copy()

ref = daily[["date", "code", "close"]].copy()
ref["date"] = ref["date"].astype(int)
ref = ref.rename(columns={"date": "signal_date", "close": "signal_close"})
buys = buys.merge(ref, on=["signal_date", "code"], how="left", validate="many_to_one")
if buys.signal_close.isna().any():
    raise RuntimeError("Missing signal-day close")

API_KEY = os.getenv("FUGLE_API_KEY", "").strip()
BASE = "https://api.fugle.tw/marketdata/v1.0/stock/historical/candles"
summary = {
    "study_start": START,
    "study_end": END,
    "filled_buys_in_window": int(len(merged)),
    "completed_buys_scored": int(len(buys)),
    "open_positions_excluded": int(len(open_pos)),
    "minute_source": "Fugle historical candles timeframe=1",
    "api_key_present": bool(API_KEY),
    "request_spacing_sec": REQUEST_SPACING_SEC,
}

if not API_KEY:
    summary["status"] = "FUGLE_API_KEY_MISSING"
    summary["reason"] = "GitHub secret FUGLE_API_KEY is absent."
    (OUT / "minute_data_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(20)

sess = requests.Session()
sess.headers.update({"X-API-KEY": API_KEY})


def fetch_day(code: str, ymd: int):
    d = pd.to_datetime(str(int(ymd))).strftime("%Y-%m-%d")
    url = f"{BASE}/{code}"
    params = {
        "timeframe": "1",
        "from": d,
        "to": d,
        "fields": "open,high,low,close,volume",
        "sort": "asc",
    }
    last = None
    for attempt in range(4):
        try:
            r = sess.get(url, params=params, timeout=30)
            last = {"http": r.status_code, "text": r.text[:300]}
            if r.status_code == 200:
                payload = r.json()
                df = pd.DataFrame(payload.get("data", []))
                if df.empty:
                    return None, {"code": code, "date": ymd, "http": 200, "msg": "empty data"}
                return df, None
            if r.status_code in (401, 402, 403):
                return None, {"code": code, "date": ymd, "http": r.status_code, "msg": r.text[:300]}
            if r.status_code == 404:
                return None, {"code": code, "date": ymd, "http": 404, "msg": "resource not found"}
            if r.status_code == 429:
                time.sleep(RATE_LIMIT_BACKOFF_SEC + 5.0 * attempt)
                continue
        except Exception as e:
            last = {"http": None, "text": repr(e)}
        time.sleep(1.5 * (attempt + 1))
    return None, {"code": code, "date": ymd, "http": last.get("http") if last else None, "msg": last.get("text") if last else "unknown"}

features = []
errors = []
kbar_cache = {}
for _, row in buys.sort_values(["scheduled_date", "code"]).iterrows():
    code = row.code
    ymd = int(row.scheduled_date)
    kb, err = fetch_day(code, ymd)
    if err:
        errors.append(err)
        if err.get("http") in (401, 402, 403):
            break
        time.sleep(REQUEST_SPACING_SEC)
        continue
    kb["ts"] = pd.to_datetime(kb["date"], errors="coerce", utc=True).dt.tz_convert("Asia/Taipei")
    kb["minute"] = kb["ts"].dt.strftime("%H:%M:%S")
    for c in ["open", "high", "low", "close", "volume"]:
        kb[c] = pd.to_numeric(kb[c], errors="coerce")
    kb = kb.sort_values("ts")
    early = kb[(kb.minute >= "09:00:00") & (kb.minute <= "09:30:00")]
    if early.empty:
        errors.append({"code": code, "date": ymd, "msg": "no 09:00-09:30 bars"})
        time.sleep(REQUEST_SPACING_SEC)
        continue
    kbar_cache[(code, ymd)] = kb
    out = {
        "code": code,
        "strategy": row.strategy,
        "signal_date": int(row.signal_date),
        "scheduled_date": ymd,
        "signal_close": float(row.signal_close),
        "limit_price": float(row.limit_price),
        "original_fill_price": float(row.raw_fill_price),
        "original_exit_date": int(row.exit_date),
        "original_exit_price": float(row.exit_raw_price),
        "original_return": float(row["return"]),
        "original_exit_reason": row.reason,
        "open_0900": float(early.iloc[0].open),
        "low_0900_0930": float(early.low.min()),
        "high_0900_0930": float(early.high.max()),
    }
    for m in WAITS:
        t = f"09:{m:02d}:00"
        exact = kb[kb.minute == t]
        out[f"close_{m:02d}"] = float(exact.iloc[0].close) if len(exact) else np.nan
        seen = early[early.minute <= t]
        out[f"min_to_{m:02d}"] = float(seen.low.min()) if len(seen) else np.nan
        out[f"max_to_{m:02d}"] = float(seen.high.max()) if len(seen) else np.nan
    features.append(out)
    time.sleep(REQUEST_SPACING_SEC)

feat = pd.DataFrame(features)
feat.to_csv(OUT / "minute_features.csv", index=False)
summary["received"] = int(len(feat))
summary["errors"] = errors[:30]
if errors and any(e.get("http") in (401, 402, 403) for e in errors):
    summary["status"] = "FUGLE_AUTH_OR_PLAN_FAIL"
elif len(feat) != len(buys):
    summary["status"] = "MINUTE_DATA_INCOMPLETE"
else:
    summary["status"] = "PASS"
(OUT / "minute_data_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
if summary["status"] != "PASS":
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(21)

# Diagnostic grid. At wait minute, if any early-session low has breached -threshold
# versus signal-day close, cancel the trade. Otherwise original frozen limit becomes active
# from that minute onward. No chasing above original limit.
rows = []
for wait in WAITS:
    start_t = f"09:{wait:02d}:00"
    min_col = f"min_to_{wait:02d}"
    for th in THRESHOLDS:
        sim = []
        for _, r in feat.iterrows():
            drop = r[min_col] / r.signal_close - 1.0
            invalid = bool(drop <= -th)
            fill = np.nan
            if not invalid:
                kb = kbar_cache[(r.code, int(r.scheduled_date))]
                after = kb[(kb.minute >= start_t) & (kb.minute <= "13:30:00")]
                hit = after[after.low <= float(r.limit_price)]
                if len(hit):
                    b = hit.iloc[0]
                    fill = float(b.open) if b.open <= float(r.limit_price) else float(r.limit_price)
            ret = float(r.original_exit_price) / fill - 1.0 if np.isfinite(fill) else np.nan
            sim.append((invalid, np.isfinite(fill), ret, drop, fill))
        s = pd.DataFrame(sim, columns=["invalidated", "filled", "return_diag", "early_drop", "sim_fill"])
        kept = s[s.filled]
        rows.append({
            "wait_min": wait,
            "threshold": th,
            "signals": len(s),
            "invalidated": int(s.invalidated.sum()),
            "filled_after_gate": int(s.filled.sum()),
            "unfilled_after_wait": int((~s.invalidated & ~s.filled).sum()),
            "win_rate_diag": float((kept.return_diag > 0).mean()) if len(kept) else np.nan,
            "mean_return_diag": float(kept.return_diag.mean()) if len(kept) else np.nan,
            "median_return_diag": float(kept.return_diag.median()) if len(kept) else np.nan,
            "worst_return_diag": float(kept.return_diag.min()) if len(kept) else np.nan,
            "sum_return_diag": float(kept.return_diag.sum()) if len(kept) else np.nan,
        })

grid = pd.DataFrame(rows)
grid.to_csv(OUT / "entry_gate_grid.csv", index=False)

summ = []
for wait in WAITS:
    col = f"min_to_{wait:02d}"
    for th in THRESHOLDS:
        bad = feat[col] / feat.signal_close - 1 <= -th
        a, b = feat[bad], feat[~bad]
        summ.append({
            "wait_min": wait,
            "threshold": th,
            "breach_count": int(bad.sum()),
            "breach_original_win_rate": float((a.original_return > 0).mean()) if len(a) else np.nan,
            "breach_original_mean_return": float(a.original_return.mean()) if len(a) else np.nan,
            "nonbreach_count": int((~bad).sum()),
            "nonbreach_original_win_rate": float((b.original_return > 0).mean()) if len(b) else np.nan,
            "nonbreach_original_mean_return": float(b.original_return.mean()) if len(b) else np.nan,
        })
threshold = pd.DataFrame(summ)
threshold.to_csv(OUT / "threshold_summary.csv", index=False)

# Candidate gates must have at least 5 invalidations, improve mean return vs no-gate diagnostic,
# and avoid selecting only tiny samples. This is ranking, not a live rule.
baseline = grid[(grid.wait_min == 0) & (grid.threshold == 0.095)].iloc[0]
cand = grid[(grid.invalidated >= 5) & (grid.filled_after_gate >= max(20, int(0.5 * len(feat))))].copy()
cand["mean_return_lift_vs_ref"] = cand.mean_return_diag - float(baseline.mean_return_diag)
cand = cand.sort_values(["mean_return_lift_vs_ref", "win_rate_diag", "worst_return_diag"], ascending=False)
cand.to_csv(OUT / "candidate_gates.csv", index=False)

print("FUGLE_MINUTE_DATA_GATE_PASS")
print("completed trades studied:", len(feat))
print("study window:", START, END)
print(cand.head(20).to_string(index=False))
