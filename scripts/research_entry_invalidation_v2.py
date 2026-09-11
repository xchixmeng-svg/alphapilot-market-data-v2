"""R10 MAX research-only T+1 entry invalidation study v2.

Frozen strategy is never modified.  This script studies whether early T+1 price
failure relative to the prior signal-day close is informative enough to CANCEL a
previously selected BUY.  It excludes the five positions still open at 2025-12-31
because they have no completed-trade outcome to score.

Required minute source: FinMind TaiwanStockKBar, sponsor entitlement.
Set FINMIND_TOKEN as a GitHub Actions secret.  If it is absent, fail explicitly.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("research_out_v2")
OUT.mkdir(exist_ok=True)
ORDERS = Path("formal_run/r10max_formal_orders.csv")
TRADES = Path("formal_run/r10max_formal_trades.csv")
DAILY = Path("formal_run/ohlcv_causal_2020_2025.csv.gz")
for p in (ORDERS, TRADES, DAILY):
    if not p.exists():
        raise FileNotFoundError(p)

orders = pd.read_csv(ORDERS, dtype={"code": str})
trades = pd.read_csv(TRADES, dtype={"code": str})
daily = pd.read_csv(DAILY, dtype={"code": str}, low_memory=False)
for d in (orders, trades, daily):
    d["code"] = d["code"].astype(str).str.zfill(4)

buys = orders[(orders.side == "BUY") & (orders.status == "FILLED")].copy()
buys = buys[(buys.scheduled_date >= 20210101) & (buys.scheduled_date <= 20251231)]
for c in ("scheduled_date", "signal_date", "fill_date"):
    buys[c] = buys[c].astype(int)
trades["entry_date"] = trades["entry_date"].astype(int)
trades["exit_date"] = trades["exit_date"].astype(int)

keep = ["code", "strategy", "entry_date", "exit_date", "exit_raw_price", "return", "reason"]
merged = buys.merge(
    trades[keep],
    left_on=["code", "strategy", "fill_date"],
    right_on=["code", "strategy", "entry_date"],
    how="left",
    validate="many_to_one",
    indicator=True,
)
open_rows = merged[merged["_merge"] != "both"][["code", "strategy", "fill_date"]].copy()
open_rows.to_csv(OUT / "excluded_open_positions.csv", index=False)
buys = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
if len(buys) != len(trades):
    raise RuntimeError(f"Expected {len(trades)} completed-entry matches, got {len(buys)}")

ref = daily[["date", "code", "close"]].copy()
ref["date"] = ref["date"].astype(int)
ref = ref.rename(columns={"date": "signal_date", "close": "signal_close"})
buys = buys.merge(ref, on=["signal_date", "code"], how="left", validate="many_to_one")
if buys.signal_close.isna().any():
    raise RuntimeError("Missing signal-date close")

TOKEN = os.getenv("FINMIND_TOKEN", "").strip()
precheck = {
    "locked_completed_trades": int(len(trades)),
    "filled_buys_total": int(len(merged)),
    "completed_buys_scored": int(len(buys)),
    "open_positions_excluded": int(len(open_rows)),
    "minute_source": "FinMind TaiwanStockKBar",
    "token_present": bool(TOKEN),
}
if not TOKEN:
    precheck["status"] = "MINUTE_DATA_ENTITLEMENT_FAIL"
    precheck["reason"] = "FINMIND_TOKEN GitHub secret is absent; TaiwanStockKBar is sponsor-only."
    (OUT / "minute_data_gate.json").write_text(json.dumps(precheck, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(precheck, ensure_ascii=False, indent=2))
    raise SystemExit(20)

URL = "https://api.finmindtrade.com/api/v4/data"
SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {TOKEN}"})


def fetch_kbar(code: str, ymd: int):
    d = pd.to_datetime(str(ymd)).strftime("%Y-%m-%d")
    params = {"dataset": "TaiwanStockKBar", "data_id": code, "start_date": d}
    last = None
    for attempt in range(4):
        try:
            r = SESSION.get(URL, params=params, timeout=30)
            payload = r.json()
            if r.status_code == 200 and payload.get("status") == 200:
                x = pd.DataFrame(payload.get("data", []))
                if not x.empty:
                    return x, None
                last = {"code": code, "date": ymd, "http": 200, "msg": "empty data"}
            else:
                last = {"code": code, "date": ymd, "http": r.status_code,
                        "status": payload.get("status"), "msg": payload.get("msg")}
                if r.status_code in (401, 402, 403, 429):
                    return None, last
        except Exception as e:
            last = {"code": code, "date": ymd, "http": None, "msg": repr(e)}
        time.sleep(1.5 * (attempt + 1))
    return None, last

waits = [0, 5, 10, 15, 20, 30]
thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.095]
features = []
errors = []
cache = {}

for _, row in buys.sort_values(["scheduled_date", "code"]).iterrows():
    code = row.code
    ymd = int(row.scheduled_date)
    kb, err = fetch_kbar(code, ymd)
    if err:
        errors.append(err)
        if err.get("http") in (401, 402, 403, 429):
            break
        continue
    kb["minute"] = kb["minute"].astype(str)
    for c in ("open", "high", "low", "close", "volume"):
        kb[c] = pd.to_numeric(kb[c], errors="coerce")
    kb = kb.sort_values("minute")
    kb = kb[(kb.minute >= "09:00:00") & (kb.minute <= "13:30:00")].copy()
    early = kb[kb.minute <= "09:30:00"]
    if early.empty:
        errors.append({"code": code, "date": ymd, "msg": "no 09:00-09:30 bars"})
        continue
    cache[(code, ymd)] = kb
    rec = {
        "code": code, "strategy": row.strategy,
        "signal_date": int(row.signal_date), "scheduled_date": ymd,
        "signal_close": float(row.signal_close), "limit_price": float(row.limit_price),
        "original_fill_price": float(row.raw_fill_price),
        "original_exit_date": int(row.exit_date), "original_exit_price": float(row.exit_raw_price),
        "original_return": float(row["return"]), "original_exit_reason": row.reason,
        "open_0900": float(early.iloc[0].open),
    }
    for w in waits:
        t = f"09:{w:02d}:00"
        seen = early[early.minute <= t]
        rec[f"min_to_{w:02d}"] = float(seen.low.min()) if len(seen) else np.nan
        at = kb[kb.minute == t]
        rec[f"close_{w:02d}"] = float(at.iloc[0].close) if len(at) else np.nan
    features.append(rec)
    time.sleep(0.08)

feat = pd.DataFrame(features)
feat.to_csv(OUT / "minute_features.csv", index=False)
status = dict(precheck)
status.update({"requested": int(len(buys)), "received": int(len(feat)), "errors": errors[:20]})
if errors and any(e.get("http") in (401, 402, 403) for e in errors):
    status["status"] = "MINUTE_DATA_ENTITLEMENT_FAIL"
elif errors and any(e.get("http") == 429 for e in errors):
    status["status"] = "MINUTE_DATA_RATE_LIMIT_FAIL"
elif len(feat) != len(buys):
    status["status"] = "MINUTE_DATA_INCOMPLETE"
else:
    status["status"] = "PASS"
(OUT / "minute_data_gate.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
if status["status"] != "PASS":
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(21)

# A. Does an early breach identify historically bad completed trades?
summary = []
for w in waits:
    drop = feat[f"min_to_{w:02d}"] / feat.signal_close - 1.0
    for th in thresholds:
        breach = drop <= -th
        a, b = feat[breach], feat[~breach]
        summary.append({
            "wait_min": w, "threshold": th,
            "breach_count": int(breach.sum()),
            "breach_win_rate": float((a.original_return > 0).mean()) if len(a) else np.nan,
            "breach_mean_return": float(a.original_return.mean()) if len(a) else np.nan,
            "breach_median_return": float(a.original_return.median()) if len(a) else np.nan,
            "nonbreach_count": int((~breach).sum()),
            "nonbreach_win_rate": float((b.original_return > 0).mean()) if len(b) else np.nan,
            "nonbreach_mean_return": float(b.original_return.mean()) if len(b) else np.nan,
            "win_rate_gap": (float((b.original_return > 0).mean()) - float((a.original_return > 0).mean())) if len(a) and len(b) else np.nan,
            "mean_return_gap": (float(b.original_return.mean()) - float(a.original_return.mean())) if len(a) and len(b) else np.nan,
        })
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "threshold_summary.csv", index=False)

# B. If we WAIT and retain the original frozen limit, what fills would remain?
grid = []
for w in waits:
    start_t = f"09:{w:02d}:00"
    drop_series = feat[f"min_to_{w:02d}"] / feat.signal_close - 1.0
    for th in thresholds:
        rets = []
        invalid = 0
        unfilled = 0
        for idx, r in feat.iterrows():
            if drop_series.loc[idx] <= -th:
                invalid += 1
                continue
            kb = cache[(r.code, int(r.scheduled_date))]
            after = kb[kb.minute >= start_t]
            hit = after[after.low <= float(r.limit_price)]
            if hit.empty:
                unfilled += 1
                continue
            bar = hit.iloc[0]
            fill = float(bar.open) if float(bar.open) <= float(r.limit_price) else float(r.limit_price)
            rets.append(float(r.original_exit_price) / fill - 1.0)
        s = pd.Series(rets, dtype=float)
        grid.append({
            "wait_min": w, "threshold": th, "signals": int(len(feat)),
            "invalidated": int(invalid), "filled_after_gate": int(len(s)), "unfilled_after_wait": int(unfilled),
            "win_rate_diag": float((s > 0).mean()) if len(s) else np.nan,
            "mean_return_diag": float(s.mean()) if len(s) else np.nan,
            "median_return_diag": float(s.median()) if len(s) else np.nan,
            "worst_return_diag": float(s.min()) if len(s) else np.nan,
        })
grid_df = pd.DataFrame(grid)
grid_df.to_csv(OUT / "entry_gate_grid.csv", index=False)

# Rank only cells with >=10 breached cases so tiny samples cannot look artificially best.
rank = summary_df[summary_df.breach_count >= 10].copy()
rank["score"] = rank["win_rate_gap"].fillna(-999) + rank["mean_return_gap"].fillna(-999)
rank = rank.sort_values(["score", "breach_count"], ascending=[False, False])
rank.head(20).to_csv(OUT / "candidate_gates.csv", index=False)

print("MINUTE_DATA_PASS")
print(f"completed={len(buys)} open_excluded={len(open_rows)}")
print(rank.head(20).to_string(index=False))
