"""Research-only full-portfolio test of causal intraday entry invalidation gates.

This does NOT modify the locked R10 MAX engine. It reuses the frozen signal pickle
produced by the locked engine, then extracts the portfolio-simulation section from
scripts/r10_max_formal.py and applies one tightly-scoped research hook to T+1 BUY
execution only.

Causality convention:
- A 09:10 checkpoint may use only completed 1-minute bars timestamped BEFORE 09:10.
- If the completed-bar minimum is <= the chosen drop threshold versus the causal
  comparison base, the T+1 BUY is canceled.
- If not canceled, the original fixed limit becomes active from the checkpoint onward.
- Corporate-action days compare the early price to the official T+1 reference price
  when available, to avoid treating an ex-right/ex-dividend adjustment as a crash.
- Before 2023-05-23, execution is identical to locked R10 because Fugle minute history
  is not available before that date.

The control variant must exactly reproduce the locked baseline or the whole research
run fails.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORMAL_SOURCE = ROOT / "scripts" / "r10_max_formal.py"
FORMAL_RUN = ROOT / "formal_run"
SIGNALS = FORMAL_RUN / "r10max_signals_final.pkl"
OUT = ROOT / "research_out_portfolio_gate"
CACHE = OUT / "minute_cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

LOCKED_END_NAV = 2403427.0678222505
LOCKED_TRADES = 150
LOCKED_WINS = 71
LOCKED_LOSSES = 79
LOCKED_MAX_DD = -0.19515438907459803
GATE_START = 20230523

# Small neighborhood around the diagnostic leader, plus the runner-up family.
# This is deliberately not a giant optimization grid.
VARIANTS = [
    {"name": "control", "enabled": False, "wait": 0, "threshold": 0.0},
    {"name": "0905_m3", "enabled": True, "wait": 5, "threshold": 0.03},
    {"name": "0910_m2", "enabled": True, "wait": 10, "threshold": 0.02},
    {"name": "0910_m3", "enabled": True, "wait": 10, "threshold": 0.03},
    {"name": "0910_m4", "enabled": True, "wait": 10, "threshold": 0.04},
    {"name": "0915_m3", "enabled": True, "wait": 15, "threshold": 0.03},
    {"name": "0915_m4", "enabled": True, "wait": 15, "threshold": 0.04},
    {"name": "0920_m4", "enabled": True, "wait": 20, "threshold": 0.04},
]

if not FORMAL_SOURCE.exists() or not SIGNALS.exists():
    raise FileNotFoundError("formal source or formal_run/r10max_signals_final.pkl missing")

src = FORMAL_SOURCE.read_text(encoding="utf-8")
marker = "px = pd.read_pickle('r10max_signals_final.pkl')"
pos = src.find(marker)
if pos < 0:
    raise RuntimeError("could not locate frozen portfolio-engine start marker")

# Extract only Step 3 (portfolio simulation). Signal construction is already reproduced
# once by the workflow from the locked formal engine.
sim = "import os, json, time\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport requests\n\n" + src[pos:]

helper = r'''
ENTRY_GATE_ENABLED = os.getenv("ENTRY_GATE_ENABLED", "0") == "1"
ENTRY_GATE_WAIT_MIN = int(os.getenv("ENTRY_GATE_WAIT_MIN", "10"))
ENTRY_GATE_THRESHOLD = float(os.getenv("ENTRY_GATE_THRESHOLD", "0.03"))
ENTRY_GATE_START = int(os.getenv("ENTRY_GATE_START", "20230523"))
FUGLE_API_KEY = os.getenv("FUGLE_API_KEY", "").strip()
MINUTE_CACHE_DIR = Path(os.getenv("MINUTE_CACHE_DIR", "../minute_cache"))
MINUTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_SPACING_SEC = 1.25
RATE_LIMIT_BACKOFF_SEC = 65.0
FUGLE_BASE = "https://api.fugle.tw/marketdata/v1.0/stock/historical/candles"
_fugle = requests.Session()
if FUGLE_API_KEY:
    _fugle.headers.update({"X-API-KEY": FUGLE_API_KEY})

def _minute_day(code, ymd):
    cache_file = MINUTE_CACHE_DIR / f"{code}_{int(ymd)}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        if not FUGLE_API_KEY:
            raise RuntimeError("FUGLE_API_KEY missing while minute data cache miss occurred")
        d = pd.to_datetime(str(int(ymd))).strftime("%Y-%m-%d")
        last = None
        for attempt in range(5):
            r = _fugle.get(
                f"{FUGLE_BASE}/{code}",
                params={"timeframe":"1", "from":d, "to":d,
                        "fields":"open,high,low,close,volume", "sort":"asc"},
                timeout=30,
            )
            last = (r.status_code, r.text[:300])
            if r.status_code == 200:
                payload = r.json()
                cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                time.sleep(REQUEST_SPACING_SEC)
                break
            if r.status_code in (401, 402, 403, 404):
                raise RuntimeError(f"Fugle minute fetch failed {code} {ymd}: HTTP {r.status_code} {r.text[:300]}")
            if r.status_code == 429:
                time.sleep(RATE_LIMIT_BACKOFF_SEC + attempt * 5.0)
                continue
            time.sleep(3.0 + attempt * 2.0)
        else:
            raise RuntimeError(f"Fugle minute fetch exhausted retries {code} {ymd}: {last}")
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"empty minute data {code} {ymd}")
    kb = pd.DataFrame(data)
    kb["ts"] = pd.to_datetime(kb["date"], errors="coerce", utc=True).dt.tz_convert("Asia/Taipei")
    kb["minute"] = kb["ts"].dt.strftime("%H:%M:%S")
    for c in ("open", "high", "low", "close", "volume"):
        kb[c] = pd.to_numeric(kb[c], errors="coerce")
    return kb.sort_values("ts")

def gated_buy_fill(code, ymd, signal_close, t1_row, limit_):
    kb = _minute_day(code, ymd)
    start_t = f"09:{ENTRY_GATE_WAIT_MIN:02d}:00"
    pre = kb[(kb.minute >= "09:00:00") & (kb.minute < start_t)]
    # Corporate-action normalization: T+1 reference price is the economically
    # comparable base on ex-right/ex-dividend dates.
    ref = t1_row.get("reference_price", np.nan)
    is_event = bool(t1_row.get("is_official_event", False))
    compare_base = float(ref) if is_event and np.isfinite(ref) and float(ref) > 0 else float(signal_close)
    early_min = float(pre.low.min()) if len(pre) else np.nan
    early_drop = early_min / compare_base - 1.0 if np.isfinite(early_min) and compare_base > 0 else np.nan
    meta = {
        "gate_wait_min": ENTRY_GATE_WAIT_MIN,
        "gate_threshold": ENTRY_GATE_THRESHOLD,
        "gate_compare_base": compare_base,
        "gate_early_min": early_min,
        "gate_early_drop": early_drop,
        "gate_reference_normalized": bool(is_event and np.isfinite(ref) and float(ref) > 0),
    }
    if np.isfinite(early_drop) and early_drop <= -ENTRY_GATE_THRESHOLD:
        return None, "CANCELED_ENTRY_GATE", meta
    after = kb[(kb.minute >= start_t) & (kb.minute <= "13:30:00")]
    hit = after[after.low <= float(limit_)]
    if hit.empty:
        return None, "UNFILLED_LIMIT_AFTER_GATE", meta
    bar = hit.iloc[0]
    fill = buy_fill(float(bar.open), float(bar.low), float(limit_))
    meta["gate_fill_bar_time"] = str(bar.minute)
    return fill, None, meta

'''
anchor = "cash = INITIAL_CAPITAL\n"
if anchor not in sim:
    raise RuntimeError("portfolio cash anchor missing")
sim = sim.replace(anchor, helper + anchor, 1)

old_fill = """        fill = buy_fill(float(r['open']), float(r['low']), o['limit'])
        if fill is None:
            complete_order(o, 'UNFILLED_LIMIT_NOT_TOUCHED')
            continue
"""
new_fill = """        gate_meta = {}
        if ENTRY_GATE_ENABLED and int(di) >= ENTRY_GATE_START:
            signal_sub = by_date.get(int(o['signal_date']))
            if signal_sub is None or code not in signal_sub.index:
                raise RuntimeError(f'missing signal-day row for gate: {code} {o[\"signal_date\"]}')
            signal_close = float(signal_sub.loc[code]['close'])
            fill, gate_status, gate_meta = gated_buy_fill(code, di, signal_close, r, o['limit'])
        else:
            fill, gate_status = buy_fill(float(r['open']), float(r['low']), o['limit']), None
        if fill is None:
            complete_order(o, gate_status or 'UNFILLED_LIMIT_NOT_TOUCHED', **gate_meta)
            continue
"""
if old_fill not in sim:
    raise RuntimeError("exact frozen BUY fill block not found")
sim = sim.replace(old_fill, new_fill, 1)

old_complete = """        complete_order(o, 'FILLED', fill_date=di, raw_fill_price=fill, gross_value=gross,
                       fee=fee, tax=0.0, net_cash=-cost)
"""
new_complete = """        complete_order(o, 'FILLED', fill_date=di, raw_fill_price=fill, gross_value=gross,
                       fee=fee, tax=0.0, net_cash=-cost, **gate_meta)
"""
if old_complete not in sim:
    raise RuntimeError("exact frozen BUY completion block not found")
sim = sim.replace(old_complete, new_complete, 1)

results = []
annual_rows = []
for v in VARIANTS:
    vdir = OUT / v["name"]
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    # symlink is sufficient and avoids duplicating the large signal pickle.
    (vdir / "r10max_signals_final.pkl").symlink_to(SIGNALS)
    (vdir / "sim.py").write_text(sim, encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "ENTRY_GATE_ENABLED": "1" if v["enabled"] else "0",
        "ENTRY_GATE_WAIT_MIN": str(v["wait"]),
        "ENTRY_GATE_THRESHOLD": str(v["threshold"]),
        "ENTRY_GATE_START": str(GATE_START),
        "MINUTE_CACHE_DIR": str(CACHE.resolve()),
    })
    print(f"RUN_VARIANT {v['name']} enabled={v['enabled']} wait={v['wait']} threshold={v['threshold']}", flush=True)
    cp = subprocess.run([sys.executable, "sim.py"], cwd=vdir, env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (vdir / "execution.log").write_text(cp.stdout, encoding="utf-8")
    print(cp.stdout[-2500:], flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"variant {v['name']} failed with exit {cp.returncode}")

    summary = json.loads((vdir / "r10max_formal_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((vdir / "contract_audit.json").read_text(encoding="utf-8"))
    nav = pd.read_csv(vdir / "r10max_formal_nav.csv")
    trades = pd.read_csv(vdir / "r10max_formal_trades.csv")
    orders = pd.read_csv(vdir / "r10max_formal_orders.csv")
    s = summary["strategy"]
    if not audit.get("all_pass", False):
        raise RuntimeError(f"contract audit failed for {v['name']}: {audit}")

    before = nav[nav.date.astype(int) < GATE_START]
    post = nav[nav.date.astype(int) >= GATE_START].copy()
    start_nav = float(before.iloc[-1].nav) if len(before) else float(nav.iloc[0].nav)
    post_return = float(nav.iloc[-1].nav / start_nav - 1.0)
    seeded = pd.concat([pd.Series([start_nav]), post.nav.reset_index(drop=True)], ignore_index=True)
    post_dd = float((seeded / seeded.cummax() - 1.0).min())

    buy_orders = orders[orders.side == "BUY"].copy()
    post_buys = buy_orders[buy_orders.scheduled_date.astype(int) >= GATE_START]
    gate_cancel = int((post_buys.status == "CANCELED_ENTRY_GATE").sum())
    gate_unfilled = int((post_buys.status == "UNFILLED_LIMIT_AFTER_GATE").sum())
    post_fills = int(((post_buys.status == "FILLED")).sum())
    worst_trade = float(trades["return"].min()) if len(trades) else np.nan
    p05_trade = float(trades["return"].quantile(0.05)) if len(trades) else np.nan

    results.append({
        "variant": v["name"], "enabled": v["enabled"], "wait_min": v["wait"],
        "threshold": v["threshold"], "end_nav": float(s["end_nav"]),
        "total_return": float(s["total_return"]), "cagr": float(s["cagr"]),
        "max_drawdown": float(s["max_drawdown"]), "profit_factor": float(s["profit_factor"]),
        "completed_trades": int(s["completed_trades"]), "wins": int(s["wins"]),
        "losses": int(s["losses"]), "win_rate": float(s["win_rate"]),
        "forced_drawdown_events": int(summary["forced_drawdown_events"]),
        "post_gate_return": post_return, "post_gate_max_drawdown": post_dd,
        "post_gate_buy_orders": int(len(post_buys)), "gate_canceled": gate_cancel,
        "gate_unfilled_after_wait": gate_unfilled, "post_gate_buy_fills": post_fills,
        "worst_trade_return": worst_trade, "p05_trade_return": p05_trade,
    })
    for row in summary.get("annual", []):
        if int(row["year"]) >= 2023:
            annual_rows.append({"variant": v["name"], **row})

res = pd.DataFrame(results)
control = res[res.variant == "control"].iloc[0]
# The copied simulation engine must be bit-for-bit equivalent at the portfolio-metric level.
assert abs(float(control.end_nav) - LOCKED_END_NAV) < 1e-7, control.to_dict()
assert int(control.completed_trades) == LOCKED_TRADES, control.to_dict()
assert int(control.wins) == LOCKED_WINS and int(control.losses) == LOCKED_LOSSES, control.to_dict()
assert abs(float(control.max_drawdown) - LOCKED_MAX_DD) < 1e-12, control.to_dict()

for c in ["end_nav", "cagr", "max_drawdown", "profit_factor", "post_gate_return", "post_gate_max_drawdown",
          "worst_trade_return", "p05_trade_return"]:
    res[f"delta_{c}_vs_control"] = res[c] - float(control[c])

# Ranking emphasizes the user's stated purpose: reduce portfolio drawdown without destroying return.
# It is a research ranking, not an automatic live-rule promotion.
res["return_floor_ok"] = res.cagr >= float(control.cagr) - 0.01
res["dd_improved"] = res.max_drawdown > float(control.max_drawdown)
res["pf_not_worse"] = res.profit_factor >= float(control.profit_factor) * 0.95
res["research_pass"] = res.return_floor_ok & res.dd_improved & res.pf_not_worse
res = res.sort_values(["research_pass", "max_drawdown", "cagr", "profit_factor"], ascending=[False, False, False, False])
res.to_csv(OUT / "portfolio_gate_summary.csv", index=False)
pd.DataFrame(annual_rows).to_csv(OUT / "portfolio_gate_annual.csv", index=False)

payload = {
    "status": "PASS",
    "locked_baseline_reproduced": True,
    "gate_start": GATE_START,
    "causal_checkpoint_rule": "checkpoint uses only completed 1-minute bars strictly before checkpoint; limit activates at checkpoint",
    "corporate_action_normalization": "official T+1 reference_price when available",
    "variants": VARIANTS,
    "minute_cache_files": len(list(CACHE.glob("*.json"))),
    "top_ranked": res.head(5).to_dict(orient="records"),
    "note": "Research only. Same 2023-2025 period was used for discovery; do not call this out-of-sample and do not modify locked R10 without separate validation.",
}
(OUT / "portfolio_gate_decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print("PORTFOLIO_GATE_RESEARCH_PASS")
print(res.to_string(index=False))
