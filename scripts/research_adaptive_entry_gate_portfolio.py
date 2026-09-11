"""Research-only adaptive T+1 entry invalidation study for locked R10 MAX.

Purpose: test whether a prior-night BUY signal can become invalid after the open for
reasons that are richer than a fixed percentage drop. The frozen R10 MAX strategy is
not modified. This script copies only the locked Step-3 portfolio simulator and adds a
causal research hook to T+1 BUY execution from 2023-05-23 onward, where Fugle 1-minute
history is available.

Methods are deliberately interpretable and predeclared: volatility-normalized shock,
relative weakness versus 0050, failed reclaim/VWAP pressure, prior-day strong-to-weak
reversal, gap failure, intraday flush, volume-confirmed pressure, and composite damage
scores. Separate wait-only controls isolate the cost of delaying the order from the
value of the invalidation classifier itself.

Causality: a checkpoint uses only completed 1-minute bars strictly BEFORE that clock
time. Corporate-action dates use the official reference price when available.
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
OUT = ROOT / "research_out_adaptive_gate"
CACHE = OUT / "minute_cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

LOCKED_END_NAV = 2403427.0678222505
LOCKED_TRADES = 150
LOCKED_WINS = 71
LOCKED_LOSSES = 79
LOCKED_MAX_DD = -0.19515438907459803
GATE_START = 20230523

# No parameter sweep. Each row is a different economic hypothesis.
VARIANTS = [
    {"name":"control",              "enabled":False, "wait":0,  "mode":"none"},
    {"name":"wait10_only",          "enabled":True,  "wait":10, "mode":"none"},
    {"name":"wait15_only",          "enabled":True,  "wait":15, "mode":"none"},
    {"name":"zshock_10",            "enabled":True,  "wait":10, "mode":"zshock"},
    {"name":"relweak_10",           "enabled":True,  "wait":10, "mode":"relweak"},
    {"name":"no_reclaim_10",        "enabled":True,  "wait":10, "mode":"no_reclaim"},
    {"name":"strong_reversal_10",   "enabled":True,  "wait":10, "mode":"strong_reversal"},
    {"name":"gap_fail_10",          "enabled":True,  "wait":10, "mode":"gap_fail"},
    {"name":"intraday_flush_10",    "enabled":True,  "wait":10, "mode":"intraday_flush"},
    {"name":"relative_pressure_10", "enabled":True,  "wait":10, "mode":"relative_pressure"},
    {"name":"volume_pressure_10",   "enabled":True,  "wait":10, "mode":"volume_pressure"},
    {"name":"composite3_10",        "enabled":True,  "wait":10, "mode":"composite3"},
    {"name":"composite4_10",        "enabled":True,  "wait":10, "mode":"composite4"},
    {"name":"reclaim_fail_15",      "enabled":True,  "wait":15, "mode":"reclaim15"},
    {"name":"reversal_rel_15",      "enabled":True,  "wait":15, "mode":"reversal15"},
]

if not FORMAL_SOURCE.exists() or not SIGNALS.exists():
    raise FileNotFoundError("formal source or formal_run/r10max_signals_final.pkl missing")

src = FORMAL_SOURCE.read_text(encoding="utf-8")
marker = "px = pd.read_pickle('r10max_signals_final.pkl')"
pos = src.find(marker)
if pos < 0:
    raise RuntimeError("could not locate frozen portfolio-engine start marker")

sim = "import os, json, time\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport requests\n\n" + src[pos:]

helper = r'''
ENTRY_GATE_ENABLED = os.getenv("ENTRY_GATE_ENABLED", "0") == "1"
ENTRY_GATE_WAIT_MIN = int(os.getenv("ENTRY_GATE_WAIT_MIN", "10"))
ENTRY_GATE_MODE = os.getenv("ENTRY_GATE_MODE", "none")
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

# Daily context known at signal-day close. Added columns are research-only and do not
# change any frozen selection, sizing, sell, DD, fee, or accounting rule.
px = px.sort_values(["code", "date"]).copy()
_gate_g = px.groupby("code", group_keys=False)
px["gate_ret1"] = _gate_g["aclose"].transform(lambda s: s.pct_change())
px["gate_sigma20"] = _gate_g["gate_ret1"].transform(lambda s: s.rolling(20, min_periods=20).std())
_range = (px["high"] - px["low"]).replace(0, np.nan)
px["gate_close_loc"] = ((px["close"] - px["low"]) / _range).clip(0, 1).fillna(0.5)
# by_date existed in the frozen engine; rebuild it so signal rows expose research-only fields.
by_date = {d: sub.set_index("code") for d, sub in px.groupby("date")}

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

def _base_price(row, fallback):
    ref = row.get("reference_price", np.nan)
    is_event = bool(row.get("is_official_event", False))
    if is_event and np.isfinite(ref) and float(ref) > 0:
        return float(ref), True
    return float(fallback), False

def _checkpoint_features(code, ymd, signal_date, signal_row, t1_row):
    start_t = f"09:{ENTRY_GATE_WAIT_MIN:02d}:00"
    kb = _minute_day(code, ymd)
    pre = kb[(kb.minute >= "09:00:00") & (kb.minute < start_t)]
    if pre.empty:
        raise RuntimeError(f"no completed bars before {start_t}: {code} {ymd}")

    base, ref_norm = _base_price(t1_row, signal_row["close"])
    first_open = float(pre.iloc[0].open)
    last_close = float(pre.iloc[-1].close)
    early_low = float(pre.low.min())
    early_high = float(pre.high.max())
    pv = (pre["close"] * pre["volume"]).sum()
    vwap = float(pv / pre["volume"].sum()) if float(pre["volume"].sum()) > 0 else last_close
    sigma = float(signal_row.get("gate_sigma20", np.nan))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.02
    sigma = max(sigma, 0.008)

    low_ret = early_low / base - 1.0
    last_ret = last_close / base - 1.0
    gap_ret = first_open / base - 1.0
    z_low = low_ret / sigma
    z_last = last_ret / sigma
    z_gap = gap_ret / sigma
    recovery = (last_close - early_low) / max(early_high - early_low, 1e-9)
    down_share = float((pre["close"] < pre["open"]).mean())
    vol20 = float(signal_row.get("vol20", np.nan))
    early_vol_ratio = float(pre["volume"].sum() / vol20) if np.isfinite(vol20) and vol20 > 0 else np.nan
    prior_ret = float(signal_row.get("gate_ret1", np.nan))
    prior_z = prior_ret / sigma if np.isfinite(prior_ret) else np.nan
    close_loc = float(signal_row.get("gate_close_loc", np.nan))

    # Same-clock 0050 context. This distinguishes stock-specific damage from a broad selloff.
    mkt_sig_sub = by_date.get(int(signal_date))
    mkt_t1_sub = by_date.get(int(ymd))
    mkt_ret = np.nan
    if mkt_sig_sub is not None and mkt_t1_sub is not None and "0050" in mkt_sig_sub.index and "0050" in mkt_t1_sub.index:
        m_sig = mkt_sig_sub.loc["0050"]
        m_t1 = mkt_t1_sub.loc["0050"]
        m_base, _ = _base_price(m_t1, m_sig["close"])
        mkb = _minute_day("0050", ymd)
        mpre = mkb[(mkb.minute >= "09:00:00") & (mkb.minute < start_t)]
        if len(mpre):
            mkt_ret = float(mpre.iloc[-1].close) / m_base - 1.0
    rel_ret = last_ret - mkt_ret if np.isfinite(mkt_ret) else np.nan
    rel_z = rel_ret / sigma if np.isfinite(rel_ret) else np.nan

    return {
        "gate_mode": ENTRY_GATE_MODE,
        "gate_wait_min": ENTRY_GATE_WAIT_MIN,
        "gate_compare_base": base,
        "gate_reference_normalized": ref_norm,
        "gate_sigma20": sigma,
        "gate_prior_ret": prior_ret,
        "gate_prior_z": prior_z,
        "gate_close_loc": close_loc,
        "gate_first_open": first_open,
        "gate_last_close": last_close,
        "gate_early_low": early_low,
        "gate_early_high": early_high,
        "gate_vwap": vwap,
        "gate_low_ret": low_ret,
        "gate_last_ret": last_ret,
        "gate_gap_ret": gap_ret,
        "gate_z_low": z_low,
        "gate_z_last": z_last,
        "gate_z_gap": z_gap,
        "gate_recovery": recovery,
        "gate_down_share": down_share,
        "gate_early_vol_ratio": early_vol_ratio,
        "gate_market_ret": mkt_ret,
        "gate_relative_ret": rel_ret,
        "gate_relative_z": rel_z,
        "gate_below_open": bool(last_close < first_open),
        "gate_below_vwap": bool(last_close < vwap),
        "gate_below_base": bool(last_close < base),
    }, kb

def _invalid(meta):
    zlow = meta["gate_z_low"]
    zlast = meta["gate_z_last"]
    zgap = meta["gate_z_gap"]
    relz = meta["gate_relative_z"]
    priorz = meta["gate_prior_z"]
    loc = meta["gate_close_loc"]
    rec = meta["gate_recovery"]
    down = meta["gate_down_share"]
    vr = meta["gate_early_vol_ratio"]
    bo = meta["gate_below_open"]
    bv = meta["gate_below_vwap"]
    bb = meta["gate_below_base"]

    if ENTRY_GATE_MODE == "none":
        return False, 0
    if ENTRY_GATE_MODE == "zshock":
        return bool(zlow <= -1.50), int(zlow <= -1.50)
    if ENTRY_GATE_MODE == "relweak":
        bad = np.isfinite(relz) and relz <= -1.25 and zlast <= -0.50 and bv
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "no_reclaim":
        bad = zlow <= -1.00 and bo and bv and bb and rec < 0.40
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "strong_reversal":
        bad = np.isfinite(priorz) and priorz >= 1.30 and loc >= 0.75 and zlast <= -0.50 and bo and bv
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "gap_fail":
        bad = zgap <= -0.75 and zlast <= -1.00 and bo and bv
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "intraday_flush":
        high_damage_z = (meta["gate_last_close"] / meta["gate_early_high"] - 1.0) / meta["gate_sigma20"]
        bad = zlow <= -1.25 and high_damage_z <= -0.75 and rec < 0.35 and bv
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "relative_pressure":
        bad = zlow <= -1.00 and np.isfinite(relz) and relz <= -0.75 and down >= 0.60 and bv
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "volume_pressure":
        bad = zlow <= -1.00 and np.isfinite(relz) and relz <= -0.75 and down >= 0.60 and bv and np.isfinite(vr) and vr >= 0.08
        return bool(bad), int(bool(bad))

    score = 0
    score += int(zlow <= -1.25)
    score += int(np.isfinite(relz) and relz <= -0.75)
    score += int(bo)
    score += int(bv)
    score += int(down >= 0.60)
    score += int(rec < 0.40)
    score += int(np.isfinite(priorz) and priorz >= 1.30 and loc >= 0.75)
    if ENTRY_GATE_MODE == "composite3":
        return score >= 3, score
    if ENTRY_GATE_MODE == "composite4":
        return score >= 4, score
    if ENTRY_GATE_MODE == "reclaim15":
        bad = zlow <= -1.00 and zlast <= -0.50 and bo and bv and rec < 0.45
        return bool(bad), int(bool(bad))
    if ENTRY_GATE_MODE == "reversal15":
        bad = np.isfinite(priorz) and priorz >= 1.30 and loc >= 0.75 and np.isfinite(relz) and relz <= -0.75 and bo and bv and rec < 0.50
        return bool(bad), int(bool(bad))
    raise RuntimeError(f"unknown ENTRY_GATE_MODE={ENTRY_GATE_MODE}")

def gated_buy_fill(code, ymd, signal_date, signal_row, t1_row, limit_):
    meta, kb = _checkpoint_features(code, ymd, signal_date, signal_row, t1_row)
    invalid, score = _invalid(meta)
    meta["gate_damage_score"] = score
    if invalid:
        return None, "CANCELED_ADAPTIVE_ENTRY_GATE", meta
    start_t = f"09:{ENTRY_GATE_WAIT_MIN:02d}:00"
    after = kb[(kb.minute >= start_t) & (kb.minute <= "13:30:00")]
    hit = after[after.low <= float(limit_)]
    if hit.empty:
        return None, "UNFILLED_LIMIT_AFTER_ADAPTIVE_GATE", meta
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
                raise RuntimeError(f'missing signal-day row for adaptive gate: {code} {o[\"signal_date\"]}')
            signal_row = signal_sub.loc[code]
            fill, gate_status, gate_meta = gated_buy_fill(code, di, int(o['signal_date']), signal_row, r, o['limit'])
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
    (vdir / "r10max_signals_final.pkl").symlink_to(SIGNALS)
    (vdir / "sim.py").write_text(sim, encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "ENTRY_GATE_ENABLED": "1" if v["enabled"] else "0",
        "ENTRY_GATE_WAIT_MIN": str(v["wait"]),
        "ENTRY_GATE_MODE": v["mode"],
        "ENTRY_GATE_START": str(GATE_START),
        "MINUTE_CACHE_DIR": str(CACHE.resolve()),
    })
    print(f"RUN_VARIANT {v['name']} enabled={v['enabled']} wait={v['wait']} mode={v['mode']}", flush=True)
    cp = subprocess.run([sys.executable, "sim.py"], cwd=vdir, env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (vdir / "execution.log").write_text(cp.stdout, encoding="utf-8")
    print(cp.stdout[-2200:], flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"variant {v['name']} failed with exit {cp.returncode}")

    summary = json.loads((vdir / "r10max_formal_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((vdir / "contract_audit.json").read_text(encoding="utf-8"))
    nav = pd.read_csv(vdir / "r10max_formal_nav.csv")
    trades = pd.read_csv(vdir / "r10max_formal_trades.csv")
    orders = pd.read_csv(vdir / "r10max_formal_orders.csv")
    if not audit.get("all_pass", False):
        raise RuntimeError(f"contract audit failed for {v['name']}: {audit}")
    s = summary["strategy"]

    before = nav[nav.date.astype(int) < GATE_START]
    start_nav = float(before.iloc[-1].nav) if len(before) else float(nav.iloc[0].nav)
    post = nav[nav.date.astype(int) >= GATE_START].copy()
    post_return = float(nav.iloc[-1].nav / start_nav - 1.0)
    seeded = pd.concat([pd.Series([start_nav]), post.nav.reset_index(drop=True)], ignore_index=True)
    post_dd = float((seeded / seeded.cummax() - 1.0).min())

    buys = orders[orders.side == "BUY"].copy()
    post_buys = buys[buys.scheduled_date.astype(int) >= GATE_START]
    canceled = int((post_buys.status == "CANCELED_ADAPTIVE_ENTRY_GATE").sum())
    wait_unfilled = int((post_buys.status == "UNFILLED_LIMIT_AFTER_ADAPTIVE_GATE").sum())
    filled = int((post_buys.status == "FILLED").sum())
    worst = float(trades["return"].min()) if len(trades) else np.nan
    p05 = float(trades["return"].quantile(0.05)) if len(trades) else np.nan
    tail15 = int((trades["return"] <= -0.15).sum()) if len(trades) else 0
    tail20 = int((trades["return"] <= -0.20).sum()) if len(trades) else 0

    results.append({
        "variant":v["name"], "enabled":v["enabled"], "wait_min":v["wait"], "mode":v["mode"],
        "end_nav":float(s["end_nav"]), "total_return":float(s["total_return"]), "cagr":float(s["cagr"]),
        "max_drawdown":float(s["max_drawdown"]), "profit_factor":float(s["profit_factor"]),
        "completed_trades":int(s["completed_trades"]), "wins":int(s["wins"]), "losses":int(s["losses"]),
        "win_rate":float(s["win_rate"]), "forced_drawdown_events":int(summary["forced_drawdown_events"]),
        "post_gate_return":post_return, "post_gate_max_drawdown":post_dd,
        "post_gate_buy_orders":int(len(post_buys)), "gate_canceled":canceled,
        "gate_unfilled_after_wait":wait_unfilled, "post_gate_buy_fills":filled,
        "worst_trade_return":worst, "p05_trade_return":p05,
        "tail_loss_le_15pct":tail15, "tail_loss_le_20pct":tail20,
    })
    for row in summary.get("annual", []):
        if int(row["year"]) >= 2023:
            annual_rows.append({"variant":v["name"], **row})

res = pd.DataFrame(results)
control = res[res.variant == "control"].iloc[0]
assert abs(float(control.end_nav) - LOCKED_END_NAV) < 1e-7, control.to_dict()
assert int(control.completed_trades) == LOCKED_TRADES, control.to_dict()
assert int(control.wins) == LOCKED_WINS and int(control.losses) == LOCKED_LOSSES, control.to_dict()
assert abs(float(control.max_drawdown) - LOCKED_MAX_DD) < 1e-12, control.to_dict()

for c in ["end_nav","cagr","max_drawdown","profit_factor","post_gate_return","post_gate_max_drawdown",
          "worst_trade_return","p05_trade_return","tail_loss_le_15pct","tail_loss_le_20pct"]:
    res[f"delta_{c}_vs_control"] = res[c] - float(control[c])

# Compare classifiers with a same-checkpoint wait-only baseline. This separates the cost
# of delaying the limit order from the incremental value of deciding to cancel it.
wait_ref = {0: control}
for w in [10, 15]:
    q = res[(res.wait_min == w) & (res["mode"] == "none")]
    if len(q): wait_ref[w] = q.iloc[0]
for c in ["end_nav","cagr","max_drawdown","profit_factor","post_gate_return","post_gate_max_drawdown",
          "p05_trade_return","tail_loss_le_15pct","tail_loss_le_20pct"]:
    vals = []
    for _, r in res.iterrows():
        ref = wait_ref.get(int(r.wait_min), control)
        vals.append(float(r[c]) - float(ref[c]))
    res[f"delta_{c}_vs_same_wait"] = vals

ann = pd.DataFrame(annual_rows)
if len(ann):
    ctrl_ann = ann[ann.variant == "control"].set_index("year")["strategy_return"]
    stable = []
    for name in res.variant:
        a = ann[ann.variant == name].set_index("year")["strategy_return"]
        yrs = sorted(set(a.index) & set(ctrl_ann.index))
        stable.append(sum(float(a.loc[y]) >= float(ctrl_ann.loc[y]) for y in yrs))
    res["years_2023_2025_beating_control"] = stable
else:
    res["years_2023_2025_beating_control"] = 0

# Research success prioritizes the actual goal: less portfolio/tail damage without
# sacrificing too much return quality. It is intentionally stricter than picking the
# highest NAV from the same sample.
res["return_floor_ok"] = res.cagr >= float(control.cagr) - 0.0075
res["dd_improved"] = res.max_drawdown >= float(control.max_drawdown) + 0.005
res["tail_improved"] = (res.tail_loss_le_15pct < float(control.tail_loss_le_15pct)) | (res.p05_trade_return > float(control.p05_trade_return) + 0.005)
res["pf_ok"] = res.profit_factor >= float(control.profit_factor) * 0.92
res["research_pass"] = res.return_floor_ok & res.dd_improved & res.tail_improved & res.pf_ok
res = res.sort_values(["research_pass","max_drawdown","tail_loss_le_15pct","cagr","profit_factor"],
                      ascending=[False,False,True,False,False])
res.to_csv(OUT / "adaptive_gate_summary.csv", index=False)
ann.to_csv(OUT / "adaptive_gate_annual.csv", index=False)

payload = {
    "status":"PASS",
    "locked_baseline_reproduced":True,
    "gate_start":GATE_START,
    "causal_checkpoint":"completed 1-minute bars strictly before checkpoint only",
    "same_wait_controls":True,
    "corporate_action_normalization":"official T+1 reference_price when available",
    "methods":[v for v in VARIANTS if v["mode"] != "none"],
    "minute_cache_files":len(list(CACHE.glob("*.json"))),
    "top_ranked":res.head(8).to_dict(orient="records"),
    "note":"Research only. These methods are interpretable hypotheses, not a formal R10 rule and not a claim of out-of-sample proof.",
}
(OUT / "adaptive_gate_decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("ADAPTIVE_ENTRY_GATE_RESEARCH_PASS")
print(res.to_string(index=False))
