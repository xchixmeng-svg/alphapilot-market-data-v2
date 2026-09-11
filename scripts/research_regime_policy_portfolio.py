"""Research-only regime-conditioned policy study for locked R10 MAX.

The frozen R10 classifier is reused exactly as produced by the locked engine:
Strong Bull / Normal Bull / Repair / Weak / Bear/Fallback.  No new classifier is
fit on the test sample.  The research question is whether different PRECOMMITTED
night-before execution/capital policies should apply to different market states.

Unlike the prior intraday-gate experiments, this study does not wait 10-15 minutes.
All policy decisions use signal-day-close information only, so the original T+1
causality is preserved and the full 2021-2025 sample can be tested.

Policy levers are deliberately simple and interpretable:
- permit/block new BUYs by regime and strategy (R7/R0.5)
- state-dependent target-size multiplier
- state-dependent precommitted limit-price discount
- a fixed 0050 20-day realized-volatility overlay, also known by T close

This file never modifies scripts/r10_max_formal.py or the locked formal branch.
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
OUT = ROOT / "research_out_regime_policy"
OUT.mkdir(exist_ok=True)

LOCKED_END_NAV = 2403427.0678222505
LOCKED_TRADES = 150
LOCKED_WINS = 71
LOCKED_LOSSES = 79
LOCKED_MAX_DD = -0.19515438907459803

# Predeclared economic hypotheses, NOT a parameter sweep.
POLICIES = [
    {"name":"control",                 "mode":"control"},
    {"name":"weak_freeze",             "mode":"weak_freeze"},
    {"name":"strategy_split",          "mode":"strategy_split"},
    {"name":"limit_ladder",            "mode":"limit_ladder"},
    {"name":"recovery_bias",           "mode":"recovery_bias"},
    {"name":"balanced_state",          "mode":"balanced_state"},
    {"name":"defensive_state",         "mode":"defensive_state"},
    {"name":"vol_only",                "mode":"vol_only"},
    {"name":"balanced_state_vol",      "mode":"balanced_state_vol"},
    {"name":"strong_normal_free",      "mode":"strong_normal_free"},
]

if not FORMAL_SOURCE.exists() or not SIGNALS.exists():
    raise FileNotFoundError("formal source or formal_run/r10max_signals_final.pkl missing")

src = FORMAL_SOURCE.read_text(encoding="utf-8")
marker = "px = pd.read_pickle('r10max_signals_final.pkl')"
pos = src.find(marker)
if pos < 0:
    raise RuntimeError("could not locate frozen portfolio-engine start marker")

# Reuse only frozen Step-3 portfolio simulator. Signal construction/classification has
# already been reproduced once by the workflow with the locked engine.
sim = "import os, json\nimport pandas as pd\nimport numpy as np\n\n" + src[pos:]

helper = r'''
REGIME_POLICY_MODE = os.getenv("REGIME_POLICY_MODE", "control")

# Add only a causal volatility context available at signal-day close.
px = px.sort_values(["code", "date"]).copy()
_m = px[px.code == "0050"][["date", "aclose"]].drop_duplicates("date").sort_values("date")
_m["policy_mkt_ret1"] = _m["aclose"].pct_change()
_m["policy_annvol20"] = _m["policy_mkt_ret1"].rolling(20, min_periods=20).std() * np.sqrt(252.0)
px = px.merge(_m[["date", "policy_annvol20"]], on="date", how="left")
by_date = {d: sub.set_index("code") for d, sub in px.groupby("date")}

def _vol_mult(annvol):
    if not np.isfinite(annvol):
        return 1.0
    if annvol >= 0.40:
        return 0.50
    if annvol >= 0.30:
        return 0.75
    return 1.0

def regime_policy(strat, regime, annvol):
    """Return size_mult, limit_extra, block, tag. All are fixed ex ante."""
    regime = str(regime)
    mode = REGIME_POLICY_MODE
    size, extra, block = 1.0, 1.0, False

    # Locked Bear/Fallback already carries zero R7 exposure and normally no candidates,
    # but research explicitly blocks to make the policy auditable.
    if regime in ("Bear", "Fallback/Bear", "Unknown"):
        return 0.0, 1.0, True, f"{mode}:{regime}:BLOCK"

    if mode == "control":
        pass

    elif mode == "weak_freeze":
        if regime == "Weak":
            block = True

    elif mode == "strategy_split":
        # R7 already has regime exposure. Apply extra defense only to R0.5.
        if strat == "R05" and regime == "Repair":
            size = 0.60
        elif strat == "R05" and regime == "Weak":
            block = True

    elif mode == "limit_ladder":
        # Keep size unchanged; demand a larger discount in fragile states.
        if regime == "Repair":
            extra = 0.99
        elif regime == "Weak":
            extra = 0.975

    elif mode == "recovery_bias":
        # Do not over-suppress Repair rallies: preserve R7, restrict R0.5; Weak is small.
        if regime == "Repair":
            if strat == "R05":
                size = 0.50
            extra = 0.995
        elif regime == "Weak":
            if strat == "R05":
                block = True
            else:
                size = 0.50
                extra = 0.985

    elif mode == "balanced_state":
        if regime == "Strong Bull":
            size = 1.05
        elif regime == "Normal Bull":
            size = 1.00
        elif regime == "Repair":
            size = 0.70
            extra = 0.995
            if strat == "R05": size *= 0.75
        elif regime == "Weak":
            size = 0.35
            extra = 0.985
            if strat == "R05": block = True

    elif mode == "defensive_state":
        if regime == "Strong Bull":
            size = 1.00
        elif regime == "Normal Bull":
            size = 0.85
        elif regime == "Repair":
            size = 0.55
            extra = 0.99
            if strat == "R05": size *= 0.50
        elif regime == "Weak":
            block = True

    elif mode == "vol_only":
        size *= _vol_mult(annvol)

    elif mode == "balanced_state_vol":
        if regime == "Strong Bull":
            size = 1.05
        elif regime == "Normal Bull":
            size = 1.00
        elif regime == "Repair":
            size = 0.70
            extra = 0.995
            if strat == "R05": size *= 0.75
        elif regime == "Weak":
            size = 0.35
            extra = 0.985
            if strat == "R05": block = True
        size *= _vol_mult(annvol)

    elif mode == "strong_normal_free":
        # Strong/Normal untouched; Repair/Weak defense only.
        if regime == "Repair":
            size = 0.60
            extra = 0.99
            if strat == "R05": size *= 0.50
        elif regime == "Weak":
            if strat == "R05":
                block = True
            else:
                size = 0.30
                extra = 0.98
    else:
        raise RuntimeError(f"unknown REGIME_POLICY_MODE={mode}")

    return float(size), float(extra), bool(block), f"{mode}:{regime}:{strat}"
'''

anchor = "cash = INITIAL_CAPITAL\n"
if anchor not in sim:
    raise RuntimeError("portfolio cash anchor missing")
sim = sim.replace(anchor, helper + anchor, 1)

old = """            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue
                limit = floor_tick(float(r['close']) * (0.98 if strat == 'R7' else 0.995))
                base = (R7_BASE if strat == 'R7' else R05_BASE) * nav * mult
                if strat == 'R7': base *= r7_exposure
                target = min(base, nav * SINGLE_CAP, nav * TOTAL_CAP - mv - created_mv,
                             max(0.0, cash - reserved_cost - 1000.0) / (1 + BUY_FEE))
"""
new = """            for strat, code, r in combined:
                if slots_free <= 0: break
                if code in used: continue
                regime = str(r.get('regime', 'Unknown'))
                annvol = float(r.get('policy_annvol20', np.nan)) if np.isfinite(r.get('policy_annvol20', np.nan)) else np.nan
                policy_size, policy_limit_extra, policy_block, policy_tag = regime_policy(strat, regime, annvol)
                if policy_block or policy_size <= 0:
                    continue
                limit = floor_tick(float(r['close']) * (0.98 if strat == 'R7' else 0.995) * policy_limit_extra)
                base = (R7_BASE if strat == 'R7' else R05_BASE) * nav * mult * policy_size
                if strat == 'R7': base *= r7_exposure
                target = min(base, nav * SINGLE_CAP, nav * TOTAL_CAP - mv - created_mv,
                             max(0.0, cash - reserved_cost - 1000.0) / (1 + BUY_FEE))
"""
if old not in sim:
    raise RuntimeError("exact frozen order-sizing block not found")
sim = sim.replace(old, new, 1)

old_submit = """                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY', limit,
                           nav, post_total, post_code)
"""
new_submit = """                o = submit('BUY', di, exdate, code, strat, shares, 'ENTRY|' + policy_tag, limit,
                           nav, post_total, post_code)
"""
if old_submit not in sim:
    raise RuntimeError("exact frozen BUY submit block not found")
sim = sim.replace(old_submit, new_submit, 1)


def period_stats(nav: pd.DataFrame, start: int, end: int):
    nav = nav.sort_values("date").copy()
    dates = nav["date"].astype(int)
    before = nav[dates < start]
    inside = nav[(dates >= start) & (dates <= end)].copy()
    if inside.empty:
        return np.nan, np.nan
    start_nav = float(before.iloc[-1].nav) if len(before) else float(inside.iloc[0].nav)
    end_nav = float(inside.iloc[-1].nav)
    seeded = pd.concat([pd.Series([start_nav]), inside.nav.reset_index(drop=True)], ignore_index=True)
    dd = float((seeded / seeded.cummax() - 1.0).min())
    return end_nav / start_nav - 1.0, dd

sig = pd.read_pickle(SIGNALS).sort_values(["date", "code"])
# Signal-date regime map is frozen by the formal classifier.
regime_by_date = sig[["date", "regime"]].drop_duplicates("date").set_index("date")["regime"].to_dict()

results, annual_rows, regime_rows = [], [], []
for p in POLICIES:
    vdir = OUT / p["name"]
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    (vdir / "r10max_signals_final.pkl").symlink_to(SIGNALS)
    (vdir / "sim.py").write_text(sim, encoding="utf-8")
    env = os.environ.copy()
    env["REGIME_POLICY_MODE"] = p["mode"]
    print(f"RUN_POLICY {p['name']} mode={p['mode']}", flush=True)
    cp = subprocess.run([sys.executable, "sim.py"], cwd=vdir, env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (vdir / "execution.log").write_text(cp.stdout, encoding="utf-8")
    print(cp.stdout[-2200:], flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"policy {p['name']} failed with exit {cp.returncode}")

    summary = json.loads((vdir / "r10max_formal_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((vdir / "contract_audit.json").read_text(encoding="utf-8"))
    nav = pd.read_csv(vdir / "r10max_formal_nav.csv")
    trades = pd.read_csv(vdir / "r10max_formal_trades.csv")
    orders = pd.read_csv(vdir / "r10max_formal_orders.csv")
    if not audit.get("all_pass", False):
        raise RuntimeError(f"contract audit failed for {p['name']}: {audit}")
    s = summary["strategy"]

    dev_ret, dev_dd = period_stats(nav, 20210104, 20231229)
    val_ret, val_dd = period_stats(nav, 20240101, 20251231)
    worst = float(trades["return"].min()) if len(trades) else np.nan
    p05 = float(trades["return"].quantile(0.05)) if len(trades) else np.nan
    tail15 = int((trades["return"] <= -0.15).sum()) if len(trades) else 0
    tail20 = int((trades["return"] <= -0.20).sum()) if len(trades) else 0

    results.append({
        "policy":p["name"], "mode":p["mode"],
        "end_nav":float(s["end_nav"]), "total_return":float(s["total_return"]), "cagr":float(s["cagr"]),
        "max_drawdown":float(s["max_drawdown"]), "profit_factor":float(s["profit_factor"]),
        "completed_trades":int(s["completed_trades"]), "wins":int(s["wins"]), "losses":int(s["losses"]),
        "win_rate":float(s["win_rate"]), "forced_drawdown_events":int(summary["forced_drawdown_events"]),
        "dev_2021_2023_return":dev_ret, "dev_2021_2023_dd":dev_dd,
        "val_2024_2025_return":val_ret, "val_2024_2025_dd":val_dd,
        "worst_trade_return":worst, "p05_trade_return":p05,
        "tail_loss_le_15pct":tail15, "tail_loss_le_20pct":tail20,
    })
    for row in summary.get("annual", []):
        annual_rows.append({"policy":p["name"], **row})

    # Attribute completed trades to the frozen regime that existed on the BUY signal date.
    filled = orders[(orders.side == "BUY") & (orders.status == "FILLED")][["code","signal_date","fill_date"]].copy()
    if len(filled) and len(trades):
        filled["signal_date"] = filled["signal_date"].astype(int)
        filled["fill_date"] = filled["fill_date"].astype(int)
        filled["regime"] = filled["signal_date"].map(regime_by_date)
        tm = trades.merge(filled, left_on=["code","entry_date"], right_on=["code","fill_date"], how="left")
        for rg, q in tm.groupby("regime", dropna=False):
            pos = q.loc[q.pnl > 0, "pnl"].sum()
            neg = -q.loc[q.pnl < 0, "pnl"].sum()
            regime_rows.append({
                "policy":p["name"], "regime":str(rg), "trades":int(len(q)),
                "mean_return":float(q["return"].mean()), "median_return":float(q["return"].median()),
                "win_rate":float((q.pnl > 0).mean()), "profit_factor":float(pos/neg) if neg > 0 else np.nan,
                "tail15":int((q["return"] <= -0.15).sum()),
            })

res = pd.DataFrame(results)
control = res[res.policy == "control"].iloc[0]
assert abs(float(control.end_nav) - LOCKED_END_NAV) < 1e-7, control.to_dict()
assert int(control.completed_trades) == LOCKED_TRADES, control.to_dict()
assert int(control.wins) == LOCKED_WINS and int(control.losses) == LOCKED_LOSSES, control.to_dict()
assert abs(float(control.max_drawdown) - LOCKED_MAX_DD) < 1e-12, control.to_dict()

for c in ["end_nav","cagr","max_drawdown","profit_factor","dev_2021_2023_return","dev_2021_2023_dd",
          "val_2024_2025_return","val_2024_2025_dd","worst_trade_return","p05_trade_return",
          "tail_loss_le_15pct","tail_loss_le_20pct"]:
    res[f"delta_{c}_vs_control"] = res[c] - float(control[c])

# A candidate must improve risk in validation without destroying validation return/PF.
# This is a reporting gate, not an automatic promotion into formal R10.
res["validation_return_ok"] = res.val_2024_2025_return >= float(control.val_2024_2025_return) - 0.03
res["validation_dd_improved"] = res.val_2024_2025_dd >= float(control.val_2024_2025_dd) + 0.005
res["overall_pf_ok"] = res.profit_factor >= float(control.profit_factor) * 0.92
res["tail_not_worse"] = res.tail_loss_le_15pct <= float(control.tail_loss_le_15pct)
res["research_pass"] = res.validation_return_ok & res.validation_dd_improved & res.overall_pf_ok & res.tail_not_worse
res = res.sort_values(["research_pass","val_2024_2025_dd","val_2024_2025_return","profit_factor","cagr"],
                      ascending=[False,False,False,False,False])
res.to_csv(OUT / "regime_policy_summary.csv", index=False)
pd.DataFrame(annual_rows).to_csv(OUT / "regime_policy_annual.csv", index=False)
pd.DataFrame(regime_rows).to_csv(OUT / "regime_policy_trade_by_regime.csv", index=False)

payload = {
    "status":"PASS",
    "locked_baseline_reproduced":True,
    "classifier":"frozen formal R10 regime classifier; no refit",
    "causality":"all policy choices fixed at signal-day close; no T+1 intraday waiting",
    "development_period":"2021-2023",
    "validation_period":"2024-2025",
    "policies":POLICIES,
    "top_ranked":res.head(8).to_dict(orient="records"),
    "note":"Research only. Formal branch unchanged. A passing row still requires robustness/stress review before any live-rule change.",
}
(OUT / "regime_policy_decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("REGIME_POLICY_RESEARCH_PASS")
print(res.to_string(index=False))
