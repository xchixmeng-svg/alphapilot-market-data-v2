#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import research_fundamental_repricing_v9 as base

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "open_tournament_v11_out"
OUT.mkdir(parents=True, exist_ok=True)

HOLDS = (5, 10, 20, 40, 60)
TOP_NS = (3, 5, 10)
MIN_DEV_TRADES = 60


def perf(t: pd.DataFrame, label: str) -> dict:
    if t.empty:
        return {"slice": label, "trades": 0, "win_rate": 0.0, "mean_net": 0.0, "median_net": 0.0,
                "pf": 0.0, "event_sharpe": 0.0, "compound": 0.0, "max_dd": 0.0, "positive_quarters": 0,
                "median_mae": np.nan, "median_mfe": np.nan}
    x = t.sort_values(["signal_date", "code"]).copy()
    r = x.return_net.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return perf(pd.DataFrame(), label)
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / sd * math.sqrt(min(len(r), 252))) if sd > 1e-12 else 0.0
    q = pd.to_datetime(x.signal_date.astype(str), format="%Y%m%d", errors="coerce").dt.quarter
    qmeans = x.assign(_q=q).groupby("_q").return_net.mean()
    return {
        "slice": label,
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "mean_net": float(r.mean()),
        "median_net": float(r.median()),
        "pf": float(base.pf(r)),
        "event_sharpe": sharpe,
        "compound": float(eq.iloc[-1] - 1.0),
        "max_dd": float(dd.min()),
        "positive_quarters": int((qmeans > 0).sum()),
        "median_mae": float(x.mae.median()) if "mae" in x else np.nan,
        "median_mfe": float(x.mfe.median()) if "mfe" in x else np.nan,
    }


def families(ev: pd.DataFrame):
    eps_ok = ev.eps_improve.fillna(False).astype(bool) if "eps_improve" in ev else pd.Series(False, index=ev.index)
    return {
        "pure_momentum": (
            lambda d: (d.r20_pr >= .80) & (d.r60_pr >= .70) & (d.aclose > d.ma20) & (d.aclose > d.ma60),
            lambda d: .45*d.r20_pr + .35*d.r60_pr + .20*d.amount20_pr),
        "momentum_flow": (
            lambda d: (d.r20_pr >= .70) & (d.r60_pr >= .60) & (d.flow5 > 0) & (d.aclose > d.ma20),
            lambda d: .35*d.r20_pr + .25*d.r60_pr + .25*d.flow5_pr + .15*d.amount20_pr),
        "low_vol_momentum": (
            lambda d: (d.r20_pr >= .65) & (d.r60_pr >= .60) & (d.vol20_pr <= .45) & (d.aclose > d.ma60),
            lambda d: .35*d.r20_pr + .25*d.r60_pr + .25*(1-d.vol20_pr) + .15*d.amount20_pr),
        "trend_pullback": (
            lambda d: (d.r60_pr >= .65) & d.r20.between(-.10, .05) & (d.aclose > d.ma60) & (d.flow5 >= 0),
            lambda d: .35*d.r60_pr + .25*(1-d.r20_pr) + .25*d.flow5_pr + .15*(1-d.vol20_pr)),
        "flow_reversal": (
            lambda d: (d.r20_pr <= .35) & (d.flow_accel_pr >= .75) & (d.flow5 > 0),
            lambda d: .40*d.flow_accel_pr + .25*d.flow5_pr + .20*(1-d.r20_pr) + .15*d.amount20_pr),
        "deep_reversal": (
            lambda d: (d.r20 <= -.08) & (d.r60 <= -.12) & (d.flow5 > 0) & (d.flow_accel > 0),
            lambda d: .35*(1-d.r20_pr) + .25*(1-d.r60_pr) + .25*d.flow_accel_pr + .15*d.flow5_pr),
        "revenue_acceleration": (
            lambda d: (d.rev_rank >= .70) & (d.accel_rank >= .70),
            lambda d: .50*d.accel_rank + .35*d.rev_rank + .15*d.amount20_pr),
        "revenue_flow": (
            lambda d: (d.rev_rank >= .65) & (d.accel_rank >= .60) & (d.flow5 > 0),
            lambda d: .35*d.rev_rank + .30*d.accel_rank + .25*d.flow5_pr + .10*d.amount20_pr),
        "fundamental_price_lag": (
            lambda d: (d.rev_rank >= .75) & (d.accel_rank >= .55) & (d.r60_pr <= .55),
            lambda d: .40*d.rev_rank + .25*d.accel_rank + .25*(1-d.r60_pr) + .10*d.amount20_pr),
        "persistent_growth": (
            lambda d: (d.yoy > 0) & (d.yoy_l1 > 0) & (d.yoy_l2 > 0) & (d.rev_rank >= .65),
            lambda d: .45*d.rev_rank + .25*d.accel_rank + .15*(1-d.vol20_pr) + .15*d.amount20_pr),
        "eps_revenue_confirmation": (
            lambda d: eps_ok.loc[d.index] & (d.rev_rank >= .60),
            lambda d: .40*d.rev_rank + .25*d.accel_rank + .20*d.flow5_pr + .15*d.amount20_pr),
        "cycle_reacceleration": (
            lambda d: (d.yoy > 0) & (d.mom > 0) & (d.accel > 0) & (d.r20_pr <= .70),
            lambda d: .35*d.accel_rank + .30*d.rev_rank + .20*(1-d.r20_pr) + .15*d.flow5_pr),
        "broad_multifactor": (
            lambda d: d.amount20.notna(),
            lambda d: .18*d.rev_rank.fillna(.5) + .16*d.accel_rank.fillna(.5) + .16*d.r20_pr.fillna(.5) +
                      .12*d.r60_pr.fillna(.5) + .16*d.flow5_pr.fillna(.5) + .10*d.flow_accel_pr.fillna(.5) +
                      .07*(1-d.vol20_pr.fillna(.5)) + .05*d.amount20_pr.fillna(.5)),
        "price_lag_flow": (
            lambda d: (d.r60_pr <= .50) & (d.flow5_pr >= .70) & (d.flow_accel_pr >= .60),
            lambda d: .35*(1-d.r60_pr) + .30*d.flow5_pr + .25*d.flow_accel_pr + .10*d.amount20_pr),
    }


def run_cfg(name: str, raw: pd.DataFrame, scoref, top_n: int, hold: int, px_all: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    z = raw.copy()
    z["score"] = scoref(z).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig = z.sort_values(["signal_date", "score", "amount20"], ascending=[True, False, False]).groupby("signal_date", as_index=False).head(top_n)
    old = base.HOLD
    try:
        base.HOLD = hold
        return base.simulate_events(name, sig, px_all)
    finally:
        base.HOLD = old


def main():
    rev, eps = base.read_fundamentals()
    px_all, daily = base.build_daily()
    ev = base.map_events_to_daily(rev, eps, daily)
    ev = ev[(ev.signal_date >= 20240101) & (ev.signal_date <= 20251231)].copy()
    ev = ev[ev.amount20 >= 50_000_000].copy()

    rows = []
    trades_by_cfg = {}
    fams = families(ev)
    for fam, (cond, scoref) in fams.items():
        mask = cond(ev)
        raw = ev[mask.fillna(False)].copy()
        for hold in HOLDS:
            for top_n in TOP_NS:
                cfg = f"{fam}__h{hold}__n{top_n}"
                tr = run_cfg(cfg, raw, scoref, top_n, hold, px_all)
                trades_by_cfg[cfg] = tr
                dev = tr[(tr.signal_date >= 20240101) & (tr.signal_date <= 20241231)] if not tr.empty else pd.DataFrame()
                s = perf(dev, "2024_dev")
                rows.append({"config": cfg, "family": fam, "hold_days": hold, "top_n": top_n, **{"dev_"+k:v for k,v in s.items() if k != "slice"}})

    devdf = pd.DataFrame(rows)
    eligible = devdf[devdf.dev_trades >= MIN_DEV_TRADES].copy()
    if eligible.empty:
        eligible = devdf.copy()
    # No user-preference target. Rank candidates equally across return quality, PF, risk, stability and win rate.
    metrics = {
        "dev_mean_net": True,
        "dev_pf": True,
        "dev_event_sharpe": True,
        "dev_max_dd": True,
        "dev_positive_quarters": True,
        "dev_win_rate": True,
        "dev_median_mae": True,
    }
    score_parts = []
    for c, high_good in metrics.items():
        x = eligible[c].replace([np.inf, -np.inf], np.nan)
        if c in ("dev_max_dd", "dev_median_mae"):
            high_good = True  # less negative is better
        score_parts.append(x.rank(pct=True, ascending=not high_good).fillna(.5))
    eligible["dev_robust_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    eligible = eligible.sort_values(["dev_robust_score", "dev_pf", "dev_mean_net"], ascending=False)
    # Pre-register at most one config per economic family before exposing 2025.
    shortlist = eligible.groupby("family", as_index=False).head(1).head(12).copy()
    shortlist["dev_rank"] = np.arange(1, len(shortlist)+1)

    hold_rows = []
    for r in shortlist.itertuples(index=False):
        tr = trades_by_cfg[r.config]
        ho = tr[(tr.signal_date >= 20250101) & (tr.signal_date <= 20251231)] if not tr.empty else pd.DataFrame()
        hs = perf(ho, "2025_blind")
        hold_rows.append({"config": r.config, "family": r.family, "hold_days": r.hold_days, "top_n": r.top_n,
                          "dev_rank": int(r.dev_rank), "dev_robust_score": float(r.dev_robust_score),
                          **{"holdout_"+k:v for k,v in hs.items() if k != "slice"}})

    hodf = pd.DataFrame(hold_rows).sort_values("dev_rank")
    devdf.to_csv(OUT / "all_dev_configs.csv", index=False)
    shortlist.to_csv(OUT / "preregistered_shortlist.csv", index=False)
    hodf.to_csv(OUT / "blind_2025_results.csv", index=False)

    champ_cfg = str(shortlist.iloc[0].config) if len(shortlist) else None
    champ = None
    if champ_cfg:
        drow = shortlist.iloc[0].to_dict()
        hrow = hodf[hodf.config == champ_cfg].iloc[0].to_dict()
        validated = bool(hrow["holdout_trades"] >= 30 and hrow["holdout_mean_net"] > 0 and hrow["holdout_pf"] > 1.0)
        champ = {"config": champ_cfg, "selected_without_2025": True, "validated_on_2025": validated,
                 "dev": drow, "holdout": hrow}
        t = trades_by_cfg[champ_cfg]
        t.to_csv(OUT / "champion_trades.csv", index=False)
        if validated:
            (OUT / "survivors_present.flag").write_text(champ_cfg + "\n", encoding="utf-8")

    exploratory_best = None
    if len(hodf):
        tmp = hodf.copy()
        tmp["holdout_quality"] = pd.concat([
            tmp.holdout_mean_net.rank(pct=True),
            tmp.holdout_pf.rank(pct=True),
            tmp.holdout_event_sharpe.rank(pct=True),
            tmp.holdout_max_dd.rank(pct=True),
        ], axis=1).mean(axis=1)
        exploratory_best = tmp.sort_values("holdout_quality", ascending=False).iloc[0].to_dict()

    audit = {
        "version": "open-tournament-v11",
        "objective": "find strongest causal strategy without user preference gates",
        "hard_rules_only": ["causal data", "T+1 execution", "realistic fees/tax/slippage", "corporate-action-correct daily layer", "blind 2025 holdout"],
        "selection": "All configurations ranked only on 2024. One configuration per economic family is pre-registered before 2025 is evaluated.",
        "families": list(fams.keys()),
        "hold_days": list(HOLDS),
        "top_n": list(TOP_NS),
        "config_count": int(len(devdf)),
        "eligible_dev_count": int(len(eligible)),
        "shortlist_count": int(len(shortlist)),
        "champion": champ,
        "exploratory_best_holdout_from_preregistered_shortlist": exploratory_best,
        "warning": "The exploratory best holdout is diagnostic only and is not allowed to replace the 2024-selected champion without a new untouched sample."
    }
    (OUT / "tournament_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(shortlist[["dev_rank","config","dev_trades","dev_win_rate","dev_mean_net","dev_pf","dev_event_sharpe","dev_max_dd","dev_robust_score"]].to_string(index=False))
    print("\nBLIND 2025\n", hodf.to_string(index=False))
    print(json.dumps({"champion": champ_cfg, "validated": None if champ is None else champ["validated_on_2025"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
