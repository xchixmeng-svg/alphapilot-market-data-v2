#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10 Monster Stock Factory -- Strict T+1.

Purpose: feed the optimizer explicit pre-runner / breakout logic instead of only
letting it reallocate the legacy R7/R05 families.

The MONSTER family is causal and uses only information known by T close:
- price/relative-strength acceleration (5/10/20D),
- 20D/60D breakout proximity,
- pre-breakout compression,
- turnover/amount expansion,
- close-location accumulation,
- foreign/trust acceleration,
- trend alignment and anti-chase extension limits.

Execution reality is inherited unchanged from r10_strategy_factory:
T-close decision -> precommitted T+1 buy limit; sells T+1 open; queued sells keep
slots through T close; integer shares; common cash pool; no leverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import r10_autonomous_portfolio_factory as auto
import r10_growth_factory as growth
import r10_strategy_factory as base

core = base.core
ENGINE_VERSION = "AlphaPilot-R10-MonsterStockFactory-STRICT-T1-v1"
OUT = base.bt.OUT_ROOT / "latest" / "monster_stock_factory_strict_t1"

# Restore the research target after importing Autonomous Portfolio Factory.
growth.ENGINE_VERSION = ENGINE_VERSION
growth.OUT = OUT
growth.base.ENGINE_VERSION = ENGINE_VERSION
base.ENGINE_VERSION = ENGINE_VERSION
base.OUT = OUT

_ORIG_PREPARE = base.prepare_context
_ORIG_CANDIDATES = base.candidates_for_day
_ORIG_SAMPLE = auto.sample_params


def _safe_col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def prepare_context() -> dict:
    ctx = _ORIG_PREPARE()
    f = ctx["feat"].copy().sort_values(["code", "date"]).reset_index(drop=True)
    g = f.groupby("code", sort=False, group_keys=False)

    # All rolling features are backward-looking; shifted highs prevent using T's
    # own close as the historical breakout reference.
    f["ms_r5"] = g["aclose"].transform(lambda s: s.pct_change(5, fill_method=None))
    f["ms_r10"] = g["aclose"].transform(lambda s: s.pct_change(10, fill_method=None))
    f["ms_r20"] = g["aclose"].transform(lambda s: s.pct_change(20, fill_method=None))
    f["ms_prev_high20"] = g["aclose"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    f["ms_prev_high60"] = g["aclose"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).max())
    f["ms_prev_low20"] = g["aclose"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).min())
    f["ms_std20"] = g["aclose"].transform(lambda s: s.pct_change(fill_method=None).rolling(20, min_periods=10).std())
    f["ms_break20"] = f["aclose"] / f["ms_prev_high20"]
    f["ms_break60"] = f["aclose"] / f["ms_prev_high60"]
    f["ms_range20"] = f["ms_prev_high20"] / f["ms_prev_low20"] - 1.0
    f["ms_accel"] = f["ms_r5"] - 0.25 * f["ms_r20"]

    # Reuse causal R10 fields already present in the historical database.
    f["ms_amount_ratio"] = _safe_col(f, "amount_ratio", 1.0)
    f["ms_clv"] = _safe_col(f, "clvflow10", 0.0)
    f["ms_ma20gap"] = _safe_col(f, "ma20gap", np.nan)
    if f["ms_ma20gap"].isna().all():
        f["ms_ma20gap"] = f["aclose"] / _safe_col(f, "ma60") - 1.0

    ins = ctx.get("ins", pd.DataFrame()).copy()
    if not ins.empty:
        cols = [c for c in ["date", "code", "Foreign3D", "Foreign10D", "Trust5D"] if c in ins.columns]
        ins = ins[cols].sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
        f = f.merge(ins, on=["date", "code"], how="left", suffixes=("", "_inst"))
    for c in ["Foreign3D", "Foreign10D", "Trust5D"]:
        if c not in f.columns:
            f[c] = 0.0
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
    f["ms_foreign_accel"] = f["Foreign3D"] / 3.0 - f["Foreign10D"] / 10.0

    monster_days: dict[int, pd.DataFrame] = {}
    for di in ctx["eval_dates"]:
        x = f[f.date.eq(di)].copy()
        if x.empty:
            monster_days[int(di)] = x
            continue
        # Cross-sectional percentiles make heterogeneous signals comparable.
        factors = {
            "p_accel": x["ms_accel"],
            "p_break20": x["ms_break20"],
            "p_break60": x["ms_break60"],
            "p_compress": -x["ms_std20"],
            "p_tight": -x["ms_range20"],
            "p_vol": x["ms_amount_ratio"],
            "p_clv": x["ms_clv"],
            "p_foreign": x["ms_foreign_accel"],
            "p_trust": x["Trust5D"],
            "p_rs20": x["ms_r20"],
            "p_runway": -x["ms_ma20gap"],
        }
        for dst, s in factors.items():
            x[dst] = core.pr(s)
        keep = [
            "code", "name", "close", "aclose", "ma60", "ma120", "amt20", "avgvol20",
            "ms_r5", "ms_r10", "ms_r20", "ms_break20", "ms_break60", "ms_range20", "ms_std20",
            "ms_amount_ratio", "ms_clv", "ms_ma20gap", "ms_foreign_accel", "Trust5D",
            "p_accel", "p_break20", "p_break60", "p_compress", "p_tight", "p_vol", "p_clv",
            "p_foreign", "p_trust", "p_rs20", "p_runway",
        ]
        monster_days[int(di)] = x[keep].copy()

    ctx["feat"] = f
    ctx["feat_idx"] = f.set_index(["date", "code"]).sort_index()
    ctx["monster_days"] = monster_days

    # Monster entries are allowed in constructive Repair/Weak regimes too;
    # candidate filters still decide whether an individual stock qualifies.
    for di, st in ctx["states"].items():
        st["r05_risk_on"] = st["regime"] in {"Strong Bull", "Normal Bull", "Repair", "Weak"}
    return ctx


def monster_weights(p: dict) -> dict[str, float]:
    default = {
        "accel": 0.16, "break20": 0.15, "break60": 0.08, "compress": 0.08,
        "tight": 0.07, "vol": 0.13, "clv": 0.09, "foreign": 0.08,
        "trust": 0.04, "rs20": 0.08, "runway": 0.04,
    }
    return p.get("monster_weights", default)


def candidates_for_day(ctx: dict, di: int, p: dict):
    r7, _legacy_r05, expo, slots = _ORIG_CANDIDATES(ctx, di, p)
    st = ctx["states"][di]
    y = ctx["monster_days"].get(di, pd.DataFrame())
    if y.empty or not st["r05_risk_on"]:
        return r7, y.iloc[0:0].copy(), expo, slots

    w = monster_weights(p)
    score = (
        w["accel"] * y.p_accel + w["break20"] * y.p_break20 + w["break60"] * y.p_break60
        + w["compress"] * y.p_compress + w["tight"] * y.p_tight + w["vol"] * y.p_vol
        + w["clv"] * y.p_clv + w["foreign"] * y.p_foreign + w["trust"] * y.p_trust
        + w["rs20"] * y.p_rs20 + w["runway"] * y.p_runway
    )
    trend_mode = p.get("monster_trend_mode", "MA60")
    mask = (
        y.close.between(p.get("monster_price_low", 8.0), p.get("monster_price_high", 6000.0))
        & (y.amt20 >= p.get("monster_min_amt", 30_000_000.0))
        & (y.ms_amount_ratio >= p.get("monster_amount_ratio", 1.0))
        & y.ms_r20.between(p.get("monster_r20_min", 0.00), p.get("monster_r20_max", 0.45))
        & (y.ms_break20 >= p.get("monster_break20_min", 0.96))
        & (y.ms_break20 <= p.get("monster_break20_max", 1.10))
        & (y.ms_ma20gap <= p.get("monster_ma20gap_max", 0.28))
        & (y.ms_range20 <= p.get("monster_range20_max", 0.45))
        & (y.ms_std20 <= p.get("monster_std20_max", 0.055))
        & score.notna()
    )
    if trend_mode == "TREND":
        mask &= (y.aclose > y.ma60) & (y.ma60 > y.ma120)
    elif trend_mode == "MA120":
        mask &= y.aclose > y.ma120
    else:
        mask &= y.aclose > y.ma60
    if p.get("monster_require_institution", False):
        inst_score = 0.65 * y.p_foreign + 0.35 * y.p_trust
        mask &= inst_score >= p.get("monster_inst_pct_min", 0.45)

    cc = y.loc[mask].copy()
    cc["score"] = score.loc[mask]
    cc = cc.sort_values(["score", "code"], ascending=[False, True])
    cc["rank"] = np.arange(1, len(cc) + 1)
    return r7, cc, expo, slots


def monster_exit_reason(pos, row, p: dict):
    if row is None or not np.isfinite(row.aclose):
        return None
    adj = float(row.aclose)
    ret = adj / pos.entry_adj - 1.0
    pos.peak_adj = max(pos.peak_adj, adj)
    pos.hold_days += 1
    eps = 1e-12

    hard = p.get("r05_hard_stop", 0.09)
    if adj <= pos.entry_adj * (1.0 - hard) + eps:
        return "MONSTER_HARD"

    # Fast failure: a supposed breakout that cannot hold quickly is discarded.
    if pos.hold_days >= p.get("monster_fail_days", 7) and ret <= p.get("monster_fail_ret", -0.02):
        return "MONSTER_FAILED_BREAKOUT"

    runner_trigger = p.get("r05_runner_trigger", 0.22)
    if ret >= runner_trigger - eps:
        pos.mode = "RUNNER"
    mega_trigger = p.get("monster_mega_trigger", 0.55)
    if ret >= mega_trigger - eps:
        pos.mode = "MEGA"

    if pos.mode == "MEGA":
        trail = p.get("monster_mega_trail", 0.16)
        if adj <= pos.peak_adj * (1.0 - trail) + eps:
            return "MONSTER_MEGA_TRAIL"
    elif pos.mode == "RUNNER":
        trail = p.get("r05_runner_trail", 0.12)
        if adj <= pos.peak_adj * (1.0 - trail) + eps:
            return "MONSTER_RUNNER_TRAIL"
    else:
        trigger = p.get("r05_base_trail_trigger", 0.14)
        trail = p.get("r05_base_trail", 0.08)
        if ret >= trigger - eps and adj <= pos.peak_adj * (1.0 - trail) + eps:
            return "MONSTER_BASE_TRAIL"

    if pos.hold_days >= p.get("r05_max_hold", 120):
        return "MONSTER_TIME"
    return None


def sample_params(trial) -> dict:
    p = _ORIG_SAMPLE(trial)

    # The old R05 sleeve is deliberately repurposed as the new MONSTER family.
    # Price ceiling now reaches high-priced leaders; no low-price bias is forced.
    mw = base._norm({
        "accel": trial.suggest_float("ms_w_accel", 0.04, 0.28),
        "break20": trial.suggest_float("ms_w_break20", 0.04, 0.26),
        "break60": trial.suggest_float("ms_w_break60", 0.00, 0.18),
        "compress": trial.suggest_float("ms_w_compress", 0.00, 0.18),
        "tight": trial.suggest_float("ms_w_tight", 0.00, 0.16),
        "vol": trial.suggest_float("ms_w_volume", 0.04, 0.24),
        "clv": trial.suggest_float("ms_w_clv", 0.00, 0.18),
        "foreign": trial.suggest_float("ms_w_foreign", 0.00, 0.18),
        "trust": trial.suggest_float("ms_w_trust", 0.00, 0.12),
        "rs20": trial.suggest_float("ms_w_rs20", 0.02, 0.20),
        "runway": trial.suggest_float("ms_w_runway", 0.00, 0.14),
    })
    p["monster_weights"] = mw
    p["monster_price_low"] = trial.suggest_categorical("ms_price_low", [5, 8, 10, 20, 30, 50])
    p["monster_price_high"] = trial.suggest_categorical("ms_price_high", [80, 150, 300, 600, 1200, 3000, 6000])
    p["monster_min_amt"] = trial.suggest_categorical("ms_min_amt_m", [20, 30, 50, 70, 100, 150]) * 1_000_000
    p["monster_amount_ratio"] = trial.suggest_float("ms_amount_ratio", 0.8, 2.0, step=0.1)
    p["monster_r20_min"] = trial.suggest_float("ms_r20_min", -0.05, 0.15, step=0.025)
    p["monster_r20_max"] = trial.suggest_float("ms_r20_max", 0.20, 0.65, step=0.025)
    if p["monster_r20_max"] <= p["monster_r20_min"]:
        p["monster_r20_max"] = p["monster_r20_min"] + 0.20
    p["monster_break20_min"] = trial.suggest_float("ms_break20_min", 0.92, 1.04, step=0.02)
    p["monster_break20_max"] = trial.suggest_float("ms_break20_max", 1.02, 1.20, step=0.02)
    if p["monster_break20_max"] <= p["monster_break20_min"]:
        p["monster_break20_max"] = p["monster_break20_min"] + 0.06
    p["monster_ma20gap_max"] = trial.suggest_float("ms_ma20gap_max", 0.08, 0.40, step=0.02)
    p["monster_range20_max"] = trial.suggest_float("ms_range20_max", 0.12, 0.60, step=0.04)
    p["monster_std20_max"] = trial.suggest_float("ms_std20_max", 0.025, 0.080, step=0.005)
    p["monster_trend_mode"] = trial.suggest_categorical("ms_trend_mode", ["MA60", "MA120", "TREND"])
    p["monster_require_institution"] = trial.suggest_categorical("ms_require_inst", [False, True])
    p["monster_inst_pct_min"] = trial.suggest_float("ms_inst_pct_min", 0.35, 0.75, step=0.05)

    # Runner logic: cut failed breakouts, but let genuine multi-baggers breathe.
    p["r05_hard_stop"] = trial.suggest_float("ms_hard_stop", 0.05, 0.14, step=0.01)
    p["r05_runner_trigger"] = trial.suggest_float("ms_runner_trigger", 0.12, 0.35, step=0.025)
    p["r05_runner_trail"] = trial.suggest_float("ms_runner_trail", 0.08, 0.20, step=0.02)
    p["r05_base_trail_trigger"] = trial.suggest_float("ms_base_trail_trigger", 0.10, 0.30, step=0.025)
    p["r05_base_trail"] = trial.suggest_float("ms_base_trail", 0.05, 0.14, step=0.01)
    p["monster_mega_trigger"] = trial.suggest_float("ms_mega_trigger", 0.40, 1.00, step=0.05)
    p["monster_mega_trail"] = trial.suggest_float("ms_mega_trail", 0.10, 0.24, step=0.02)
    p["monster_fail_days"] = trial.suggest_int("ms_fail_days", 4, 15)
    p["monster_fail_ret"] = trial.suggest_float("ms_fail_ret", -0.08, 0.02, step=0.01)
    p["r05_max_hold"] = trial.suggest_int("ms_max_hold", 40, 250, step=10)
    p["candidate_depth"] = trial.suggest_int("ms_candidate_depth", 5, 80, step=5)

    arch = p.get("portfolio_architecture")
    if arch == "R05_ONLY":
        p["portfolio_architecture"] = "MONSTER_ONLY"
    elif arch == "MIXED_COMMON_POOL":
        p["portfolio_architecture"] = "R7_MONSTER_COMMON_POOL"
    elif arch == "R7_ONLY":
        p["portfolio_architecture"] = "R7_ONLY"
    return p


def main() -> None:
    base.prepare_context = prepare_context
    base.candidates_for_day = candidates_for_day
    base.r05_exit_reason = monster_exit_reason
    growth.sample_params = sample_params
    growth.main()

    sp = OUT / "summary.json"
    if sp.exists():
        q = json.loads(sp.read_text(encoding="utf-8"))
        q["alpha_library"] = {
            "legacy_r7": True,
            "legacy_r05": False,
            "monster_family": True,
            "monster_logic": [
                "5/10/20D momentum acceleration",
                "20D/60D breakout proximity",
                "pre-breakout compression/tightness",
                "turnover expansion",
                "close-location accumulation",
                "foreign/trust acceleration",
                "trend alignment",
                "anti-chase extension cap",
                "failed-breakout early exit",
                "runner/mega trailing exits",
                "high-priced leaders allowed",
            ],
            "execution_semantics": "Strict T+1 inherited unchanged",
        }
        sp.write_text(json.dumps(q, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("MONSTER_ALPHA=" + json.dumps(q["alpha_library"], ensure_ascii=False))


if __name__ == "__main__":
    main()
