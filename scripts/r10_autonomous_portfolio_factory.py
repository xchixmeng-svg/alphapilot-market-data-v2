#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10 Autonomous Portfolio Factory.

This layer deliberately removes hand-written portfolio assumptions. It does NOT
pre-fix 62/38, number of holdings, cash reserve, R7/R05 sleeve sizes, or whether
one/both available R10 alpha families are active. Those are optimizer choices.

Only execution reality stays immutable:
- 2020 warm-up; 2021-2025 evaluation.
- NT$1,000,000 starting common cash pool.
- no leverage / no shorting / stock exposure <= 100% NAV.
- T-close decisions; T+1 precommitted buys; sells T+1 open with adverse slippage.
- pending sells occupy slots until T+1 open.
- integer shares and actual cash availability.

The current R10 signal library exposes two alpha families (R7 and R05). This
factory may activate either one or both; all capital remains a COMMON pool, so
there is no fixed strategy bucket. Regime exposure, per-position target size,
position count, cash reserve, rebalance speed, exits and DD response are all
searched.
"""
from __future__ import annotations

import json
from pathlib import Path

import r10_growth_factory as growth

ENGINE_VERSION = "AlphaPilot-R10-AutonomousPortfolio-STRICT-T1-v3"
OUT = growth.base.bt.OUT_ROOT / "latest" / "autonomous_portfolio_strict_t1"

# Keep the research goals from Growth Factory.
growth.ENGINE_VERSION = ENGINE_VERSION
growth.OUT = OUT
growth.base.ENGINE_VERSION = ENGINE_VERSION

_original_sample = growth.sample_params


def sample_params(trial) -> dict:
    # First let the alpha/search layer freely sample signal logic.
    p = _original_sample(trial)

    # Portfolio architecture itself is then independently searched.  These
    # names intentionally differ from v2 parameter names so Optuna can explore
    # a much wider architecture space rather than inheriting old assumptions.
    architecture = trial.suggest_categorical(
        "portfolio_architecture",
        ["R7_ONLY", "R05_ONLY", "MIXED_COMMON_POOL"],
    )

    p["max_positions"] = trial.suggest_int("ap_max_positions", 1, 30)
    p["max_single"] = trial.suggest_float("ap_max_single", 0.03, 1.00, step=0.01)
    p["max_total"] = trial.suggest_float("ap_max_total", 0.10, 1.00, step=0.05)

    # No fixed 62/38 or sleeve budget. These are target sizes per opportunity;
    # actual orders still compete for the same common cash pool at T+1.
    p["r7_base"] = trial.suggest_float("ap_r7_target_per_position", 0.02, 1.00, step=0.01)
    p["r05_base"] = trial.suggest_float("ap_r05_target_per_position", 0.02, 1.00, step=0.01)
    p["r05_max_slots"] = trial.suggest_int("ap_r05_slots", 0, 30)

    # State-dependent capital rotation. A low R7 exposure leaves common-pool
    # capacity for R05/cash; a high value routes capital toward R7. The search
    # decides the rotation profile rather than a human-authored split.
    p["strong_exposure"] = trial.suggest_float("ap_strong_r7_exposure", 0.0, 1.0, step=0.05)
    p["normal_exposure"] = trial.suggest_float("ap_normal_r7_exposure", 0.0, 1.0, step=0.05)
    p["repair_exposure"] = trial.suggest_float("ap_repair_r7_exposure", 0.0, 1.0, step=0.05)
    p["weak_exposure"] = trial.suggest_float("ap_weak_r7_exposure", 0.0, 1.0, step=0.05)
    p["strong_slots"] = trial.suggest_int("ap_strong_r7_slots", 0, 30)
    p["normal_slots"] = trial.suggest_int("ap_normal_r7_slots", 0, 30)
    p["repair_slots"] = trial.suggest_int("ap_repair_r7_slots", 0, 30)
    p["weak_slots"] = trial.suggest_int("ap_weak_r7_slots", 0, 30)

    # Rotation / turnover / order breadth are not fixed either.
    p["r7_rebalance_days"] = trial.suggest_int("ap_r7_rebalance_days", 1, 60)
    p["candidate_depth"] = trial.suggest_int("ap_r05_candidate_depth", 5, 100, step=5)
    p["candidate_depth_r7"] = trial.suggest_int("ap_r7_candidate_depth", 1, 40)

    # Search a much wider exit and risk-response surface. A force threshold
    # below the research floor effectively means 'do not force deleverage'.
    p["r7_hard_stop"] = trial.suggest_float("ap_r7_hard_stop", 0.03, 0.30, step=0.01)
    p["r05_hard_stop"] = trial.suggest_float("ap_r05_hard_stop", 0.03, 0.30, step=0.01)
    p["r05_max_hold"] = trial.suggest_int("ap_r05_max_hold", 10, 250, step=5)
    p["dd_level1"] = trial.suggest_float("ap_dd_level1", -0.15, -0.02, step=0.01)
    p["dd_level2"] = trial.suggest_float("ap_dd_level2", -0.24, -0.06, step=0.01)
    p["dd_level3"] = trial.suggest_float("ap_dd_level3", -0.29, -0.10, step=0.01)
    if p["dd_level2"] >= p["dd_level1"]:
        p["dd_level2"] = p["dd_level1"] - 0.02
    if p["dd_level3"] >= p["dd_level2"]:
        p["dd_level3"] = p["dd_level2"] - 0.02
    p["dd_level3"] = max(-0.29, p["dd_level3"])
    p["dd_mult1"] = trial.suggest_float("ap_dd_mult1", 0.0, 1.0, step=0.05)
    p["dd_mult2"] = trial.suggest_float("ap_dd_mult2", 0.0, 1.0, step=0.05)
    p["dd_mult3"] = trial.suggest_float("ap_dd_mult3", 0.0, 1.0, step=0.05)
    p["force_dd"] = trial.suggest_float("ap_force_dd", -0.35, -0.05, step=0.01)
    p["force_target_exposure"] = trial.suggest_float("ap_force_target_exposure", 0.0, 1.0, step=0.05)
    p["force_no_buy"] = trial.suggest_int("ap_force_no_buy_days", 0, 40)
    p["force_cooldown"] = trial.suggest_int("ap_force_cooldown_days", 1, 60)

    # Architecture switch: allow the optimizer to decide how many currently
    # available alpha families run in parallel. Common cash is still shared.
    if architecture == "R7_ONLY":
        p["r05_max_slots"] = 0
    elif architecture == "R05_ONLY":
        p["strong_exposure"] = p["normal_exposure"] = 0.0
        p["repair_exposure"] = p["weak_exposure"] = 0.0
        p["strong_slots"] = p["normal_slots"] = 0
        p["repair_slots"] = p["weak_slots"] = 0
    else:
        # Prevent a sampled mixed architecture from accidentally disabling one
        # family entirely; allocation magnitudes remain optimizer-selected.
        p["r05_max_slots"] = max(1, p["r05_max_slots"])
        if max(p["strong_slots"], p["normal_slots"], p["repair_slots"], p["weak_slots"]) == 0:
            p["strong_slots"] = 1
        if max(p["strong_exposure"], p["normal_exposure"], p["repair_exposure"], p["weak_exposure"]) <= 0:
            p["strong_exposure"] = 0.05

    # Preserve the architecture label in best_strategy.json/leaderboard while
    # base.simulate safely ignores unknown metadata keys.
    p["portfolio_architecture"] = architecture
    return p


def main() -> None:
    growth.sample_params = sample_params
    growth.main()

    # Enrich summary with explicit proof that allocation was optimizer-owned.
    summary_path = OUT / "summary.json"
    if summary_path.exists():
        q = json.loads(summary_path.read_text(encoding="utf-8"))
        best_path = OUT / "best_strategy.json"
        best_params = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
        q["portfolio_search"] = {
            "fixed_62_38": False,
            "fixed_strategy_buckets": False,
            "common_cash_pool": True,
            "architecture_optimizer_owned": True,
            "available_alpha_families": ["R7", "R05"],
            "chosen_architecture": best_params.get("portfolio_architecture"),
            "chosen_max_positions": best_params.get("max_positions"),
            "chosen_max_single": best_params.get("max_single"),
            "chosen_max_total": best_params.get("max_total"),
            "chosen_r7_target_per_position": best_params.get("r7_base"),
            "chosen_r05_target_per_position": best_params.get("r05_base"),
            "chosen_r05_slots": best_params.get("r05_max_slots"),
            "chosen_r7_regime_exposure": {
                "strong": best_params.get("strong_exposure"),
                "normal": best_params.get("normal_exposure"),
                "repair": best_params.get("repair_exposure"),
                "weak": best_params.get("weak_exposure"),
            },
        }
        summary_path.write_text(json.dumps(q, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("AUTONOMOUS_PORTFOLIO=" + json.dumps(q["portfolio_search"], ensure_ascii=False))


if __name__ == "__main__":
    main()
