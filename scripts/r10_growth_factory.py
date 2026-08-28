#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10 Growth Factory -- Strict T+1, all-positive-year search.

Goal hierarchy:
1) preserve the locked Strict T+1 execution clock;
2) require every evaluation year 2021..2025 to finish with positive return;
3) maximize CAGR, explicitly testing whether CAGR >= 50% is attainable;
4) drawdown is no longer capped at 15%; a -30% research safety floor remains.

No leverage is introduced: max total stock exposure is capped at 100% NAV.
2020 is warm-up only. Initial capital is NT$1,000,000.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import r10_strategy_factory as base

TARGET_CAGR = 0.50
RESEARCH_DD_FLOOR = -0.30
INITIAL_CAPITAL = 1_000_000.0
ENGINE_VERSION = "AlphaPilot-R10-GrowthFactory-STRICT-T1-v2"
OUT = base.bt.OUT_ROOT / "latest" / "growth_factory_strict_t1"
YEARS = ["2021", "2022", "2023", "2024", "2025"]

# Reuse the proven simulator, but change only research constraints/version.
base.ENGINE_VERSION = ENGINE_VERSION
base.INITIAL_CAPITAL = INITIAL_CAPITAL
base.MAX_DD_FLOOR = RESEARCH_DD_FLOOR
base.bt.INITIAL_CAPITAL = INITIAL_CAPITAL


def sample_params(trial) -> dict:
    r7w = base._norm({
        "p10": trial.suggest_float("r7_w_p10", 0.08, 0.42),
        "p20": trial.suggest_float("r7_w_p20", 0.06, 0.38),
        "p60": trial.suggest_float("r7_w_p60", 0.00, 0.22),
        "pf": trial.suggest_float("r7_w_flow", 0.02, 0.28),
        "pa": trial.suggest_float("r7_w_amtacc", 0.02, 0.24),
        "pc": trial.suggest_float("r7_w_clv", 0.00, 0.18),
        "pn": trial.suggest_float("r7_w_near", 0.00, 0.18),
    })
    p = {
        "r7_weights": r7w,
        "r7_min_amt": trial.suggest_categorical("r7_min_amt_m", [20, 30, 40, 50, 70, 100]) * 1_000_000,
        "r7_nearhigh": trial.suggest_float("r7_nearhigh", 0.68, 0.92, step=0.02),
        "r7_ma_mode": trial.suggest_categorical("r7_ma_mode", ["MA60", "MA120", "TREND"]),
        "r7_base": trial.suggest_float("r7_base", 0.14, 0.35, step=0.01),
        "r7_limit_mult": trial.suggest_float("r7_limit_mult", 0.960, 1.000, step=0.005),
        "r7_rebalance_days": trial.suggest_int("r7_rebalance_days", 5, 25),
        "r7_hard_stop": trial.suggest_float("r7_hard_stop", 0.07, 0.18, step=0.01),
        "r7_rank_exit_mult": trial.suggest_float("r7_rank_exit_mult", 1.25, 4.0, step=0.25),
        "r05_weights": {
            "pclv10": trial.suggest_float("r05_w_clv10", 0.20, 0.75),
            "pamt": trial.suggest_float("r05_w_amt", 0.05, 0.45),
            "pclv5": trial.suggest_float("r05_w_clv5", 0.00, 0.20),
            "pf3": trial.suggest_float("r05_w_f3", -0.05, 0.20),
            "pf10": trial.suggest_float("r05_w_f10", -0.20, 0.10),
            "pt5": trial.suggest_float("r05_w_t5", -0.05, 0.15),
            "pgap": trial.suggest_float("r05_w_gap", -0.40, 0.00),
        },
        "r05_price_low": trial.suggest_categorical("r05_price_low", [5, 8, 10, 12, 15, 20]),
        "r05_price_high": trial.suggest_categorical("r05_price_high", [30, 40, 50, 60, 80, 120]),
        "r05_min_amt": trial.suggest_categorical("r05_min_amt_m", [20, 30, 50, 70, 100, 150]) * 1_000_000,
        "r05_amount_ratio": trial.suggest_float("r05_amount_ratio", 0.7, 1.7, step=0.1),
        "r05_r20_max": trial.suggest_float("r05_r20_max", 0.15, 0.45, step=0.025),
        "r05_ma20gap_max": trial.suggest_float("r05_ma20gap_max", 0.08, 0.30, step=0.025),
        "r05_prior60_min": trial.suggest_float("r05_prior60_min", -0.30, -0.025, step=0.025),
        "r05_base": trial.suggest_float("r05_base", 0.10, 0.30, step=0.01),
        "r05_limit_mult": trial.suggest_float("r05_limit_mult", 0.975, 1.000, step=0.005),
        "r05_max_slots": trial.suggest_int("r05_max_slots", 1, 4),
        "r05_hard_stop": trial.suggest_float("r05_hard_stop", 0.06, 0.16, step=0.01),
        "r05_runner_trigger": trial.suggest_float("r05_runner_trigger", 0.25, 0.55, step=0.05),
        "r05_runner_trail": trial.suggest_float("r05_runner_trail", 0.08, 0.22, step=0.02),
        "r05_base_trail_trigger": trial.suggest_float("r05_base_trail_trigger", 0.25, 0.70, step=0.05),
        "r05_base_trail": trial.suggest_float("r05_base_trail", 0.06, 0.22, step=0.02),
        "r05_max_hold": trial.suggest_int("r05_max_hold", 35, 130, step=5),
        "max_positions": trial.suggest_int("max_positions", 3, 6),
        "max_single": trial.suggest_float("max_single", 0.20, 0.35, step=0.025),
        "max_total": trial.suggest_float("max_total", 0.80, 1.00, step=0.05),
        "strong_exposure": trial.suggest_float("strong_exposure", 0.90, 1.00, step=0.05),
        "strong_slots": trial.suggest_int("strong_slots", 3, 6),
        "normal_exposure": trial.suggest_float("normal_exposure", 0.65, 1.00, step=0.05),
        "normal_slots": trial.suggest_int("normal_slots", 2, 5),
        "repair_exposure": trial.suggest_float("repair_exposure", 0.30, 0.85, step=0.05),
        "repair_slots": trial.suggest_int("repair_slots", 1, 4),
        "weak_exposure": trial.suggest_float("weak_exposure", 0.00, 0.50, step=0.05),
        "weak_slots": trial.suggest_int("weak_slots", 0, 3),
        "dd_level1": trial.suggest_float("dd_level1", -0.08, -0.04, step=0.01),
        "dd_level2": trial.suggest_float("dd_level2", -0.14, -0.08, step=0.01),
        "dd_level3": trial.suggest_float("dd_level3", -0.22, -0.13, step=0.01),
        "dd_mult1": trial.suggest_float("dd_mult1", 0.75, 1.00, step=0.05),
        "dd_mult2": trial.suggest_float("dd_mult2", 0.50, 1.00, step=0.05),
        "dd_mult3": trial.suggest_float("dd_mult3", 0.25, 0.90, step=0.05),
        "force_dd": trial.suggest_float("force_dd", -0.25, -0.10, step=0.01),
        "force_target_exposure": trial.suggest_float("force_target_exposure", 0.25, 0.80, step=0.05),
        "force_no_buy": trial.suggest_int("force_no_buy", 0, 15),
        "force_cooldown": trial.suggest_int("force_cooldown", 5, 30, step=5),
        "candidate_depth": 25,
        "candidate_depth_r7": 8,
    }
    # Enforce monotonic DD thresholds after sampling.
    if p["dd_level2"] >= p["dd_level1"]:
        p["dd_level2"] = p["dd_level1"] - 0.03
    if p["dd_level3"] >= p["dd_level2"]:
        p["dd_level3"] = p["dd_level2"] - 0.04
    p["dd_level3"] = max(p["dd_level3"], -0.26)
    return p


def annual_diagnostics(r: dict) -> tuple[bool, float, int]:
    annual = r.get("annual_returns") or {}
    vals = [float(annual.get(y, -9.0)) for y in YEARS]
    return all(v > 0.0 for v in vals), min(vals), sum(v > 0.0 for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=180)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    strict = base.strict_slot_contract()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strict_t1_contract.json").write_text(json.dumps(strict, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx = base.prepare_context()
    baseline = base.baseline_params()
    baseline_result = base.simulate(ctx, baseline, collect=False)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    rows: list[dict] = []

    def objective(trial):
        p = sample_params(trial)
        try:
            r = base.simulate(ctx, p, collect=False)
            all_pos, worst_year, positive_years = annual_diagnostics(r)
            research_dd_ok = bool(r["finished"] and r["max_dd"] >= RESEARCH_DD_FLOOR - 1e-12)
            target_hit = bool(research_dd_ok and all_pos and r["cagr"] >= TARGET_CAGR)

            # Strongly separate all-positive strategies from those with a losing year.
            # Once all years are positive, CAGR dominates; DD is only a mild soft cost.
            if not research_dd_ok:
                score = -100.0 + float(r["max_dd"])
            elif not all_pos:
                score = -10.0 + positive_years + 4.0 * worst_year + 0.25 * float(r["cagr"])
            else:
                dd_soft = max(0.0, -float(r["max_dd"]) - 0.20)
                score = 10.0 + float(r["cagr"]) + 0.15 * min(worst_year, 0.30) - 0.03 * dd_soft

            for k, v in {
                "end_nav": r["end_nav"], "cagr": r["cagr"], "max_dd": r["max_dd"],
                "trades": r["completed_trades"], "all_years_positive": all_pos,
                "worst_year": worst_year, "positive_years": positive_years,
                "target_50pct_hit": target_hit,
            }.items():
                trial.set_user_attr(k, v)

            rows.append({
                "trial": trial.number, "objective": score, "end_nav": r["end_nav"],
                "cagr": r["cagr"], "max_dd": r["max_dd"], "trades": r["completed_trades"],
                "all_years_positive": all_pos, "worst_year": worst_year,
                "positive_years": positive_years, "target_50pct_hit": target_hit,
                **{f"return_{y}": float((r.get("annual_returns") or {}).get(y, np.nan)) for y in YEARS},
                "params_json": json.dumps(p, ensure_ascii=False, separators=(",", ":")),
            })
            return score
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            rows.append({"trial": trial.number, "objective": -999.0, "error": repr(exc), "all_years_positive": False, "target_50pct_hit": False})
            return -999.0

    study.optimize(objective, n_trials=max(1, args.trials), gc_after_trial=True, show_progress_bar=False)
    trials_df = pd.DataFrame(rows)
    if trials_df.empty:
        raise SystemExit("No completed trials")
    trials_df = trials_df.sort_values(["target_50pct_hit", "all_years_positive", "cagr"], ascending=[False, False, False], na_position="last")
    trials_df.to_csv(OUT / "leaderboard.csv", index=False, encoding="utf-8-sig")

    positive = trials_df[(trials_df.all_years_positive.eq(True)) & (trials_df.max_dd >= RESEARCH_DD_FLOOR - 1e-12)].copy()
    hits = positive[positive.cagr >= TARGET_CAGR - 1e-12].copy()
    if positive.empty:
        best_row = trials_df.iloc[0]
        status = "NO_ALL_POSITIVE_STRATEGY_FOUND"
    else:
        best_row = positive.sort_values(["cagr", "worst_year", "max_dd"], ascending=[False, False, False]).iloc[0]
        status = "TARGET_50PCT_FOUND" if not hits.empty else "ALL_POSITIVE_FOUND_BELOW_50PCT"

    best_params = json.loads(best_row.params_json)
    best = base.simulate(ctx, best_params, collect=True)
    nav_df = best.pop("nav_df")
    trades_df = best.pop("trades_df")
    orders_df = best.pop("orders_df")
    nav_df.to_csv(OUT / "best_daily_nav.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(OUT / "best_trades.csv", index=False, encoding="utf-8-sig")
    orders_df.to_csv(OUT / "best_orders.csv", index=False, encoding="utf-8-sig")
    (OUT / "best_strategy.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    all_pos, worst_year, positive_years = annual_diagnostics(best)
    summary = {
        "status": status,
        "engine_version": ENGINE_VERSION,
        "trials": int(args.trials),
        "goal": {"cagr_target": TARGET_CAGR, "every_year_positive": True, "research_dd_floor": RESEARCH_DD_FLOOR},
        "hard_rules": {
            "warmup": "2020", "evaluation": "2021-2025", "initial_capital_twd": INITIAL_CAPITAL,
            "leverage": False, "max_total_exposure": 1.0,
            "sell_execution": "T decision -> T+1 Open with -0.5% adverse slippage",
            "pending_sell_slot_rule": "occupies slot through T close; released only at T+1 open",
            "buy_execution": "T decision -> precommitted T+1 limit; no touch=no fill",
        },
        "strict_t1_contract": strict,
        "baseline_strict_t1": baseline_result,
        "target_hit_count": int(len(hits)),
        "all_positive_count": int(len(positive)),
        "best": best,
        "best_trial": int(best_row.trial),
        "best_all_years_positive": bool(all_pos),
        "best_worst_year": float(worst_year),
        "best_positive_years": int(positive_years),
        "best_hits_50pct": bool(all_pos and best["cagr"] >= TARGET_CAGR and best["max_dd"] >= RESEARCH_DD_FLOOR),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("GROWTH_FACTORY_RESULT=" + json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
