#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "independent_ai_buy_selector_v5_results"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("v4core", ROOT / "scripts" / "backtest_independent_ai_buy_selector_v4.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

TEST_YEARS = range(2021, 2026)
PURGE = 120
CAL_DAYS = 252
MAX_TRAIN_ROWS = 320_000
RNG_SEED = 92651
TAIL_QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.995)
MIN_CAL_SELECTED = 200
TARGET_LIFT = 1.50


def add_joint_quality(ds: pd.DataFrame) -> pd.DataFrame:
    x = ds.copy()
    rank_cols = ["rank_fwd20", "rank_fwd60", "rank_fwd120", "rank_mae60", "rank_mae120"]
    ranks = x[rank_cols]
    positive_endpoints = (x[["fwd20", "fwd60", "fwd120"]] > 0).all(axis=1)
    available = x[["fwd20", "fwd60", "fwd120", "mae60", "mae120"]].notna().all(axis=1)
    y = ranks.min(axis=1).astype(float)
    y.loc[~positive_endpoints] = 0.0
    y.loc[~available] = np.nan
    x["joint_quality"] = y.astype(np.float32)
    return x


def sample_train(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(train) <= MAX_TRAIN_ROWS:
        return train
    return train.sample(MAX_TRAIN_ROWS, random_state=seed).sort_values(["date", "code"])


def fit_model(train: pd.DataFrame, feats: list[str], seed: int) -> HistGradientBoostingRegressor:
    tr = sample_train(train, seed)
    X = v4.model_frame(tr, feats)
    y = tr["joint_quality"].astype(np.float32).to_numpy()
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=3.0,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def wilson_lower(k: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - rad) / den


def learn_abstention(train: pd.DataFrame, feats: list[str], seed: int) -> dict:
    dates = np.array(sorted(train["date"].unique()), dtype=np.int64)
    if len(dates) <= CAL_DAYS + 120:
        return {"eligible": False, "threshold": float("inf"), "quantile": None, "reason": "insufficient_calibration_history"}
    cal_dates = dates[-CAL_DAYS:]
    fit = train[train["date"] < cal_dates[0]].copy()
    cal = train[train["date"].isin(cal_dates)].copy()
    if len(fit) < 20_000 or len(cal) < 2_000:
        return {"eligible": False, "threshold": float("inf"), "quantile": None, "reason": "insufficient_calibration_rows"}

    model = fit_model(fit, feats, seed + 1000)
    score = model.predict(v4.model_frame(cal, feats))
    base = float(cal["worth_buy_label"].mean())
    rows = []
    chosen = None
    for q in TAIL_QUANTILES:
        threshold = float(np.quantile(score, q))
        mask = score >= threshold
        n = int(mask.sum())
        if n == 0:
            continue
        y = cal.loc[mask, "worth_buy_label"].astype(int).to_numpy()
        k = int(y.sum())
        precision = k / n
        lift = precision / base if base > 0 else float("nan")
        lower = wilson_lower(k, n)
        row = {
            "quantile": q,
            "threshold": threshold,
            "selected": n,
            "successes": k,
            "precision": precision,
            "base_rate": base,
            "lift": lift,
            "wilson_lower": lower,
            "eligible": bool(n >= MIN_CAL_SELECTED and lift >= TARGET_LIFT and lower > base),
        }
        rows.append(row)
        if row["eligible"] and chosen is None:
            chosen = row

    if chosen is None:
        return {
            "eligible": False,
            "threshold": float("inf"),
            "quantile": None,
            "reason": "no_calibration_tail_proved_enrichment",
            "base_rate": base,
            "calibration_rows": int(len(cal)),
            "tails": rows,
        }
    return {
        "eligible": True,
        "threshold": float(chosen["threshold"]),
        "quantile": float(chosen["quantile"]),
        "reason": "calibration_tail_proved_enrichment",
        "base_rate": base,
        "calibration_rows": int(len(cal)),
        "calibration_precision": float(chosen["precision"]),
        "calibration_lift": float(chosen["lift"]),
        "calibration_wilson_lower": float(chosen["wilson_lower"]),
        "tails": rows,
    }


def eval_year(test: pd.DataFrame, selected: pd.DataFrame, year: int, cal: dict) -> dict:
    row = {
        "test_year": year,
        "eligible_calibration": bool(cal["eligible"]),
        "threshold": cal["threshold"] if np.isfinite(cal["threshold"]) else np.nan,
        "chosen_quantile": cal.get("quantile"),
        "calibration_base_rate": cal.get("base_rate", np.nan),
        "calibration_precision": cal.get("calibration_precision", np.nan),
        "calibration_lift": cal.get("calibration_lift", np.nan),
        "test_rows": int(len(test)),
        "selected_rows": int(len(selected)),
        "signal_days": int(selected["date"].nunique()) if len(selected) else 0,
        "total_test_days": int(test["date"].nunique()),
    }
    row["no_signal_days"] = row["total_test_days"] - row["signal_days"]
    lab_u = test[test["worth_buy_label"].notna()]
    lab_s = selected[selected["worth_buy_label"].notna()]
    row["universe_label_rate"] = float(lab_u["worth_buy_label"].mean()) if len(lab_u) else np.nan
    row["selected_label_precision"] = float(lab_s["worth_buy_label"].mean()) if len(lab_s) else np.nan
    row["precision_lift"] = (
        row["selected_label_precision"] / row["universe_label_rate"]
        if row["universe_label_rate"] > 0 and np.isfinite(row["selected_label_precision"])
        else np.nan
    )
    for h in (5, 20, 60, 120):
        u = test[test[f"fwd{h}"].notna()]
        med = u.groupby("date")[f"fwd{h}"].median()
        s = selected[selected[f"fwd{h}"].notna()].copy()
        if len(s):
            s[f"alpha{h}"] = s[f"fwd{h}"] - s["date"].map(med)
            row[f"mean_fwd{h}"] = float(s[f"fwd{h}"].mean())
            row[f"mean_alpha{h}"] = float(s[f"alpha{h}"].mean())
            row[f"positive_rate{h}"] = float((s[f"fwd{h}"] > 0).mean())
        else:
            row[f"mean_fwd{h}"] = np.nan
            row[f"mean_alpha{h}"] = np.nan
            row[f"positive_rate{h}"] = np.nan
    for h in (20, 60, 120):
        u = test[test[f"mae{h}"].notna()]
        s = selected[selected[f"mae{h}"].notna()]
        row[f"mean_mae{h}"] = float(s[f"mae{h}"].mean()) if len(s) else np.nan
        row[f"universe_mean_mae{h}"] = float(u[f"mae{h}"].mean()) if len(u) else np.nan
        row[f"mean_mfe{h}"] = float(s[f"mfe{h}"].mean()) if len(s) else np.nan
        row[f"universe_mean_mfe{h}"] = float(u[f"mfe{h}"].mean()) if len(u) else np.nan
    return row


def combined_metrics(ds: pd.DataFrame, selected: pd.DataFrame) -> dict:
    overall = {
        "architecture": "raw temporal tape -> bottleneck joint-quality regressor -> precision-controlled abstention",
        "fixed_top_n": False,
        "can_abstain": True,
        "selected_rows": int(len(selected)),
        "unique_selected_codes": int(selected["code"].nunique()) if len(selected) else 0,
        "signal_days": int(selected["date"].nunique()) if len(selected) else 0,
    }
    oos = ds[(ds["date"] >= 20210101) & (ds["date"] <= 20251231) & ds["universe_ok"]].copy()
    for h in (5, 20, 60, 120):
        u = oos[oos[f"fwd{h}"].notna()]
        med = u.groupby("date")[f"fwd{h}"].median()
        s = selected[selected[f"fwd{h}"].notna()].copy()
        if len(s):
            s[f"alpha{h}"] = s[f"fwd{h}"] - s["date"].map(med)
            overall[f"mean_fwd{h}"] = float(s[f"fwd{h}"].mean())
            overall[f"mean_alpha{h}"] = float(s[f"alpha{h}"].mean())
        else:
            overall[f"mean_fwd{h}"] = np.nan
            overall[f"mean_alpha{h}"] = np.nan
    for h in (20, 60, 120):
        u = oos[oos[f"mae{h}"].notna()]
        s = selected[selected[f"mae{h}"].notna()]
        overall[f"mean_mae{h}"] = float(s[f"mae{h}"].mean()) if len(s) else np.nan
        overall[f"universe_mean_mae{h}"] = float(u[f"mae{h}"].mean()) if len(u) else np.nan
        overall[f"mean_mfe{h}"] = float(s[f"mfe{h}"].mean()) if len(s) else np.nan
        overall[f"universe_mean_mfe{h}"] = float(u[f"mfe{h}"].mean()) if len(u) else np.nan
    return overall


def main() -> None:
    print("[V5] load full-market data", flush=True)
    px = v4.load_px()
    inst = v4.load_inst()
    print(f"[V5] px rows={len(px):,} dates={px.date.min()}..{px.date.max()} inst={len(inst):,}", flush=True)
    ds = v4.add_channels(px, inst)
    ds, feats = v4.add_raw_tape_features(ds)
    ds = v4.add_labels(ds)
    ds = add_joint_quality(ds)
    ds["feature_ok"] = ds[feats].notna().mean(axis=1) >= 0.85
    ds["universe_ok"] = ds["valid_equity"] & (ds["hist_count"] >= 120) & ds["feature_ok"]
    ds = ds.sort_values(["date", "code"]).reset_index(drop=True)
    print(f"[V5] features={len(feats)} universe rows={int(ds.universe_ok.sum()):,}", flush=True)

    dates = np.array(sorted(ds["date"].unique()), dtype=np.int64)
    d2i = {int(d): i for i, d in enumerate(dates)}
    rows, picks, cal_evidence = [], [], {}

    for year in TEST_YEARS:
        test = ds[(ds["date"] >= year * 10000 + 101) & (ds["date"] <= year * 10000 + 1231) & ds["universe_ok"]].copy()
        first = int(test["date"].min())
        cutoff = int(dates[d2i[first] - PURGE])
        train = ds[(ds["date"] < cutoff) & ds["universe_ok"] & ds["joint_quality"].notna()].copy()
        if len(train) < 20_000:
            raise RuntimeError(f"insufficient train {year}: {len(train)}")
        cal = learn_abstention(train, feats, RNG_SEED + year)
        cal_evidence[str(year)] = cal
        if cal["eligible"]:
            model = fit_model(train, feats, RNG_SEED + year)
            test["ai_score"] = model.predict(v4.model_frame(test, feats))
            selected = test[test["ai_score"] >= cal["threshold"]].copy()
        else:
            test["ai_score"] = np.nan
            selected = test.iloc[0:0].copy()
        selected["test_year"] = year
        selected["train_cutoff"] = cutoff
        selected["learned_threshold"] = cal["threshold"] if np.isfinite(cal["threshold"]) else np.nan
        picks.append(selected)
        row = eval_year(test, selected, year, cal)
        row["train_cutoff"] = cutoff
        row["train_rows_full"] = int(len(train))
        rows.append(row)
        print(
            f"[V5] OOS {year} cal={cal['eligible']} q={cal.get('quantile')} "
            f"selected={len(selected):,} signal_days={row['signal_days']}/{row['total_test_days']} "
            f"lift={row['precision_lift'] if np.isfinite(row['precision_lift']) else float('nan'):.2f}",
            flush=True,
        )

    diag = pd.DataFrame(rows)
    selected = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    diag.to_csv(OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_YEARLY.csv", index=False)
    if len(selected):
        keep = ["date", "code", "name", "close", "ai_score", "learned_threshold", "test_year", "train_cutoff",
                "joint_quality", "worth_buy_label", "fwd5", "fwd20", "fwd60", "fwd120",
                "mfe20", "mae20", "mfe60", "mae60", "mfe120", "mae120"]
        selected[keep].to_csv(OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_OOS_PICKS.csv", index=False)

    overall = combined_metrics(ds, selected)
    emitting = diag[diag["selected_rows"] > 0].copy()
    overall["emitting_years"] = int(len(emitting))
    overall["no_signal_days_sum_by_year"] = int(diag["no_signal_days"].sum())
    overall["median_precision_lift_emitting_years"] = float(emitting["precision_lift"].median()) if len(emitting) else np.nan
    for h in (5, 20, 60, 120):
        overall[f"positive_emitting_years_{h}"] = int((emitting[f"mean_fwd{h}"] > 0).sum()) if len(emitting) else 0

    gate = {
        "absolute_positive_20_60_120": bool(len(selected) and all(overall[f"mean_fwd{h}"] > 0 for h in (20, 60, 120))),
        "positive_alpha_all_horizons": bool(len(selected) and all(overall[f"mean_alpha{h}"] > 0 for h in (5, 20, 60, 120))),
        "emits_candidates_in_at_least_3_years": bool(overall["emitting_years"] >= 3),
        "four_positive_emitting_years_20_60_120": bool(
            overall["emitting_years"] >= 4 and all(overall[f"positive_emitting_years_{h}"] >= 4 for h in (20, 60, 120))
        ),
        "median_precision_lift_at_least_1p5": bool(
            np.isfinite(overall["median_precision_lift_emitting_years"]) and overall["median_precision_lift_emitting_years"] >= TARGET_LIFT
        ),
        "mae_better_than_universe_60_120": bool(
            len(selected)
            and overall["mean_mae60"] >= overall["universe_mean_mae60"]
            and overall["mean_mae120"] >= overall["universe_mean_mae120"]
        ),
        "selector_demonstrates_abstention": bool(overall["no_signal_days_sum_by_year"] > 0),
        "selector_emits_some_candidates": bool(len(selected) > 0),
    }
    gate["promising_independent_ai_selector_v5"] = all(gate.values())

    # Live/current inference.
    latest_date = int(ds["date"].max())
    latest_i = d2i[latest_date]
    live_cutoff = int(dates[latest_i - PURGE])
    live_train = ds[(ds["date"] <= live_cutoff) & ds["universe_ok"] & ds["joint_quality"].notna()].copy()
    live = ds[(ds["date"] == latest_date) & ds["universe_ok"]].copy()
    live_cal = learn_abstention(live_train, feats, RNG_SEED + 999)
    if live_cal["eligible"]:
        live_model = fit_model(live_train, feats, RNG_SEED + 999)
        live["ai_score"] = live_model.predict(v4.model_frame(live, feats))
        live["ai_buy"] = live["ai_score"] >= live_cal["threshold"]
    else:
        live["ai_score"] = np.nan
        live["ai_buy"] = False
    live["learned_threshold"] = live_cal["threshold"] if np.isfinite(live_cal["threshold"]) else np.nan
    live = live.sort_values("ai_score", ascending=False, na_position="last")
    live_out = live[["date", "code", "name", "close", "ai_score", "learned_threshold", "ai_buy"]].copy()
    live_out.to_csv(OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_LIVE.csv", index=False)
    buys = live_out[live_out["ai_buy"]]
    live_summary = {
        "as_of": latest_date,
        "train_label_cutoff": live_cutoff,
        "universe_count": int(len(live)),
        "calibration_eligible": bool(live_cal["eligible"]),
        "calibration_reason": live_cal["reason"],
        "chosen_quantile": live_cal.get("quantile"),
        "learned_threshold": live_cal["threshold"] if np.isfinite(live_cal["threshold"]) else None,
        "calibration_base_rate": live_cal.get("base_rate"),
        "calibration_precision": live_cal.get("calibration_precision"),
        "calibration_lift": live_cal.get("calibration_lift"),
        "buy_count": int(len(buys)),
        "decision": "BUY_CANDIDATES" if len(buys) else "NONE",
        "candidates": buys.head(100).to_dict(orient="records"),
        "top_scores_for_audit": live_out.head(20).to_dict(orient="records"),
    }

    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_CALIBRATION.json").write_text(json.dumps(cal_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_OVERALL.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_GATE.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V5_LIVE.json").write_text(json.dumps(live_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[V5 YEARLY]\n" + diag.to_string(index=False), flush=True)
    print("\n[V5 OVERALL]\n" + json.dumps(overall, indent=2), flush=True)
    print("\n[V5 GATE]\n" + json.dumps(gate, indent=2), flush=True)
    print("\n[V5 LIVE]\n" + json.dumps(live_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
