#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ai_market_reasoning_v6_cpu_quantile_results"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("v6base", ROOT / "scripts" / "backtest_ai_market_reasoning_v6.py")
v6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v6)

# PRE-OOS locked architecture constants. Do not tune from OOS.
MAX_HORIZON = 120
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
AUDIT_HORIZONS = (5, 10, 20, 40, 60, 120)  # reporting checkpoints only, not separate models
PURGE = 120
MAX_LONG_TRAIN_ROWS = 480_000
RNG_SEED = 926616
MODEL_PARAMS = dict(
    learning_rate=0.05,
    max_iter=160,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=4.0,
)


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Reuse causal/PIT context assembly; no R10/R7/R6 engine is imported."""
    px = v6.v4.load_px()
    px = px[px["date"] >= v6.START_DATE].copy()
    inst = v6.v4.load_inst()
    ds = v6.v4.add_channels(px, inst)
    ds, f_tape = v6.add_tape_features(ds)
    ds, f_struct = v6.add_structure_and_seasonality(ds)
    ds, f_macro = v6.add_macro_context(ds)
    ds, f_sector, _ = v6.add_sector_context(ds)
    ds, f_fund = v6.add_fundamental_context(ds)

    # Frozen observable transforms/context, not buy rules or hand-set weights.
    feats = f_tape + f_struct + f_macro + f_sector + f_fund
    ds["core_feature_ok"] = ds[f_tape].notna().mean(axis=1) >= 0.80
    ds["universe_ok"] = ds["valid_equity"] & (ds["hist_count"] >= MAX_HORIZON) & ds["core_feature_ok"]
    ds = ds.sort_values(["code", "date"]).reset_index(drop=True)
    return ds, feats


def model_frame(df: pd.DataFrame, feats: list[str], horizon: np.ndarray | float) -> np.ndarray:
    x = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=False)
    h = np.asarray(horizon, dtype=np.float32)
    if h.ndim == 0:
        h = np.full(len(df), float(h), dtype=np.float32)
    h1 = (h / MAX_HORIZON).reshape(-1, 1)
    h2 = np.sqrt(h / MAX_HORIZON).reshape(-1, 1)
    return np.concatenate([x, h1, h2], axis=1)


def make_long_training(train: pd.DataFrame, feats: list[str], seed: int):
    """Uniformly represent every horizon 1..120 without materializing 120 full label columns."""
    rng = np.random.default_rng(seed)
    per_h = max(500, MAX_LONG_TRAIN_ROWS // MAX_HORIZON)
    chunks_x, chunks_y = [], []
    g = train.groupby("code", group_keys=False)["close"]
    for h in range(1, MAX_HORIZON + 1):
        y = g.shift(-h) / train["close"] - 1.0
        valid = train["universe_ok"].to_numpy() & np.isfinite(y.to_numpy(dtype=float))
        idx = np.flatnonzero(valid)
        if len(idx) > per_h:
            idx = rng.choice(idx, size=per_h, replace=False)
        if len(idx) == 0:
            continue
        part = train.iloc[idx]
        chunks_x.append(model_frame(part, feats, np.full(len(part), h, dtype=np.float32)))
        chunks_y.append(y.iloc[idx].to_numpy(dtype=np.float32))
    if not chunks_x:
        raise RuntimeError("no usable quantile path training rows")
    return np.concatenate(chunks_x, axis=0), np.concatenate(chunks_y, axis=0)


def fit_quantile_models(train: pd.DataFrame, feats: list[str], seed: int):
    X, y = make_long_training(train, feats, seed)
    models = {}
    for i, q in enumerate(QUANTILES):
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            random_state=seed + i,
            **MODEL_PARAMS,
        )
        model.fit(X, y)
        models[q] = model
        print(f"[V6 CPU-Q FIT] q={q:.2f} rows={len(y):,}", flush=True)
    return models, len(y)


def predict_quantiles(models, df: pd.DataFrame, feats: list[str], horizon: int) -> pd.DataFrame:
    X = model_frame(df, feats, float(horizon))
    arr = np.column_stack([models[q].predict(X) for q in QUANTILES]).astype(np.float32)
    # Deterministic non-crossing projection; no OOS-dependent tuning.
    arr.sort(axis=1)
    return pd.DataFrame(arr, index=df.index, columns=[f"q{int(q*100):02d}" for q in QUANTILES])


def audit_year(models, test: pd.DataFrame, feats: list[str], year: int) -> list[dict]:
    rows = []
    g = test.groupby("code", group_keys=False)["close"]
    for h in AUDIT_HORIZONS:
        actual = g.shift(-h) / test["close"] - 1.0
        pred = predict_quantiles(models, test, feats, h)
        valid = np.isfinite(actual.to_numpy(dtype=float))
        if not valid.any():
            continue
        y = actual.to_numpy(dtype=float)[valid]
        p = pred.to_numpy(dtype=float)[valid]
        row = {"test_year": year, "horizon": h, "n": int(len(y))}
        for j, q in enumerate(QUANTILES):
            row[f"pinball_q{int(q*100):02d}"] = float(mean_pinball_loss(y, p[:, j], alpha=q))
            row[f"empirical_cdf_q{int(q*100):02d}"] = float(np.mean(y <= p[:, j]))
        row["coverage_q10_q90"] = float(np.mean((y >= p[:, 0]) & (y <= p[:, -1])))
        row["median_abs_error"] = float(np.median(np.abs(y - p[:, 2])))
        rows.append(row)
    return rows


def assert_2025_unlock() -> None:
    if os.getenv("V6_OPEN_2025") != "1":
        return
    lock = ROOT / "research" / "V6_CPU_QUANTILE_LOCK.json"
    gate = OUT / "V6_CPU_QUANTILE_STAGE_A_GATE.json"
    if not lock.exists() or not gate.exists():
        raise RuntimeError("2025 SEALED: lock manifest and Stage-A gate are both required")
    lock_obj = json.loads(lock.read_text(encoding="utf-8"))
    gate_obj = json.loads(gate.read_text(encoding="utf-8"))
    if lock_obj.get("status") != "LOCKED" or gate_obj.get("stage_a_passed") is not True:
        raise RuntimeError("2025 SEALED: Stage-A gate has not authorized opening 2025")


def live_diagnostic(latest: int, cutoff: int, nfit: int, features: int, universe_count: int) -> dict:
    # No fixed Top-N sample and no uncalibrated ranking is emitted.
    return {
        "as_of": latest,
        "decision": "NONE_UNTIL_STAGE3_CALIBRATION",
        "model_family": "HistGradientBoostingRegressor quantile; horizon-conditioned 1..120",
        "quantiles": QUANTILES,
        "max_horizon": MAX_HORIZON,
        "train_label_cutoff": cutoff,
        "fit_long_rows": nfit,
        "feature_count": features,
        "universe_count": universe_count,
        "note": "Full 1..120 live paths are withheld from action/selection until past-only probability calibration is implemented and passed."
    }


def main():
    assert_2025_unlock()
    ds, feats = build_dataset()
    dates = np.array(sorted(ds["date"].unique()), dtype=np.int64)
    d2i = {int(d): i for i, d in enumerate(dates)}

    test_years = [2021, 2022, 2023, 2024]
    if os.getenv("V6_OPEN_2025") == "1":
        test_years.append(2025)

    audits = []
    for year in test_years:
        test = ds[(ds["date"] >= year*10000+101) & (ds["date"] <= year*10000+1231) & ds["universe_ok"]].copy()
        if test.empty:
            raise RuntimeError(f"empty test year {year}")
        first = int(test["date"].min())
        cutoff = int(dates[d2i[first] - PURGE])
        train = ds[(ds["date"] < cutoff) & ds["universe_ok"]].copy()
        models, nfit = fit_quantile_models(train, feats, RNG_SEED + year)
        yr = audit_year(models, test, feats, year)
        for row in yr:
            row.update({"train_cutoff": cutoff, "fit_long_rows": nfit, "feature_count": len(feats)})
        audits.extend(yr)
        print("[V6 CPU-Q OOS AUDIT]", year, json.dumps(yr, ensure_ascii=False), flush=True)

    pd.DataFrame(audits).to_csv(OUT / "V6_CPU_QUANTILE_STAGE_A_AUDIT.csv", index=False)

    latest = int(ds["date"].max())
    cutoff = int(dates[d2i[latest] - PURGE])
    hist = ds[(ds["date"] <= cutoff) & ds["universe_ok"]].copy()
    _, nfit = fit_quantile_models(hist, feats, RNG_SEED + 999)
    live = ds[(ds["date"] == latest) & ds["universe_ok"]].copy()
    live_json = live_diagnostic(latest, cutoff, nfit, len(feats), len(live))
    (OUT / "V6_CPU_QUANTILE_LIVE_DIAGNOSTIC.json").write_text(json.dumps(live_json, ensure_ascii=False, indent=2), encoding="utf-8")

    spec_out = {
        "architecture": "CPU HistGradientBoostingRegressor quantile regression, horizon-conditioned 1..120",
        "quantiles": QUANTILES,
        "audit_horizons": AUDIT_HORIZONS,
        "max_horizon": MAX_HORIZON,
        "model_params": MODEL_PARAMS,
        "stage3_calibration_required_before_user_probability_or_buy": True,
        "deep_representation_learning": "deferred_to_successor_preregistered_version",
    }
    (OUT / "V6_CPU_QUANTILE_MODEL_SPEC.json").write_text(json.dumps(spec_out, indent=2), encoding="utf-8")
    print(json.dumps(spec_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
