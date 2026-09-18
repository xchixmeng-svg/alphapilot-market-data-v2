#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "history" / "v6-layered" / "0F_FINAL_ASSEMBLY"
PANEL_PATH = FROZEN / "decision_ticker_panel.parquet"
FEATURE_REGISTRY_PATH = FROZEN / "feature_registry.json"
FINAL_MANIFEST_PATH = ROOT / "data" / "history" / "v6-layered" / "manifests" / "0F_FINAL_ASSEMBLY.json"
SOURCE_FREEZE_PATH = ROOT / "research" / "V6_SOURCE_FREEZE.json"
LOCK_PATH = ROOT / "research" / "V6_PREREG_LOCK.json"
OUT = ROOT / "ai_market_reasoning_v6_cpu_quantile_results"
OUT.mkdir(exist_ok=True)

MAX_HORIZON = 120
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
AUDIT_HORIZONS = (5, 10, 20, 40, 60, 120)
PURGE = 120
MAX_LONG_TRAIN_ROWS = 480_000
RNG_SEED = 926616
LAGS = (0, 1, 2, 5, 10, 20, 60, 119)
STOCK_CHANNELS = ("ret1", "gap1", "range1", "body1", "amount_logchg")
MODEL_PARAMS = dict(
    learning_rate=0.05,
    max_iter=160,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=4.0,
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_MEAN_BLOCK = 20
BONFERRONI_ALPHA = 0.05 / len(AUDIT_HORIZONS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_lock_meta() -> dict:
    if not LOCK_PATH.exists():
        raise RuntimeError("formal OOS prohibited: V6_PREREG_LOCK.json missing")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("lock_status") != "LOCKED":
        raise RuntimeError("formal OOS prohibited: lock_status is not LOCKED")
    return lock


def load_frozen_panel() -> tuple[pd.DataFrame, list[str], dict]:
    lock = load_lock_meta()
    freeze = json.loads(SOURCE_FREEZE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    freg = json.loads(FEATURE_REGISTRY_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_PASS" or manifest.get("pit_rules", {}).get("formal_oos_eligible") is not True:
        raise RuntimeError("0F is not FROZEN_PASS/formal_oos_eligible")
    if manifest.get("pit_rules", {}).get("2025_model_evaluation_opened") is not False:
        raise RuntimeError("2025 seal is not intact")
    expected = freeze["stage0_final"]["decision_ticker_panel_sha256"]
    got = sha256(PANEL_PATH)
    if got != expected:
        raise RuntimeError(f"0F panel hash mismatch: {got} != {expected}")
    manifest_hash = manifest.get("file_sha256", {}).get("decision_ticker_panel.parquet")
    if manifest_hash != got:
        raise RuntimeError(f"0F manifest panel hash mismatch: {manifest_hash} != {got}")

    base_feats = list(freg.get("features") or [])
    if len(base_feats) != int(freg.get("feature_count", -1)):
        raise RuntimeError("feature registry count mismatch")
    x = pd.read_parquet(PANEL_PATH)
    if int(len(x)) != int(manifest.get("row_count", -1)):
        raise RuntimeError("0F row count mismatch")
    x["code"] = x["code"].astype(str).str.zfill(4)
    x["date"] = pd.to_numeric(x["date"], errors="raise").astype(np.int64)
    x = x.sort_values(["code", "date"]).reset_index(drop=True)
    return x, base_feats, lock


def add_frozen_observable_transforms(
    x: pd.DataFrame, base_feats: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    z = x.copy()
    for c in base_feats:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)

    g = z.groupby("code", group_keys=False)
    prev_close = g["close"].shift(1)
    prev_amount = (z["close"] * z["volume"]).groupby(z["code"]).shift(1)
    z["ret1"] = z["close"] / prev_close - 1.0
    z["gap1"] = z["open"] / prev_close - 1.0
    z["range1"] = (z["high"] - z["low"]) / prev_close.replace(0, np.nan)
    z["body1"] = z["close"] / z["open"].replace(0, np.nan) - 1.0
    amt = (z["close"] * z["volume"]).clip(lower=0)
    z["amount_logchg"] = np.log1p(amt) - np.log1p(prev_amount.clip(lower=0))
    for c in STOCK_CHANNELS:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)

    feats = list(base_feats)
    g = z.groupby("code", group_keys=False)
    for lag in LAGS:
        for c in STOCK_CHANNELS:
            name = f"{c}_lag{lag}"
            z[name] = g[c].shift(lag).astype(np.float32)
            feats.append(name)

    z["ret20_obs"] = g["close"].pct_change(20)
    z["ret60_obs"] = g["close"].pct_change(60)
    z["vol20_obs"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    z["vol60_obs"] = g["ret1"].transform(lambda s: s.rolling(60, min_periods=40).std())
    hi20 = g["high"].transform(lambda s: s.rolling(20, min_periods=15).max())
    lo20 = g["low"].transform(lambda s: s.rolling(20, min_periods=15).min())
    hi60 = g["high"].transform(lambda s: s.rolling(60, min_periods=40).max())
    lo60 = g["low"].transform(lambda s: s.rolling(60, min_periods=40).min())
    z["range_width20"] = hi20 / lo20.replace(0, np.nan) - 1
    z["range_width60"] = hi60 / lo60.replace(0, np.nan) - 1
    z["close_pos20"] = (z["close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    z["close_pos60"] = (z["close"] - lo60) / (hi60 - lo60).replace(0, np.nan)
    z["season_ret20_1y"] = g["close"].shift(232) / g["close"].shift(252) - 1
    z["season_ret60_1y"] = g["close"].shift(192) / g["close"].shift(252) - 1
    z["season_ret20_2y"] = g["close"].shift(484) / g["close"].shift(504) - 1
    structure = [
        "ret20_obs", "ret60_obs", "vol20_obs", "vol60_obs",
        "range_width20", "range_width60", "close_pos20", "close_pos60",
        "season_ret20_1y", "season_ret60_1y", "season_ret20_2y",
    ]
    for c in structure:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)
    feats.extend(structure)

    dt = pd.to_datetime(z["date"].astype(str), format="%Y%m%d", errors="coerce")
    doy = dt.dt.dayofyear.astype(float)
    month = dt.dt.month.astype(float)
    dow = dt.dt.dayofweek.astype(float)
    z["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    z["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    z["cal_month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    z["cal_month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    z["cal_dow_sin"] = np.sin(2 * np.pi * dow / 5.0).astype(np.float32)
    z["cal_dow_cos"] = np.cos(2 * np.pi * dow / 5.0).astype(np.float32)
    cal = ["cal_doy_sin", "cal_doy_cos", "cal_month_sin", "cal_month_cos", "cal_dow_sin", "cal_dow_cos"]
    feats.extend(cal)

    daily = z[["date", "macro_fed_funds", "macro_us2y", "macro_us10y"]].drop_duplicates("date").sort_values("date")
    macro_delta = []
    for c in ("macro_fed_funds", "macro_us2y", "macro_us10y"):
        for h in (5, 20, 60):
            name = f"{c}_d{h}"
            daily[name] = (daily[c] - daily[c].shift(h)).astype(np.float32)
            macro_delta.append(name)
    z = z.merge(daily[["date"] + macro_delta], on="date", how="left", validate="many_to_one")
    feats.extend(macro_delta)

    ind_cols = [c for c in base_feats if c.startswith("industry_ret1_")]
    if ind_cols:
        d = z[["date"] + ind_cols].drop_duplicates("date").sort_values("date").set_index("date")
        d["sector_adv"] = (d[ind_cols] > 0).mean(axis=1).astype(np.float32)
        d["sector_median1"] = d[ind_cols].median(axis=1).astype(np.float32)
        d["sector_dispersion1"] = d[ind_cols].std(axis=1).astype(np.float32)
        ret20 = (1.0 + d[ind_cols]).rolling(20, min_periods=15).apply(np.prod, raw=True) - 1.0
        d["sector_top3_20"] = ret20.apply(
            lambda r: r.nlargest(min(3, r.notna().sum())).mean() if r.notna().any() else np.nan,
            axis=1,
        ).astype(np.float32)
        d["sector_bottom3_20"] = ret20.apply(
            lambda r: r.nsmallest(min(3, r.notna().sum())).mean() if r.notna().any() else np.nan,
            axis=1,
        ).astype(np.float32)
        sf = ["sector_adv", "sector_median1", "sector_dispersion1", "sector_top3_20", "sector_bottom3_20"]
        z = z.merge(d[sf].reset_index(), on="date", how="left", validate="many_to_one")
        feats.extend(sf)

    z = z.sort_values(["code", "date"]).reset_index(drop=True)
    z["hist_count"] = z.groupby("code").cumcount() + 1
    z["universe_ok"] = (z["hist_count"] >= MAX_HORIZON) & (z["close"] > 0)
    feats = list(dict.fromkeys(feats))
    missing = [c for c in feats if c not in z.columns]
    if missing:
        raise RuntimeError(f"frozen transform feature missing: {missing[:20]}")
    return z, feats


def model_frame(df: pd.DataFrame, feats: list[str], horizon: np.ndarray | float) -> np.ndarray:
    x = df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=False)
    h = np.asarray(horizon, dtype=np.float32)
    if h.ndim == 0:
        h = np.full(len(df), float(h), dtype=np.float32)
    h1 = (h / MAX_HORIZON).reshape(-1, 1)
    h2 = np.sqrt(h / MAX_HORIZON).reshape(-1, 1)
    return np.concatenate([x, h1, h2], axis=1)


def fwd_return(df: pd.DataFrame, h: int) -> pd.Series:
    return df.groupby("code", group_keys=False)["close"].shift(-h) / df["close"] - 1.0


def make_long_training(train: pd.DataFrame, feats: list[str], seed: int):
    rng = np.random.default_rng(seed)
    per_h = max(500, MAX_LONG_TRAIN_ROWS // MAX_HORIZON)
    chunks_x, chunks_y = [], []
    for h in range(1, MAX_HORIZON + 1):
        y = fwd_return(train, h)
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


def fit_models(train: pd.DataFrame, feats: list[str], seed: int):
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
        print(f"[V6 STAGE-A FIT] q={q:.2f} rows={len(y):,}", flush=True)
    return models, len(y)


def predict_quantiles(models, df: pd.DataFrame, feats: list[str], h: int) -> np.ndarray:
    X = model_frame(df, feats, float(h))
    p = np.column_stack([models[q].predict(X) for q in QUANTILES]).astype(np.float32)
    p.sort(axis=1)
    return p


def pinball_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    losses = []
    for j, q in enumerate(QUANTILES):
        e = y - p[:, j]
        losses.append(np.maximum(q * e, (q - 1.0) * e))
    return np.column_stack(losses)


def naive_quantiles(train: pd.DataFrame, h: int) -> np.ndarray:
    y = fwd_return(train, h).to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 1000:
        raise RuntimeError(f"naive reference support too small for h={h}: {len(y)}")
    return np.quantile(y, QUANTILES).astype(np.float32)


def run_year(year: int) -> None:
    if year not in (2021, 2022, 2023, 2024):
        raise RuntimeError("Stage A authorizes only 2021-2024; 2025 remains sealed")
    panel, base_feats, lock = load_frozen_panel()
    ds, feats = add_frozen_observable_transforms(panel, base_feats)
    dates = np.array(sorted(ds["date"].unique()), dtype=np.int64)
    test = ds[(ds["date"] >= year * 10000 + 101) & (ds["date"] <= year * 10000 + 1231) & ds["universe_ok"]].copy()
    if test.empty:
        raise RuntimeError(f"empty test year {year}")
    first = int(test["date"].min())
    pos = int(np.searchsorted(dates, first))
    if pos < PURGE:
        raise RuntimeError(f"insufficient purge history for {year}")
    cutoff = int(dates[pos - PURGE])
    train = ds[(ds["date"] < cutoff) & ds["universe_ok"]].copy()
    models, nfit = fit_models(train, feats, RNG_SEED + year)

    summaries = []
    date_losses = []
    for h in AUDIT_HORIZONS:
        actual = fwd_return(test, h)
        valid = np.isfinite(actual.to_numpy(dtype=float))
        eval_df = test.loc[valid].copy()
        y = actual.to_numpy(dtype=float)[valid]
        p = predict_quantiles(models, eval_df, feats, h)
        nq = naive_quantiles(train, h)
        npred = np.tile(nq.reshape(1, -1), (len(y), 1))

        row = {
            "test_year": year,
            "horizon": h,
            "n": int(len(y)),
            "train_cutoff": cutoff,
            "fit_long_rows": nfit,
            "feature_count": len(feats),
        }
        for j, q in enumerate(QUANTILES):
            row[f"pinball_q{int(q*100):02d}"] = float(mean_pinball_loss(y, p[:, j], alpha=q))
            row[f"naive_pinball_q{int(q*100):02d}"] = float(mean_pinball_loss(y, npred[:, j], alpha=q))
            row[f"empirical_cdf_q{int(q*100):02d}"] = float(np.mean(y <= p[:, j]))
            row[f"naive_empirical_cdf_q{int(q*100):02d}"] = float(np.mean(y <= npred[:, j]))
        row["coverage_q10_q90"] = float(np.mean((y >= p[:, 0]) & (y <= p[:, -1])))
        row["naive_coverage_q10_q90"] = float(np.mean((y >= npred[:, 0]) & (y <= npred[:, -1])))
        row["median_abs_error"] = float(np.median(np.abs(y - p[:, 2])))
        row["naive_median_abs_error"] = float(np.median(np.abs(y - npred[:, 2])))
        m_loss = pinball_rows(y, p).mean(axis=1)
        n_loss = pinball_rows(y, npred).mean(axis=1)
        row["mean_pinball_all_q"] = float(m_loss.mean())
        row["naive_mean_pinball_all_q"] = float(n_loss.mean())
        row["pinball_delta_vs_naive"] = float(n_loss.mean() - m_loss.mean())
        summaries.append(row)

        dl = pd.DataFrame({
            "date": eval_df["date"].to_numpy(dtype=np.int64),
            "horizon": h,
            "loss_diff_naive_minus_model": n_loss - m_loss,
        })
        dl = dl.groupby(["date", "horizon"], as_index=False)["loss_diff_naive_minus_model"].mean()
        date_losses.append(dl)
        print("[V6 STAGE-A YEAR]", json.dumps(row, ensure_ascii=False), flush=True)

    s = pd.DataFrame(summaries)
    d = pd.concat(date_losses, ignore_index=True)
    s.to_csv(OUT / f"V6_STAGE_A_{year}_AUDIT.csv", index=False)
    d.to_csv(OUT / f"V6_STAGE_A_{year}_DATE_LOSS.csv", index=False)
    meta = {
        "year": year,
        "lock_id": lock.get("lock_id"),
        "source_freeze_commit": lock.get("source_freeze_commit"),
        "panel_sha256": sha256(PANEL_PATH),
        "feature_count": len(feats),
        "test_rows": int(len(test)),
        "train_cutoff": cutoff,
        "2025_opened": False,
    }
    (OUT / f"V6_STAGE_A_{year}_META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def stationary_bootstrap_mean(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 100:
        raise RuntimeError(f"date-level bootstrap support too small: {n}")
    p_restart = 1.0 / BOOTSTRAP_MEAN_BLOCK
    out = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for b in range(BOOTSTRAP_REPLICATES):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        for i in range(1, n):
            if rng.random() < p_restart:
                idx[i] = rng.integers(0, n)
            else:
                idx[i] = (idx[i - 1] + 1) % n
        out[b] = float(np.mean(x[idx]))
    return out


def aggregate(input_dir: Path) -> None:
    audits = []
    losses = []
    for y in (2021, 2022, 2023, 2024):
        ap = input_dir / f"V6_STAGE_A_{y}_AUDIT.csv"
        lp = input_dir / f"V6_STAGE_A_{y}_DATE_LOSS.csv"
        if not ap.exists() or not lp.exists():
            raise RuntimeError(f"missing Stage-A year artifact for {y}")
        audits.append(pd.read_csv(ap))
        losses.append(pd.read_csv(lp))
    a = pd.concat(audits, ignore_index=True)
    dl = pd.concat(losses, ignore_index=True)

    infer = []
    for h in AUDIT_HORIZONS:
        x = dl.loc[dl["horizon"] == h].sort_values("date")["loss_diff_naive_minus_model"].to_numpy(dtype=float)
        boots = stationary_bootstrap_mean(x, RNG_SEED + 50_000 + h)
        mean = float(np.mean(x))
        p_improve = float((1 + np.sum(boots <= 0.0)) / (BOOTSTRAP_REPLICATES + 1))
        p_degrade = float((1 + np.sum(boots >= 0.0)) / (BOOTSTRAP_REPLICATES + 1))
        infer.append({
            "horizon": h,
            "date_level_n": int(len(x)),
            "mean_pinball_improvement_naive_minus_model": mean,
            "bootstrap_ci95_low": float(np.quantile(boots, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boots, 0.975)),
            "one_sided_p_improve": p_improve,
            "one_sided_p_degrade": p_degrade,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "significant_improvement": bool(mean > 0 and p_improve <= BONFERRONI_ALPHA),
            "significant_degradation": bool(mean < 0 and p_degrade <= BONFERRONI_ALPHA),
        })
    inf = pd.DataFrame(infer)

    weighted = []
    for h in AUDIT_HORIZONS:
        q = a[a["horizon"] == h].copy()
        w = q["n"].to_numpy(dtype=float)
        def wm(col):
            return float(np.average(q[col].to_numpy(dtype=float), weights=w))
        weighted.append({
            "horizon": h,
            "n": int(w.sum()),
            "model_pinball": wm("mean_pinball_all_q"),
            "naive_pinball": wm("naive_mean_pinball_all_q"),
            "model_coverage80": wm("coverage_q10_q90"),
            "naive_coverage80": wm("naive_coverage_q10_q90"),
            "model_median_abs_error": wm("median_abs_error"),
            "naive_median_abs_error": wm("naive_median_abs_error"),
        })
    wdf = pd.DataFrame(weighted).merge(inf, on="horizon", how="left")
    wdf["mae_improved"] = wdf["model_median_abs_error"] < wdf["naive_median_abs_error"]
    wdf["coverage_not_materially_worse"] = (
        (wdf["model_coverage80"] - 0.80).abs()
        <= (wdf["naive_coverage80"] - 0.80).abs() + 0.03
    )

    sig_improve = int(wdf["significant_improvement"].sum())
    sig_degrade = int(wdf["significant_degradation"].sum())
    mae_improve = int(wdf["mae_improved"].sum())
    coverage_ok = int(wdf["coverage_not_materially_worse"].sum())
    stage_a_passed = bool(
        sig_improve >= 4
        and sig_degrade == 0
        and mae_improve >= 4
        and coverage_ok >= 5
    )

    gate = {
        "stage": "A",
        "years": [2021, 2022, 2023, 2024],
        "stage_a_passed": stage_a_passed,
        "success_gate_frozen_before_oos": {
            "significant_pinball_improvement_horizons_min": 4,
            "significant_degradation_horizons_max": 0,
            "mae_improved_horizons_min": 4,
            "coverage_not_materially_worse_horizons_min": 5,
            "coverage_tolerance_absolute": 0.03,
            "confirmatory_bonferroni_alpha": BONFERRONI_ALPHA,
        },
        "observed": {
            "significant_pinball_improvement_horizons": sig_improve,
            "significant_degradation_horizons": sig_degrade,
            "mae_improved_horizons": mae_improve,
            "coverage_not_materially_worse_horizons": coverage_ok,
        },
        "2025_opened": False,
        "next_action": "AUTHORIZE_STAGE_B_2025" if stage_a_passed else "STOP_V6_LOCKED_VERSION_NO_RETUNE",
    }
    a.to_csv(OUT / "V6_CPU_QUANTILE_STAGE_A_AUDIT.csv", index=False)
    wdf.to_csv(OUT / "V6_CPU_QUANTILE_STAGE_A_HORIZON_GATE.csv", index=False)
    (OUT / "V6_CPU_QUANTILE_STAGE_A_GATE.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[V6 STAGE-A FINAL GATE]", json.dumps(gate, ensure_ascii=False), flush=True)
    print(wdf.to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--aggregate-dir")
    ns = ap.parse_args()
    if bool(ns.year) == bool(ns.aggregate_dir):
        raise SystemExit("specify exactly one of --year or --aggregate-dir")
    if ns.year:
        run_year(ns.year)
    else:
        aggregate(Path(ns.aggregate_dir))


if __name__ == "__main__":
    main()
