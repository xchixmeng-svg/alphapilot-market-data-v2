#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / "data" / "history" / "v6-context"
OUT = ROOT / "ai_market_reasoning_v6_results"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("v4core", ROOT / "scripts" / "backtest_independent_ai_buy_selector_v4.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

START_DATE = 20180101
TEST_YEARS = range(2021, 2026)
PURGE = 120
CAL_DAYS = 252
MAX_TRAIN_ROWS = 220_000
RNG_SEED = 926615
LAGS = (0, 1, 2, 5, 10, 20, 60, 119)
TARGET_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.30)
FORMAL_PROB = 0.90
MIN_CELL_N = 300
MFE_BINS = 20
MAE_BINS = 10
MODEL_TARGETS = ("mfe20h", "mfe60h", "mfe120h", "mae20l", "mae60l", "mae120l", "fwd60")


def ymd(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y%m%d").astype(np.int64)
    return pd.to_numeric(s.astype(str).str.replace("-", "", regex=False).str.replace("/", "", regex=False), errors="coerce").astype("Int64")


def model_frame(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=False)


def add_tape_features(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    z = x.sort_values(["code", "date"]).copy()
    feats: list[str] = []
    g = z.groupby("code", group_keys=False)
    for lag in LAGS:
        for c in v4.STOCK_CHANNELS:
            name = f"{c}_lag{lag}"
            z[name] = g[c].shift(lag).astype(np.float32)
            feats.append(name)
    daily = z[["date", *v4.MARKET_CHANNELS]].drop_duplicates("date").sort_values("date").copy()
    for lag in LAGS:
        for c in v4.MARKET_CHANNELS:
            name = f"{c}_lag{lag}"
            daily[name] = daily[c].shift(lag).astype(np.float32)
            feats.append(name)
    drop = [c for c in v4.MARKET_CHANNELS if c in z.columns]
    z = z.drop(columns=drop)
    z = z.merge(daily[["date"] + [f"{c}_lag{lag}" for lag in LAGS for c in v4.MARKET_CHANNELS]], on="date", how="left")
    return z, feats


def add_structure_and_seasonality(z: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    z = z.sort_values(["code", "date"]).copy()
    g = z.groupby("code", group_keys=False)
    feats = []
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
    for c in ("ret20_obs", "ret60_obs", "vol20_obs", "vol60_obs", "range_width20", "range_width60", "close_pos20", "close_pos60", "season_ret20_1y", "season_ret60_1y", "season_ret20_2y"):
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)
        feats.append(c)
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
    feats += ["cal_doy_sin", "cal_doy_cos", "cal_month_sin", "cal_month_cos", "cal_dow_sin", "cal_dow_cos"]
    return z, feats


def add_macro_context(z: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    p = CTX / "macro_daily.csv.gz"
    m = pd.read_csv(p)
    m["date"] = ymd(pd.to_datetime(m["date"])).astype(np.int64)
    m = m.sort_values("date")
    feats = []
    vals = [c for c in m.columns if c != "date"]
    for c in vals:
        m[c] = pd.to_numeric(m[c], errors="coerce")
        raw = f"macro_{c}"
        m[raw] = m[c].astype(np.float32)
        feats.append(raw)
        for h in (5, 20, 60):
            d = f"macro_{c}_d{h}"
            m[d] = (m[c] - m[c].shift(h)).astype(np.float32)
            feats.append(d)
    z = z.merge(m[["date"] + feats], on="date", how="left")
    return z, feats


def _sector_tables():
    p = CTX / "twse_industry_index_daily.csv.gz"
    d = pd.read_csv(p)
    d["date"] = ymd(pd.to_datetime(d["date"])).astype(np.int64)
    d["close_index"] = pd.to_numeric(d["close_index"], errors="coerce")
    piv = d.pivot_table(index="date", columns="index_name", values="close_index", aggfunc="last").sort_index()
    coverage = piv.notna().mean()
    cols = coverage[coverage >= 0.60].index.tolist()
    piv = piv[cols]
    ret1 = piv.pct_change(fill_method=None)
    ret5 = piv.pct_change(5, fill_method=None)
    ret20 = piv.pct_change(20, fill_method=None)
    return piv, ret1, ret5, ret20


def add_sector_context(z: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    piv, sret1, sret5, sret20 = _sector_tables()
    common = np.array(sorted(set(z["date"].unique()).intersection(piv.index)), dtype=np.int64)
    sret1 = sret1.reindex(common)
    sret5 = sret5.reindex(common)
    sret20 = sret20.reindex(common)
    agg = pd.DataFrame(index=common)
    agg["sector_adv"] = (sret1 > 0).mean(axis=1)
    agg["sector_median1"] = sret1.median(axis=1)
    agg["sector_dispersion1"] = sret1.std(axis=1)
    agg["sector_top3_20"] = sret20.apply(lambda r: r.nlargest(min(3, r.notna().sum())).mean(), axis=1)
    agg["sector_bottom3_20"] = sret20.apply(lambda r: r.nsmallest(min(3, r.notna().sum())).mean(), axis=1)
    agg = agg.reset_index(names="date")
    agg_feats = [c for c in agg.columns if c != "date"]
    for c in agg_feats:
        agg[c] = agg[c].astype(np.float32)
    z = z.merge(agg, on="date", how="left")

    stock = z.loc[z["valid_equity"], ["date", "code", "ret1"]].pivot_table(index="date", columns="code", values="ret1", aggfunc="last").reindex(common)
    sector_cols = list(sret1.columns)
    mapping_rows = []
    block_dates = []
    block_id = 0
    for i in range(60, len(common) - 1, 20):
        win = common[i - 59 : i + 1]
        S = stock.reindex(win).to_numpy(dtype=np.float64)
        I = sret1.reindex(win).to_numpy(dtype=np.float64)
        valid = np.isfinite(S)
        count = valid.sum(axis=0)
        smean = np.nanmean(S, axis=0)
        sstd = np.nanstd(S, axis=0)
        imean = np.nanmean(I, axis=0)
        istd = np.nanstd(I, axis=0)
        Sz = (S - smean) / np.where(sstd > 1e-12, sstd, np.nan)
        Iz = (I - imean) / np.where(istd > 1e-12, istd, np.nan)
        Sz = np.nan_to_num(Sz, nan=0.0, posinf=0.0, neginf=0.0)
        Iz = np.nan_to_num(Iz, nan=0.0, posinf=0.0, neginf=0.0)
        corr = Sz.T @ Iz
        corr = corr / np.maximum(count[:, None], 1)
        corr[count < 40, :] = -np.inf
        best = np.argmax(corr, axis=1)
        best_corr = corr[np.arange(len(best)), best]
        for j, code in enumerate(stock.columns):
            if np.isfinite(best_corr[j]) and best_corr[j] > -0.99:
                mapping_rows.append({"block_id": block_id, "code": str(code), "peer_index": sector_cols[int(best[j])], "peer_corr60": float(best_corr[j])})
        future_dates = common[i + 1 : min(i + 21, len(common))]
        block_dates.extend({"date": int(d), "block_id": block_id} for d in future_dates)
        block_id += 1
    peer_map = pd.DataFrame(mapping_rows)
    block_map = pd.DataFrame(block_dates)
    z = z.merge(block_map, on="date", how="left")
    if len(peer_map):
        z = z.merge(peer_map, on=["block_id", "code"], how="left")
    else:
        z["peer_index"] = None
        z["peer_corr60"] = np.nan

    rank20 = sret20.rank(axis=1, pct=True)
    long_parts = []
    for name, mat in (("peer_ret1", sret1), ("peer_ret5", sret5), ("peer_ret20", sret20), ("peer_rank20", rank20)):
        q = mat.stack(dropna=False).rename(name).reset_index().rename(columns={"index_name": "peer_index"})
        long_parts.append(q)
    peer_daily = long_parts[0]
    for q in long_parts[1:]:
        peer_daily = peer_daily.merge(q, on=["date", "peer_index"], how="outer")
    z = z.merge(peer_daily, on=["date", "peer_index"], how="left")
    peer_feats = ["peer_corr60", "peer_ret1", "peer_ret5", "peer_ret20", "peer_rank20"]
    for c in peer_feats:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)
    return z, agg_feats + peer_feats, peer_map


def add_fundamental_context(z: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = []
    rp = CTX / "monthly_revenue_point_in_time.csv.gz"
    r = pd.read_csv(rp, low_memory=False)
    r["date"] = ymd(pd.to_datetime(r["available_date"])).astype(np.int64)
    r["code"] = r["code"].astype(str).str.zfill(4)
    for c in ("revenue_thousand", "yoy_pct", "mom_pct"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.sort_values(["date", "code"])
    rg = r.groupby("code", group_keys=False)
    r["rev_log"] = np.log1p(r["revenue_thousand"].clip(lower=0))
    r["rev_yoy"] = r["yoy_pct"] / 100.0
    r["rev_mom"] = r["mom_pct"] / 100.0
    r["rev_yoy_accel1"] = r["rev_yoy"] - rg["rev_yoy"].shift(1)
    r["rev_yoy_accel3"] = r["rev_yoy"] - rg["rev_yoy"].shift(3)
    r["rev_yoy_mean3"] = rg["rev_yoy"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    rev_feats = ["rev_log", "rev_yoy", "rev_mom", "rev_yoy_accel1", "rev_yoy_accel3", "rev_yoy_mean3"]
    left = z.sort_values(["date", "code"])
    right = r[["date", "code"] + rev_feats].sort_values(["date", "code"])
    z = pd.merge_asof(left, right, on="date", by="code", direction="backward", allow_exact_matches=True)
    feats += rev_feats

    vp = CTX / "daily_valuation_point_in_time.csv.gz"
    v = pd.read_csv(vp, low_memory=False)
    v["date"] = ymd(pd.to_datetime(v["date"])).astype(np.int64)
    v["code"] = v["code"].astype(str).str.zfill(4)
    for c in ("pe", "pb", "dividend_yield_pct"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v = v.sort_values(["date", "code"])
    vg = v.groupby("code", group_keys=False)
    v["pe_med252"] = vg["pe"].transform(lambda s: s.where(s > 0).rolling(252, min_periods=60).median())
    v["pb_med252"] = vg["pb"].transform(lambda s: s.where(s > 0).rolling(252, min_periods=60).median())
    v["pe_rel252"] = v["pe"] / v["pe_med252"] - 1
    v["pb_rel252"] = v["pb"] / v["pb_med252"] - 1
    val_feats = ["pe", "pb", "dividend_yield_pct", "pe_rel252", "pb_rel252"]
    left = z.sort_values(["date", "code"])
    right = v[["date", "code"] + val_feats].sort_values(["date", "code"])
    z = pd.merge_asof(left, right, on="date", by="code", direction="backward", allow_exact_matches=True)
    feats += val_feats
    for c in feats:
        z[c] = pd.to_numeric(z[c], errors="coerce").astype(np.float32)
    return z, feats


def add_path_labels(z: pd.DataFrame) -> pd.DataFrame:
    z = z.sort_values(["code", "date"]).copy()
    g = z.groupby("code", group_keys=False)
    z["ret_jump"] = g["close"].pct_change().abs().gt(0.35)
    future_bad = g["ret_jump"].transform(lambda s: s.shift(-1).iloc[::-1].rolling(120, min_periods=1).max().iloc[::-1]).fillna(False).astype(bool)
    for h in (20, 60, 120):
        z[f"fwd{h}"] = g["close"].shift(-h) / z["close"] - 1
        fhi = g["high"].transform(lambda s, h=h: v4.future_roll(s, h, "max"))
        flo = g["low"].transform(lambda s, h=h: v4.future_roll(s, h, "min"))
        z[f"mfe{h}h"] = fhi / z["close"] - 1
        z[f"mae{h}l"] = flo / z["close"] - 1
        z.loc[future_bad, [f"fwd{h}", f"mfe{h}h", f"mae{h}l"]] = np.nan
    fhi5 = g["high"].transform(lambda s: v4.future_roll(s, 5, "max"))
    flo5 = g["low"].transform(lambda s: v4.future_roll(s, 5, "min"))
    z["mfe5h"] = fhi5 / z["close"] - 1
    z["mae5l"] = flo5 / z["close"] - 1
    z.loc[future_bad, ["mfe5h", "mae5l"]] = np.nan

    t_hi = np.full(len(z), np.nan, dtype=np.float32)
    t_lo5 = np.full(len(z), np.nan, dtype=np.float32)
    hit_days = {t: np.full(len(z), np.nan, dtype=np.float32) for t in TARGET_LEVELS}
    for _, idx in z.groupby("code", sort=False).indices.items():
        pos = np.asarray(idx, dtype=int)
        n = len(pos)
        if n <= 120:
            continue
        hi = z.iloc[pos]["high"].to_numpy(dtype=float)
        lo = z.iloc[pos]["low"].to_numpy(dtype=float)
        cl = z.iloc[pos]["close"].to_numpy(dtype=float)
        wh = np.lib.stride_tricks.sliding_window_view(hi[1:], 120)
        relh = wh / cl[: len(wh), None] - 1.0
        safe_h = np.where(np.isfinite(relh), relh, -np.inf)
        t_hi[pos[: len(wh)]] = (np.argmax(safe_h, axis=1) + 1).astype(np.float32)
        for target in TARGET_LEVELS:
            mask = relh >= target
            anyhit = mask.any(axis=1)
            first = np.argmax(mask, axis=1) + 1
            arr = np.where(anyhit, first, np.nan).astype(np.float32)
            hit_days[target][pos[: len(wh)]] = arr
        wl = np.lib.stride_tricks.sliding_window_view(lo[1:], 5)
        rell = wl / cl[: len(wl), None] - 1.0
        safe_l = np.where(np.isfinite(rell), rell, np.inf)
        t_lo5[pos[: len(wl)]] = (np.argmin(safe_l, axis=1) + 1).astype(np.float32)
    z["time_to_high120"] = t_hi
    z["time_to_low5"] = t_lo5
    for t, arr in hit_days.items():
        z[f"hitday_{int(t*100)}"] = arr
    z.loc[future_bad, ["time_to_high120", "time_to_low5"] + [f"hitday_{int(t*100)}" for t in TARGET_LEVELS]] = np.nan
    return z


def sample_fit(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    x = train.dropna(subset=list(MODEL_TARGETS))
    if len(x) <= MAX_TRAIN_ROWS:
        return x
    return x.sample(MAX_TRAIN_ROWS, random_state=seed).sort_values(["date", "code"])


def fit_path_models(train: pd.DataFrame, feats: list[str], seed: int):
    tr = sample_fit(train, seed)
    X = model_frame(tr, feats)
    models = {}
    for i, target in enumerate(MODEL_TARGETS):
        model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=140, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=3.0, random_state=seed + i)
        model.fit(X, tr[target].to_numpy(dtype=np.float32))
        models[target] = model
    return models, len(tr)


def predict_models(models, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = model_frame(df, feats)
    out = pd.DataFrame(index=df.index)
    for target, model in models.items():
        out[f"pred_{target}"] = model.predict(X).astype(np.float32)
    return out


def wilson_lower(k: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return np.nan
    p = k / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    rad = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
    return (center-rad)/den


def build_surface(cal: pd.DataFrame, pred: pd.DataFrame):
    x = cal.copy().join(pred)
    valid = x["pred_mfe120h"].notna() & x["pred_mae120l"].notna() & x["mfe120h"].notna()
    x = x[valid].copy()
    x["mfe_bin"], mfe_edges = pd.qcut(x["pred_mfe120h"], MFE_BINS, labels=False, retbins=True, duplicates="drop")
    x["mae_bin"], mae_edges = pd.qcut(x["pred_mae120l"], MAE_BINS, labels=False, retbins=True, duplicates="drop")
    rows = []
    for (mb, ab), q in x.groupby(["mfe_bin", "mae_bin"], dropna=True):
        n = len(q)
        r = {"mfe_bin": int(mb), "mae_bin": int(ab), "cell_n": int(n)}
        for h in (20, 60, 120):
            for t in TARGET_LEVELS:
                col = f"p_hit{int(t*100)}_{h}"
                mask = q[f"mfe{h}h"].notna()
                y = (q.loc[mask, f"mfe{h}h"] >= t)
                if len(y):
                    k = int(y.sum())
                    r[col] = float(k / len(y))
                    r[col + "_wilson"] = float(wilson_lower(k, len(y)))
                else:
                    r[col] = np.nan
                    r[col + "_wilson"] = np.nan
            for t in (0.05, 0.10):
                mask = q[f"mae{h}l"].notna()
                y = (q.loc[mask, f"mae{h}l"] <= -t)
                r[f"p_down{int(t*100)}_{h}"] = float(y.mean()) if len(y) else np.nan
        for col in ("mfe120h", "mae120l", "mae5l", "time_to_high120", "time_to_low5"):
            s = pd.to_numeric(q[col], errors="coerce").dropna()
            for label, qq in (("q25", .25), ("q50", .50), ("q75", .75)):
                r[f"{col}_{label}"] = float(s.quantile(qq)) if len(s) else np.nan
        for t in TARGET_LEVELS:
            s = pd.to_numeric(q[f"hitday_{int(t*100)}"], errors="coerce").dropna()
            r[f"hitday_{int(t*100)}_q25"] = float(s.quantile(.25)) if len(s) else np.nan
            r[f"hitday_{int(t*100)}_q50"] = float(s.quantile(.50)) if len(s) else np.nan
            r[f"hitday_{int(t*100)}_q75"] = float(s.quantile(.75)) if len(s) else np.nan
        rows.append(r)
    return pd.DataFrame(rows), np.asarray(mfe_edges, float), np.asarray(mae_edges, float)


def assign_bins(pred: pd.DataFrame, mfe_edges: np.ndarray, mae_edges: np.ndarray) -> pd.DataFrame:
    out = pred.copy()
    out["mfe_bin"] = np.clip(np.digitize(out["pred_mfe120h"], mfe_edges[1:-1], right=True), 0, max(0, len(mfe_edges)-2))
    out["mae_bin"] = np.clip(np.digitize(out["pred_mae120l"], mae_edges[1:-1], right=True), 0, max(0, len(mae_edges)-2))
    return out


def attach_surface(df: pd.DataFrame, pred: pd.DataFrame, surface: pd.DataFrame, mfe_edges, mae_edges) -> pd.DataFrame:
    p = assign_bins(pred, mfe_edges, mae_edges)
    x = df.copy().join(p)
    x = x.merge(surface, on=["mfe_bin", "mae_bin"], how="left")
    stated = np.full(len(x), np.nan, dtype=float)
    stated_p = np.full(len(x), np.nan, dtype=float)
    stated_w = np.full(len(x), np.nan, dtype=float)
    for t in TARGET_LEVELS:
        pc = f"p_hit{int(t*100)}_120"
        wc = pc + "_wilson"
        good = (x["cell_n"] >= MIN_CELL_N) & (x[pc] >= FORMAL_PROB) & (x[wc] >= 0.80)
        stated[good.to_numpy()] = t
        stated_p[good.to_numpy()] = x.loc[good, pc]
        stated_w[good.to_numpy()] = x.loc[good, wc]
    x["stated_target_return"] = stated
    x["stated_target_probability"] = stated_p
    x["stated_target_wilson"] = stated_w
    x["ai_buy"] = np.isfinite(stated)
    return x


def ece_score(prob: pd.Series, y: pd.Series, bins=10):
    q = pd.DataFrame({"p": prob, "y": y}).dropna()
    if not len(q):
        return np.nan
    q["bin"] = pd.cut(q["p"], bins=np.linspace(0, 1, bins+1), include_lowest=True, labels=False)
    total = len(q)
    ece = 0.0
    for _, g in q.groupby("bin"):
        ece += len(g)/total * abs(float(g["p"].mean()) - float(g["y"].mean()))
    return ece


def eval_year(test_scored: pd.DataFrame, year: int) -> dict:
    sel = test_scored[test_scored["ai_buy"]].copy()
    row = {"test_year": year, "test_rows": int(len(test_scored)), "selected_rows": int(len(sel)), "signal_days": int(sel["date"].nunique()), "total_days": int(test_scored["date"].nunique()), "unique_codes": int(sel["code"].nunique()), "mean_stated_target": float(sel["stated_target_return"].mean()) if len(sel) else np.nan, "mean_stated_probability": float(sel["stated_target_probability"].mean()) if len(sel) else np.nan}
    if len(sel):
        hit = sel["mfe120h"] >= sel["stated_target_return"]
        row["stated_target_realized_hit_rate"] = float(hit.mean())
        row["actual_down10_120_rate"] = float((sel["mae120l"] <= -0.10).mean())
        row["mean_mfe120h"] = float(sel["mfe120h"].mean())
        row["mean_mae120l"] = float(sel["mae120l"].mean())
        for h in (20, 60, 120):
            med = test_scored.groupby("date")[f"fwd{h}"].median()
            s = sel[sel[f"fwd{h}"].notna()].copy()
            if len(s):
                s["alpha"] = s[f"fwd{h}"] - s["date"].map(med)
                row[f"mean_fwd{h}"] = float(s[f"fwd{h}"].mean())
                row[f"mean_alpha{h}"] = float(s["alpha"].mean())
            else:
                row[f"mean_fwd{h}"] = np.nan
                row[f"mean_alpha{h}"] = np.nan
    else:
        row.update({"stated_target_realized_hit_rate": np.nan, "actual_down10_120_rate": np.nan, "mean_mfe120h": np.nan, "mean_mae120l": np.nan})
        for h in (20, 60, 120):
            row[f"mean_fwd{h}"] = np.nan
            row[f"mean_alpha{h}"] = np.nan
    p = test_scored["p_hit10_120"]
    y = (test_scored["mfe120h"] >= .10).astype(float).where(test_scored["mfe120h"].notna())
    q = pd.DataFrame({"p": p, "y": y}).dropna()
    if len(q):
        row["brier_hit10_120"] = float(np.mean((q["p"] - q["y"])**2))
        base = float(q["y"].mean())
        row["brier_baseline_hit10_120"] = float(np.mean((base - q["y"])**2))
        row["ece_hit10_120"] = float(ece_score(q["p"], q["y"]))
        grouped = q.groupby(pd.qcut(q["p"], 10, duplicates="drop"), observed=False)[["p", "y"]].mean()
        row["calibration_spearman_hit10_120"] = float(spearmanr(grouped["p"], grouped["y"], nan_policy="omit").statistic) if len(grouped) > 2 else np.nan
    else:
        row["brier_hit10_120"] = row["brier_baseline_hit10_120"] = row["ece_hit10_120"] = row["calibration_spearman_hit10_120"] = np.nan
    lo = test_scored["mfe120h_q25"]
    hi = test_scored["mfe120h_q75"]
    ok = test_scored["mfe120h"].notna() & lo.notna() & hi.notna()
    row["mfe120_interquartile_coverage"] = float(((test_scored.loc[ok, "mfe120h"] >= lo[ok]) & (test_scored.loc[ok, "mfe120h"] <= hi[ok])).mean()) if ok.any() else np.nan
    return row


def live_records(scored: pd.DataFrame, semantic: pd.DataFrame) -> dict:
    latest = int(scored["date"].max())
    live = scored[(scored["date"] == latest) & scored["ai_buy"]].copy()
    semantic = semantic.copy()
    semantic["code"] = semantic["code"].astype(str).str.zfill(4)
    live = live.merge(semantic[["code", "industry_current"]], on="code", how="left")
    if len(live):
        live["entry_low"] = live["close"] * (1 + live["mae5l_q25"])
        live["entry_high"] = live["close"] * (1 + live["mae5l_q50"])
        live["target_conservative"] = live["close"] * (1 + live["mfe120h_q25"])
        live["target_base"] = live["close"] * (1 + live["mfe120h_q50"])
        live["target_optimistic"] = live["close"] * (1 + live["mfe120h_q75"])
        live = live.sort_values(["stated_target_return", "stated_target_probability", "pred_mfe120h"], ascending=False)
    keep = ["date", "code", "name", "close", "industry_current", "peer_index", "peer_corr60", "peer_ret20", "peer_rank20", "rev_yoy", "rev_yoy_accel1", "pe", "pb", "pe_rel252", "pb_rel252", "stated_target_return", "stated_target_probability", "stated_target_wilson", "p_hit5_20", "p_hit5_60", "p_hit5_120", "p_hit10_20", "p_hit10_60", "p_hit10_120", "p_hit15_120", "p_hit20_120", "p_hit30_120", "p_down5_60", "p_down10_60", "p_down10_120", "entry_low", "entry_high", "time_to_low5_q25", "time_to_low5_q50", "time_to_low5_q75", "target_conservative", "target_base", "target_optimistic", "time_to_high120_q25", "time_to_high120_q50", "time_to_high120_q75", "cell_n", "pred_mfe20h", "pred_mfe60h", "pred_mfe120h", "pred_mae60l", "pred_mae120l"]
    keep = [c for c in keep if c in live.columns]
    return {"as_of": latest, "decision": "BUY_CANDIDATES" if len(live) else "NONE", "buy_count": int(len(live)), "candidates": live[keep].head(200).replace({np.nan: None}).to_dict("records")}


def main():
    print("[V6] load market tape", flush=True)
    px = v4.load_px()
    px = px[px["date"] >= START_DATE].copy()
    inst = v4.load_inst()
    ds = v4.add_channels(px, inst)
    ds, f_tape = add_tape_features(ds)
    ds, f_struct = add_structure_and_seasonality(ds)
    ds, f_macro = add_macro_context(ds)
    print("[V6] add sector context", flush=True)
    ds, f_sector, peer_map = add_sector_context(ds)
    print("[V6] add fundamental context", flush=True)
    ds, f_fund = add_fundamental_context(ds)
    print("[V6] labels", flush=True)
    ds = add_path_labels(ds)
    feats = f_tape + f_struct + f_macro + f_sector + f_fund
    core = f_tape
    ds["core_feature_ok"] = ds[core].notna().mean(axis=1) >= .80
    ds["universe_ok"] = ds["valid_equity"] & (ds["hist_count"] >= 120) & ds["core_feature_ok"]
    ds = ds.sort_values(["date", "code"]).reset_index(drop=True)
    print(f"[V6] rows={len(ds):,} features={len(feats)} universe={int(ds.universe_ok.sum()):,}", flush=True)

    dates = np.array(sorted(ds["date"].unique()), dtype=np.int64)
    d2i = {int(d): i for i, d in enumerate(dates)}
    yearly = []
    all_selected = []
    cal_audit = {}
    for year in TEST_YEARS:
        test = ds[(ds["date"] >= year*10000+101) & (ds["date"] <= year*10000+1231) & ds["universe_ok"]].copy()
        first = int(test["date"].min())
        cutoff = int(dates[d2i[first] - PURGE])
        train = ds[(ds["date"] < cutoff) & ds["universe_ok"] & ds["mfe120h"].notna()].copy()
        tdates = np.array(sorted(train["date"].unique()), dtype=np.int64)
        if len(tdates) <= CAL_DAYS + 100:
            raise RuntimeError(f"insufficient V6 history {year}")
        cal_dates = tdates[-CAL_DAYS:]
        fit = train[train["date"] < cal_dates[0]].copy()
        cal = train[train["date"].isin(cal_dates)].copy()
        models, fit_rows = fit_path_models(fit, feats, RNG_SEED + year)
        pred_cal = predict_models(models, cal, feats)
        surface, mfe_edges, mae_edges = build_surface(cal, pred_cal)
        pred_test = predict_models(models, test, feats)
        scored = attach_surface(test, pred_test, surface, mfe_edges, mae_edges)
        row = eval_year(scored, year)
        row.update({"train_cutoff": cutoff, "fit_rows": fit_rows, "calibration_rows": int(len(cal)), "surface_cells": int(len(surface))})
        yearly.append(row)
        sel = scored[scored["ai_buy"]].copy()
        sel["test_year"] = year
        all_selected.append(sel)
        cal_audit[str(year)] = {"mfe_edges": mfe_edges.tolist(), "mae_edges": mae_edges.tolist(), "cells": surface.replace({np.nan: None}).to_dict("records")}
        print("[V6 OOS]", year, json.dumps(row, ensure_ascii=False), flush=True)

    yearly_df = pd.DataFrame(yearly)
    selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    overall = {"architecture": "multi-context causal path regressors -> past-only 2D calibration surface -> target/time/entry distribution", "features": len(feats), "fixed_top_n": False, "can_abstain": True, "selected_rows": int(len(selected)), "signal_days": int(selected["date"].nunique()) if len(selected) else 0, "unique_selected_codes": int(selected["code"].nunique()) if len(selected) else 0, "mean_stated_target": float(selected["stated_target_return"].mean()) if len(selected) else np.nan, "realized_stated_target_hit_rate": float((selected["mfe120h"] >= selected["stated_target_return"]).mean()) if len(selected) else np.nan, "mean_stated_probability": float(selected["stated_target_probability"].mean()) if len(selected) else np.nan, "positive_alpha_years_20": int((yearly_df["mean_alpha20"] > 0).sum()), "positive_alpha_years_60": int((yearly_df["mean_alpha60"] > 0).sum()), "positive_alpha_years_120": int((yearly_df["mean_alpha120"] > 0).sum()), "median_ece_hit10_120": float(yearly_df["ece_hit10_120"].median()), "median_calibration_spearman_hit10_120": float(yearly_df["calibration_spearman_hit10_120"].median())}
    gate = {"realized_high_confidence_hit_rate_at_least_80pct": bool(np.isfinite(overall["realized_stated_target_hit_rate"]) and overall["realized_stated_target_hit_rate"] >= .80), "four_of_five_positive_alpha_60_120": bool(overall["positive_alpha_years_60"] >= 4 and overall["positive_alpha_years_120"] >= 4), "median_ece_le_0p08": bool(np.isfinite(overall["median_ece_hit10_120"]) and overall["median_ece_hit10_120"] <= .08), "calibration_monotone_positive": bool(np.isfinite(overall["median_calibration_spearman_hit10_120"]) and overall["median_calibration_spearman_hit10_120"] > 0), "selector_abstains_some_days": bool(sum(int(r["total_days"] - r["signal_days"]) for r in yearly) > 0), "selector_emits_some_candidates": bool(len(selected) > 0)}
    gate["promising_v6"] = bool(all(gate.values()))

    latest = int(ds["date"].max())
    li = d2i[latest]
    label_cutoff = int(dates[li - PURGE])
    hist = ds[(ds["date"] <= label_cutoff) & ds["universe_ok"] & ds["mfe120h"].notna()].copy()
    hdates = np.array(sorted(hist["date"].unique()), dtype=np.int64)
    cal_dates = hdates[-CAL_DAYS:]
    fit = hist[hist["date"] < cal_dates[0]].copy()
    cal = hist[hist["date"].isin(cal_dates)].copy()
    models, fit_rows = fit_path_models(fit, feats, RNG_SEED + 999)
    pred_cal = predict_models(models, cal, feats)
    surface, mfe_edges, mae_edges = build_surface(cal, pred_cal)
    live = ds[(ds["date"] == latest) & ds["universe_ok"]].copy()
    pred_live = predict_models(models, live, feats)
    scored_live = attach_surface(live, pred_live, surface, mfe_edges, mae_edges)
    semantic = pd.read_csv(CTX / "company_industry_live_snapshot.csv.gz", dtype={"code": str})
    live_json = live_records(scored_live, semantic)
    live_json.update({"train_label_cutoff": label_cutoff, "fit_rows": fit_rows, "calibration_rows": int(len(cal)), "universe_count": int(len(live)), "formal_probability_threshold": FORMAL_PROB})

    yearly_df.to_csv(OUT / "V6_YEARLY_OOS.csv", index=False)
    selected.to_csv(OUT / "V6_OOS_CANDIDATES.csv", index=False)
    (OUT / "V6_OVERALL.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "V6_GATE.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "V6_LIVE.json").write_text(json.dumps(live_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "V6_CALIBRATION_AUDIT.json").write_text(json.dumps(cal_audit, ensure_ascii=False), encoding="utf-8")
    print("\n[V6 YEARLY]\n", yearly_df.to_string(index=False), flush=True)
    print("\n[V6 OVERALL]\n", json.dumps(overall, ensure_ascii=False, indent=2), flush=True)
    print("\n[V6 GATE]\n", json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
    print("\n[V6 LIVE]\n", json.dumps({**live_json, "candidates": live_json["candidates"][:30]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
