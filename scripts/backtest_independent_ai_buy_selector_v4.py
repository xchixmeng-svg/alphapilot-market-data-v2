#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
HIST_OLD = ROOT / "data" / "history" / "2007-2019" / "raw"
HIST_NEW = ROOT / "data" / "history" / "2020-2025"
YTD = ROOT / "data" / "history" / "2026-YTD"
OUT = ROOT / "independent_ai_buy_selector_v4_results"
OUT.mkdir(exist_ok=True)

START_YEAR = 2016
TEST_YEARS = range(2021, 2026)
PURGE = 120
LAGS = (0, 1, 2, 5, 10, 20, 60, 119)
MAX_TRAIN_ROWS = 320_000
CAL_DAYS = 252
RNG_SEED = 92615

STOCK_CHANNELS = (
    "ret1",
    "gap1",
    "range1",
    "body1",
    "amount_logchg",
    "foreign_ratio",
    "trust_ratio",
)
MARKET_CHANNELS = ("mkt_ret1", "mkt_advance", "mkt_dispersion")


def to_yyyymmdd(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y%m%d").astype(np.int64)
    z = s.astype(str).str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    return pd.to_numeric(z, errors="coerce").astype("Int64")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ren = {
        "trade_date": "date", "stock_id": "code",
        "Trading_Volume": "volume", "Trade_Volume": "volume",
    }
    x = x.rename(columns={k: v for k, v in ren.items() if k in x.columns})
    need = ["date", "code", "name", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in x.columns]
    if missing:
        raise RuntimeError(f"OHLCV missing columns {missing}; got={list(x.columns)}")
    x = x[need].copy()
    x["date"] = to_yyyymmdd(x["date"])
    x["code"] = x["code"].astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(4)
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date", "code", "close"]).copy()
    x["date"] = x["date"].astype(np.int64)
    return x


def load_old_year(year: int) -> pd.DataFrame:
    zp = HIST_OLD / f"yearly_{year}.zip"
    if not zp.exists():
        raise RuntimeError(f"missing immutable history {zp}")
    with zipfile.ZipFile(zp) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"no csv in {zp}")
        with z.open(members[0]) as f:
            return normalize_ohlcv(pd.read_csv(f, low_memory=False))


def load_px() -> pd.DataFrame:
    parts = [load_old_year(y) for y in range(START_YEAR, 2020)]
    for y in range(2020, 2026):
        p = HIST_NEW / f"ohlcv_{y}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing {p}")
        parts.append(normalize_ohlcv(pd.read_parquet(p)))

    ytdp = YTD / "ohlcv_2026_ytd.csv"
    if ytdp.exists():
        parts.append(normalize_ohlcv(pd.read_csv(ytdp, low_memory=False)))

    # Daily folders are authoritative for dates later than a weekly YTD asset.
    for d in sorted((ROOT / "data").glob("2026-??-??")):
        n = d / "normalized"
        for p in (n / "twse_ohlcv.csv", n / "tpex_ohlcv.csv"):
            if p.exists():
                try:
                    parts.append(normalize_ohlcv(pd.read_csv(p, low_memory=False)))
                except Exception as e:
                    print(f"[WARN] skip daily OHLCV {p}: {e}", flush=True)

    px = pd.concat(parts, ignore_index=True)
    px = px.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
    px = px[(px["date"] >= START_YEAR * 10000 + 101) & (px["close"] > 0)].copy()
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    return px


def normalize_inst(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ren = {"trade_date": "date", "stock_id": "code"}
    x = x.rename(columns={k: v for k, v in ren.items() if k in x.columns})
    if "date" not in x.columns or "code" not in x.columns:
        raise RuntimeError(f"institutional missing date/code; got={list(x.columns)}")
    for c in ("foreign_net", "trust_net"):
        if c not in x.columns:
            x[c] = 0.0
    x["date"] = to_yyyymmdd(x["date"])
    x["code"] = x["code"].astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(4)
    for c in ("foreign_net", "trust_net"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["date", "code"]).copy()
    x["date"] = x["date"].astype(np.int64)
    return x[["date", "code", "foreign_net", "trust_net"]]


def load_inst() -> pd.DataFrame:
    parts = []
    p = HIST_NEW / "institutional_2020_2025.parquet"
    if p.exists():
        parts.append(normalize_inst(pd.read_parquet(p)))
    yp = YTD / "institutional_2026_ytd.csv"
    if yp.exists():
        parts.append(normalize_inst(pd.read_csv(yp, low_memory=False)))
    if not parts:
        return pd.DataFrame(columns=["date", "code", "foreign_net", "trust_net"])
    x = pd.concat(parts, ignore_index=True)
    x = x.groupby(["date", "code"], as_index=False)[["foreign_net", "trust_net"]].sum(min_count=1)
    return x


def future_roll(s: pd.Series, h: int, kind: str) -> pd.Series:
    f = s.shift(-1).iloc[::-1]
    r = f.rolling(h, min_periods=h)
    out = (r.min() if kind == "min" else r.max()).iloc[::-1]
    return out


def add_channels(px: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    x = px.copy()
    x["valid_equity"] = x["code"].str.fullmatch(r"[1-9]\d{3}", na=False)
    x = x.merge(inst, on=["date", "code"], how="left")
    x["inst_available"] = (x["foreign_net"].notna() | x["trust_net"].notna()).astype(np.float32)
    x[["foreign_net", "trust_net"]] = x[["foreign_net", "trust_net"]].fillna(0.0)

    x = x.sort_values(["code", "date"]).reset_index(drop=True)
    g = x.groupby("code", group_keys=False)
    prev_close = g["close"].shift(1)
    prev_amount = (x["close"] * x["volume"]).groupby(x["code"]).shift(1)

    x["ret1"] = x["close"] / prev_close - 1.0
    x["gap1"] = x["open"] / prev_close - 1.0
    x["range1"] = (x["high"] - x["low"]) / prev_close.replace(0, np.nan)
    x["body1"] = x["close"] / x["open"].replace(0, np.nan) - 1.0
    amt = (x["close"] * x["volume"]).clip(lower=0)
    x["amount_logchg"] = np.log1p(amt) - np.log1p(prev_amount.clip(lower=0))
    vol = x["volume"].replace(0, np.nan)
    x["foreign_ratio"] = (x["foreign_net"] / vol).clip(-5, 5).fillna(0.0)
    x["trust_ratio"] = (x["trust_net"] / vol).clip(-5, 5).fillna(0.0)

    daily = x.groupby("date").agg(
        mkt_ret1=("ret1", "median"),
        mkt_advance=("ret1", lambda s: float((s > 0).mean())),
        mkt_dispersion=("ret1", "std"),
    ).reset_index()
    x = x.merge(daily, on="date", how="left").sort_values(["code", "date"]).reset_index(drop=True)

    g = x.groupby("code", group_keys=False)
    x["hist_count"] = g.cumcount() + 1
    for c in STOCK_CHANNELS:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float32)
    for c in MARKET_CHANNELS:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float32)
    return x


def add_raw_tape_features(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    z = x.copy()
    feats = []
    g = z.groupby("code", group_keys=False)
    for lag in LAGS:
        for c in STOCK_CHANNELS:
            name = f"{c}_lag{lag}"
            z[name] = g[c].shift(lag).astype(np.float32)
            feats.append(name)

    # Market channels are identical across stocks on a date; lag them on a daily table, then merge.
    daily = z[["date", *MARKET_CHANNELS]].drop_duplicates("date").sort_values("date").copy()
    for lag in LAGS:
        for c in MARKET_CHANNELS:
            name = f"{c}_lag{lag}"
            daily[name] = daily[c].shift(lag).astype(np.float32)
            feats.append(name)
    z = z.drop(columns=[c for c in MARKET_CHANNELS if c in z.columns])
    z = z.merge(daily[["date"] + [f"{c}_lag{lag}" for lag in LAGS for c in MARKET_CHANNELS]],
                on="date", how="left")
    return z, feats


def add_labels(x: pd.DataFrame) -> pd.DataFrame:
    z = x.sort_values(["code", "date"]).copy()
    g = z.groupby("code", group_keys=False)
    z["ret_jump"] = g["close"].pct_change().abs().gt(0.35)
    g = z.groupby("code", group_keys=False)
    future_bad = g["ret_jump"].transform(
        lambda s: s.shift(-1).iloc[::-1].rolling(120, min_periods=1).max().iloc[::-1]
    ).fillna(False).astype(bool)

    for h in (5, 20, 60, 120):
        z[f"fwd{h}"] = g["close"].shift(-h) / z["close"] - 1.0
        z.loc[future_bad, f"fwd{h}"] = np.nan

    for h in (20, 60, 120):
        fmin = g["close"].transform(lambda s, h=h: future_roll(s, h, "min"))
        fmax = g["close"].transform(lambda s, h=h: future_roll(s, h, "max"))
        z[f"mae{h}"] = fmin / z["close"] - 1.0
        z[f"mfe{h}"] = fmax / z["close"] - 1.0
        z.loc[future_bad, [f"mae{h}", f"mfe{h}"]] = np.nan

    label_cols = ["fwd20", "fwd60", "fwd120", "mae60", "mae120"]
    for c in label_cols:
        z[f"rank_{c}"] = z.groupby("date")[c].rank(pct=True)
    ranks_ok = np.logical_and.reduce([(z[f"rank_{c}"] > 0.5).to_numpy() for c in label_cols])
    abs_ok = np.logical_and.reduce([(z[f"fwd{h}"] > 0).to_numpy() for h in (20, 60, 120)])
    labels_available = z[label_cols].notna().all(axis=1).to_numpy()
    y = (ranks_ok & abs_ok).astype(np.float32)
    y[~labels_available] = np.nan
    z["worth_buy_label"] = y
    return z


def model_frame(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32, copy=False)


def sample_training(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    pos = df[df["worth_buy_label"] == 1]
    neg = df[df["worth_buy_label"] == 0]
    keep_pos = pos if len(pos) <= max_rows // 2 else pos.sample(max_rows // 2, random_state=seed)
    remaining = max_rows - len(keep_pos)
    keep_neg = neg.sample(min(remaining, len(neg)), random_state=seed + 1)
    out = pd.concat([keep_pos, keep_neg], ignore_index=False)
    return out.sort_values(["date", "code"])


def fit_classifier(train: pd.DataFrame, feats: list[str], seed: int) -> HistGradientBoostingClassifier:
    tr = sample_training(train, MAX_TRAIN_ROWS, seed)
    X = model_frame(tr, feats)
    y = tr["worth_buy_label"].astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=3.0,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return model


def learned_threshold(train: pd.DataFrame, feats: list[str], seed: int) -> tuple[float, float, int]:
    dates = np.array(sorted(train["date"].unique()), dtype=np.int64)
    if len(dates) < CAL_DAYS + 120:
        base_rate = float(train["worth_buy_label"].mean())
        return 0.5, base_rate, 0
    cal_dates = dates[-CAL_DAYS:]
    fit = train[train["date"] < cal_dates[0]].copy()
    cal = train[train["date"].isin(cal_dates)].copy()
    if len(fit) < 5000 or len(cal) < 1000:
        base_rate = float(train["worth_buy_label"].mean())
        return 0.5, base_rate, len(cal)
    m = fit_classifier(fit, feats, seed + 100)
    p = m.predict_proba(model_frame(cal, feats))[:, 1]
    base_rate = float(cal["worth_buy_label"].mean())
    q = float(np.clip(1.0 - base_rate, 0.50, 0.995))
    threshold = max(0.5, float(np.quantile(p, q)))
    return threshold, base_rate, len(cal)


def evaluation_row(test: pd.DataFrame, selected: pd.DataFrame, year: int, threshold: float, base_rate: float) -> dict:
    row = {
        "test_year": year,
        "threshold": threshold,
        "train_calibration_base_rate": base_rate,
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
        if row["universe_label_rate"] and np.isfinite(row["selected_label_precision"])
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


def main() -> None:
    print("[V4] load full-market data", flush=True)
    px = load_px()
    inst = load_inst()
    print(f"[V4] px rows={len(px):,} dates={px.date.min()}..{px.date.max()} inst={len(inst):,}", flush=True)

    ds = add_channels(px, inst)
    ds, feats = add_raw_tape_features(ds)
    ds = add_labels(ds)
    ds["feature_ok"] = ds[feats].notna().mean(axis=1) >= 0.85
    ds["universe_ok"] = ds["valid_equity"] & (ds["hist_count"] >= 120) & ds["feature_ok"]
    ds = ds.sort_values(["date", "code"]).reset_index(drop=True)
    print(f"[V4] features={len(feats)} universe rows={int(ds.universe_ok.sum()):,}", flush=True)

    all_dates = np.array(sorted(ds["date"].unique()), dtype=np.int64)
    date_to_i = {int(d): i for i, d in enumerate(all_dates)}
    yearly = []
    all_selected = []

    for year in TEST_YEARS:
        test = ds[
            (ds["date"] >= year * 10000 + 101)
            & (ds["date"] <= year * 10000 + 1231)
            & ds["universe_ok"]
        ].copy()
        if test.empty:
            raise RuntimeError(f"no OOS rows for {year}")
        first = int(test["date"].min())
        cutoff_i = date_to_i[first] - PURGE
        if cutoff_i <= 0:
            raise RuntimeError(f"insufficient purge before {year}")
        cutoff = int(all_dates[cutoff_i])
        train = ds[
            (ds["date"] < cutoff)
            & ds["universe_ok"]
            & ds["worth_buy_label"].notna()
        ].copy()
        if len(train) < 20_000:
            raise RuntimeError(f"insufficient train rows {year}: {len(train)}")

        threshold, base_rate, cal_n = learned_threshold(train, feats, RNG_SEED + year)
        model = fit_classifier(train, feats, RNG_SEED + year)
        test["buy_probability"] = model.predict_proba(model_frame(test, feats))[:, 1]
        selected = test[test["buy_probability"] >= threshold].copy()
        selected["test_year"] = year
        selected["train_cutoff"] = cutoff
        selected["learned_threshold"] = threshold
        all_selected.append(selected)

        row = evaluation_row(test, selected, year, threshold, base_rate)
        row["train_cutoff"] = cutoff
        row["train_rows_full"] = int(len(train))
        row["calibration_rows"] = int(cal_n)
        yearly.append(row)
        print(
            f"[V4] OOS {year} train={len(train):,} threshold={threshold:.4f} "
            f"selected={len(selected):,} signal_days={row['signal_days']}/{row['total_test_days']} "
            f"precision={row['selected_label_precision']:.4f} lift={row['precision_lift']:.2f}",
            flush=True,
        )

    diag = pd.DataFrame(yearly)
    selected = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    diag.to_csv(OUT / "INDEPENDENT_AI_BUY_SELECTOR_V4_YEARLY.csv", index=False)
    if len(selected):
        keep = [
            "date", "code", "name", "buy_probability", "learned_threshold",
            "test_year", "train_cutoff", "worth_buy_label",
            "fwd5", "fwd20", "fwd60", "fwd120",
            "mfe20", "mae20", "mfe60", "mae60", "mfe120", "mae120",
        ]
        selected[keep].to_csv(OUT / "INDEPENDENT_AI_BUY_SELECTOR_V4_OOS_PICKS.csv", index=False)

    overall = {
        "architecture": "raw temporal tape -> HistGradientBoostingClassifier -> learned calibration threshold -> variable-count/abstaining selector",
        "features": feats,
        "fixed_top_n": False,
        "can_abstain": True,
        "selected_rows": int(len(selected)),
        "unique_selected_codes": int(selected["code"].nunique()) if len(selected) else 0,
        "signal_days": int(selected["date"].nunique()) if len(selected) else 0,
        "no_signal_days_sum_by_year": int(diag["no_signal_days"].sum()),
        "median_threshold": float(diag["threshold"].median()),
        "median_precision_lift": float(diag["precision_lift"].median()),
    }
    for h in (5, 20, 60, 120):
        overall[f"mean_fwd{h}"] = float(diag[f"mean_fwd{h}"].mean())
        overall[f"mean_alpha{h}"] = float(diag[f"mean_alpha{h}"].mean())
        overall[f"positive_years_{h}"] = int((diag[f"mean_fwd{h}"] > 0).sum())
    for h in (20, 60, 120):
        overall[f"mean_mae{h}"] = float(diag[f"mean_mae{h}"].mean())
        overall[f"universe_mean_mae{h}"] = float(diag[f"universe_mean_mae{h}"].mean())
        overall[f"mean_mfe{h}"] = float(diag[f"mean_mfe{h}"].mean())
        overall[f"universe_mean_mfe{h}"] = float(diag[f"universe_mean_mfe{h}"].mean())

    gate = {
        "absolute_positive_20_60_120": bool(all(overall[f"mean_fwd{h}"] > 0 for h in (20, 60, 120))),
        "positive_alpha_all_horizons": bool(all(overall[f"mean_alpha{h}"] > 0 for h in (5, 20, 60, 120))),
        "four_of_five_positive_years_20_60_120": bool(all(overall[f"positive_years_{h}"] >= 4 for h in (20, 60, 120))),
        "precision_lift_at_least_1p5": bool(overall["median_precision_lift"] >= 1.5),
        "mae_better_than_universe_60_120": bool(
            overall["mean_mae60"] >= overall["universe_mean_mae60"]
            and overall["mean_mae120"] >= overall["universe_mean_mae120"]
        ),
        "selector_demonstrates_abstention": bool(overall["no_signal_days_sum_by_year"] > 0),
        "selector_emits_some_candidates": bool(overall["selected_rows"] > 0),
    }
    gate["promising_independent_ai_selector_v4"] = all(gate.values())

    # Live/current inference: train only on labels whose full 120-session future is already known.
    latest_date = int(ds["date"].max())
    latest_i = date_to_i[latest_date]
    live_cutoff_i = latest_i - PURGE
    if live_cutoff_i <= 0:
        raise RuntimeError("insufficient history for live inference")
    live_cutoff = int(all_dates[live_cutoff_i])
    live_train = ds[
        (ds["date"] <= live_cutoff)
        & ds["universe_ok"]
        & ds["worth_buy_label"].notna()
    ].copy()
    live = ds[(ds["date"] == latest_date) & ds["universe_ok"]].copy()
    if live.empty:
        raise RuntimeError(f"no live universe rows for {latest_date}")
    live_threshold, live_base_rate, live_cal_n = learned_threshold(live_train, feats, RNG_SEED + 999)
    live_model = fit_classifier(live_train, feats, RNG_SEED + 999)
    live["buy_probability"] = live_model.predict_proba(model_frame(live, feats))[:, 1]
    live = live.sort_values("buy_probability", ascending=False)
    live["learned_threshold"] = live_threshold
    live["ai_buy"] = live["buy_probability"] >= live_threshold
    live_out = live[["date", "code", "name", "close", "buy_probability", "learned_threshold", "ai_buy"]].copy()
    live_out.to_csv(OUT / "INDEPENDENT_AI_BUY_SELECTOR_V4_LIVE.csv", index=False)
    live_buys = live_out[live_out["ai_buy"]].copy()
    live_summary = {
        "as_of": latest_date,
        "train_label_cutoff": live_cutoff,
        "universe_count": int(len(live_out)),
        "learned_threshold": float(live_threshold),
        "calibration_base_rate": float(live_base_rate),
        "calibration_rows": int(live_cal_n),
        "buy_count": int(len(live_buys)),
        "decision": "BUY_CANDIDATES" if len(live_buys) else "NONE",
        "candidates": live_buys.head(100).to_dict(orient="records"),
        "top_scores_for_audit": live_out.head(20).to_dict(orient="records"),
    }

    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V4_OVERALL.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V4_GATE.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "INDEPENDENT_AI_BUY_SELECTOR_V4_LIVE.json").write_text(
        json.dumps(live_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[V4 YEARLY]\n" + diag.to_string(index=False), flush=True)
    print("\n[V4 OVERALL]\n" + json.dumps(overall, indent=2), flush=True)
    print("\n[V4 GATE]\n" + json.dumps(gate, indent=2), flush=True)
    print("\n[V4 LIVE]\n" + json.dumps(live_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
