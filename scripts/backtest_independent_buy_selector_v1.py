#!/usr/bin/env python3
"""AlphaPilot Independent Buy Selector V1.

Fully independent from R10. Reads only immutable OHLCV/institutional history,
builds causal features, and ranks the full executable Taiwan-stock universe.
Future horizons are research diagnostics only; this script defines no exit rule.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "history" / "2020-2025"
OUT = ROOT / "independent_buy_selector_v1_results"
OUT.mkdir(exist_ok=True)
HORIZONS = (5, 20, 60, 120)
PURGE = 120
TOPN = 10

FEATURES = [
    "ret5", "ret10", "ret20", "ret60", "ret120",
    "vol10", "vol20", "vol60",
    "gap_ma10", "gap_ma20", "gap_ma60", "gap_ma120",
    "dist_high20", "dist_high60", "dist_high120",
    "amt_ratio_5_20", "amt_ratio_20_60", "vol_ratio_5_20",
    "clv10", "range20", "foreign5_norm", "foreign20_norm",
    "trust5_norm", "trust20_norm",
    "cs_ret20", "cs_ret60", "cs_amount", "cs_foreign5",
    "mkt_median_ret20", "mkt_median_ret60", "breadth_ma60", "advance20",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    for y in range(2020, 2026):
        d = pd.read_parquet(DATA / f"ohlcv_{y}.parquet")
        d["code"] = d["code"].astype(str).str.zfill(4)
        if pd.api.types.is_datetime64_any_dtype(d["date"]):
            d["date"] = d["date"].dt.strftime("%Y%m%d").astype(int)
        else:
            d["date"] = d["date"].astype(str).str.replace("-", "", regex=False).astype(int)
        dfs.append(d)
    px = pd.concat(dfs, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    inst = pd.read_parquet(DATA / "institutional_2020_2025.parquet")
    inst["code"] = inst["code"].astype(str).str.zfill(4)
    if pd.api.types.is_datetime64_any_dtype(inst["date"]):
        inst["date"] = inst["date"].dt.strftime("%Y%m%d").astype(int)
    else:
        inst["date"] = pd.to_datetime(inst["date"]).dt.strftime("%Y%m%d").astype(int)
    return px, inst


def future_bad_jump(s: pd.Series, h: int = 120) -> pd.Series:
    # A >35% one-session raw-close discontinuity is treated as likely corporate action/data discontinuity.
    z = s.shift(-1).iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1]
    return z.fillna(False).astype(bool)


def build_dataset(px: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    x = px.copy()
    x["valid_code"] = x["code"].str.fullmatch(r"[1-9]\d{3}") & ~x["name"].astype(str).str.contains("KY", case=False, na=False)
    x["amt"] = x["close"] * x["volume"]
    g = x.groupby("code", group_keys=False)
    x["ret1"] = g["close"].pct_change()
    for h in (5, 10, 20, 60, 120):
        x[f"ret{h}"] = g["close"].pct_change(h)
    for h in (10, 20, 60):
        x[f"vol{h}"] = g["ret1"].transform(lambda s, h=h: s.rolling(h, min_periods=h).std())
    for h in (10, 20, 60, 120):
        x[f"ma{h}"] = g["close"].transform(lambda s, h=h: s.rolling(h, min_periods=h).mean())
        x[f"gap_ma{h}"] = x["close"] / x[f"ma{h}"] - 1.0
    for h in (20, 60, 120):
        hh = g["close"].transform(lambda s, h=h: s.rolling(h, min_periods=h).max())
        x[f"dist_high{h}"] = x["close"] / hh - 1.0
    x["amt5"] = g["amt"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["amt20"] = g["amt"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["amt60"] = g["amt"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    x["v5"] = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["v20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["amt_ratio_5_20"] = x["amt5"] / x["amt20"]
    x["amt_ratio_20_60"] = x["amt20"] / x["amt60"]
    x["vol_ratio_5_20"] = x["v5"] / x["v20"]
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    x["clv"] = ((2 * x["close"] - x["high"] - x["low"]) / rng).clip(-1, 1).fillna(0.0)
    x["clv10"] = g["clv"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    x["intrange"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["range20"] = g["intrange"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    x = x.merge(inst[["date", "code", "foreign_net", "trust_net"]], on=["date", "code"], how="left")
    x[["foreign_net", "trust_net"]] = x[["foreign_net", "trust_net"]].fillna(0.0)
    x = x.sort_values(["code", "date"]).reset_index(drop=True)
    g = x.groupby("code", group_keys=False)
    for h in (5, 20):
        x[f"foreign{h}"] = g["foreign_net"].transform(lambda s, h=h: s.rolling(h, min_periods=h).sum())
        x[f"trust{h}"] = g["trust_net"].transform(lambda s, h=h: s.rolling(h, min_periods=h).sum())
    denom = x["v20"].replace(0, np.nan)
    x["foreign5_norm"] = x["foreign5"] / denom
    x["foreign20_norm"] = x["foreign20"] / denom
    x["trust5_norm"] = x["trust5"] / denom
    x["trust20_norm"] = x["trust20"] / denom

    # Cross-sectional and market context features use only same-day information.
    x["cs_ret20"] = x.groupby("date")["ret20"].rank(pct=True)
    x["cs_ret60"] = x.groupby("date")["ret60"].rank(pct=True)
    x["cs_amount"] = x.groupby("date")["amt20"].rank(pct=True)
    x["cs_foreign5"] = x.groupby("date")["foreign5_norm"].rank(pct=True)
    daily = x.groupby("date").agg(
        mkt_median_ret20=("ret20", "median"),
        mkt_median_ret60=("ret60", "median"),
        breadth_ma60=("gap_ma60", lambda s: float((s > 0).mean())),
        advance20=("ret20", lambda s: float((s > 0).mean())),
    ).reset_index()
    x = x.merge(daily, on="date", how="left")

    # Future labels. Horizons are diagnostics, not sell dates.
    g = x.groupby("code", group_keys=False)
    for h in HORIZONS:
        x[f"fwd{h}"] = g["close"].shift(-h) / x["close"] - 1.0
    bad = x["ret1"].abs().gt(0.35)
    x["future_bad120"] = bad.groupby(x["code"], group_keys=False).transform(lambda s: future_bad_jump(s, 120))
    for h in HORIZONS:
        invalid = x["future_bad120"] | ~np.isfinite(x[f"fwd{h}"])
        x.loc[invalid, f"fwd{h}"] = np.nan
        x[f"rank_fwd{h}"] = x.groupby("date")[f"fwd{h}"].rank(pct=True)
    x["y_quality"] = x[[f"rank_fwd{h}" for h in HORIZONS]].mean(axis=1, skipna=False)

    dates = np.array(sorted(x["date"].unique()), dtype=np.int64)
    date_id = {int(d): i for i, d in enumerate(dates)}
    x["date_id"] = x["date"].map(date_id).astype(int)
    # Feasibility/data-quality only; not an alpha screen.
    x["universe_ok"] = x["valid_code"] & (x["amt20"] >= 30_000_000) & x["ma120"].notna()
    return x


def fit_model(train: pd.DataFrame):
    med = train[FEATURES].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    X = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y = train["y_quality"].astype(float)
    model = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=160, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=3.0, random_state=915,
    )
    model.fit(X, y)
    return model, med


def predict(model, med, frame: pd.DataFrame) -> np.ndarray:
    X = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return model.predict(X)


def main() -> None:
    px, inst = load_data()
    ds = build_dataset(px, inst)
    all_dates = np.array(sorted(ds.date.unique()), dtype=np.int64)
    date_to_i = {int(d): i for i, d in enumerate(all_dates)}
    picks, year_diag = [], []

    for year in range(2021, 2026):
        test = ds[(ds.date >= year * 10000 + 101) & (ds.date <= year * 10000 + 1231) & ds.universe_ok].copy()
        if test.empty:
            raise RuntimeError(f"no test rows for {year}")
        first_date = int(test.date.min())
        cutoff_i = date_to_i[first_date] - PURGE
        if cutoff_i <= 0:
            raise RuntimeError(f"insufficient purge history for {year}")
        cutoff = int(all_dates[cutoff_i])
        train = ds[(ds.date < cutoff) & ds.universe_ok & ds.y_quality.notna()].copy()
        train = train[(train.date_id % 5) == 0].copy()
        if len(train) < 2000:
            raise RuntimeError(f"insufficient training rows {year}: {len(train)}")
        model, med = fit_model(train)
        test["ai_score"] = predict(model, med, test)
        test["ai_rank"] = test.groupby("date")["ai_score"].rank(method="first", ascending=False)
        top = test[test.ai_rank <= TOPN].copy()
        top["test_year"] = year
        top["train_cutoff"] = cutoff
        picks.append(top)

        labeled = test[test.y_quality.notna()].copy()
        top_lab = top[top.y_quality.notna()].copy()
        ic = labeled.ai_score.corr(labeled.y_quality, method="spearman") if len(labeled) else np.nan
        row = {"test_year": year, "train_cutoff": cutoff, "train_rows": len(train),
               "test_rows": len(test), "rank_ic": ic, "selected_rows_labeled": len(top_lab)}
        for h in HORIZONS:
            market_med = labeled.groupby("date")[f"fwd{h}"].median()
            top_lab[f"alpha{h}"] = top_lab[f"fwd{h}"] - top_lab.date.map(market_med)
            row[f"top10_mean_fwd{h}"] = float(top_lab[f"fwd{h}"].mean())
            row[f"top10_mean_alpha{h}"] = float(top_lab[f"alpha{h}"].mean())
            row[f"top10_positive_rate{h}"] = float((top_lab[f"fwd{h}"] > 0).mean())
            row[f"top10_positive_alpha_rate{h}"] = float((top_lab[f"alpha{h}"] > 0).mean())
        year_diag.append(row)
        print(f"INDEPENDENT_BUY_SELECTOR fold={year} cutoff={cutoff} train={len(train)} test={len(test)} top={len(top)} ic={ic:.4f}", flush=True)

    sel = pd.concat(picks, ignore_index=True)
    diag = pd.DataFrame(year_diag)
    keep = ["date", "code", "name", "ai_score", "ai_rank", "test_year", "train_cutoff"] + FEATURES + [f"fwd{h}" for h in HORIZONS] + ["y_quality"]
    sel[keep].to_csv(OUT / "INDEPENDENT_BUY_SELECTOR_TOP10_OOS.csv", index=False)
    diag.to_csv(OUT / "INDEPENDENT_BUY_SELECTOR_YEARLY_DIAGNOSTICS.csv", index=False)

    # Overall diagnostics use same-day medians from the complete OOS universe, never R10.
    oos = ds[(ds.date >= 20210101) & (ds.date <= 20251231) & ds.universe_ok & ds.y_quality.notna()].copy()
    overall = {
        "selected_rows": int(len(sel)),
        "unique_selected_codes": int(sel.code.nunique()),
        "test_years": [2021, 2022, 2023, 2024, 2025],
        "median_yearly_rank_ic": float(diag.rank_ic.median()),
    }
    robust_years = {}
    for h in HORIZONS:
        med = oos.groupby("date")[f"fwd{h}"].median()
        z = sel[sel[f"fwd{h}"].notna()].copy()
        z[f"alpha{h}"] = z[f"fwd{h}"] - z.date.map(med)
        overall[f"top10_mean_fwd{h}"] = float(z[f"fwd{h}"].mean())
        overall[f"top10_mean_alpha{h}"] = float(z[f"alpha{h}"].mean())
        overall[f"top10_positive_rate{h}"] = float((z[f"fwd{h}"] > 0).mean())
        overall[f"top10_positive_alpha_rate{h}"] = float((z[f"alpha{h}"] > 0).mean())
        robust_years[str(h)] = int((diag[f"top10_mean_alpha{h}"] > 0).sum())
    overall["positive_alpha_years"] = robust_years
    (OUT / "INDEPENDENT_BUY_SELECTOR_OVERALL.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    gate = {
        "overall_positive_alpha_all_horizons": bool(all(overall[f"top10_mean_alpha{h}"] > 0 for h in HORIZONS)),
        "at_least_4_of_5_positive_alpha_years_20_60_120": bool(all(robust_years[str(h)] >= 4 for h in (20, 60, 120))),
        "median_yearly_rank_ic_positive": bool(overall["median_yearly_rank_ic"] > 0),
    }
    gate["promising_selector"] = all(gate.values())
    gate["note"] = "Research gate only. This is not portfolio-strategy success and defines no exit rule."
    (OUT / "INDEPENDENT_BUY_SELECTOR_RESEARCH_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("=== YEARLY OOS ===")
    print(diag.to_string(index=False))
    print("=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("=== RESEARCH GATE ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
