#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "independent_buy_selector_v3_results"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("v1core", ROOT / "scripts" / "backtest_independent_buy_selector_v1.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
PATH_H = (20, 60, 120)


def future_roll(s: pd.Series, h: int, kind: str) -> pd.Series:
    f = s.shift(-1).iloc[::-1]
    if kind == "max":
        return f.rolling(h, min_periods=h).max().iloc[::-1]
    return f.rolling(h, min_periods=h).min().iloc[::-1]


def add_path_labels(ds: pd.DataFrame) -> pd.DataFrame:
    x = ds.sort_values(["code", "date"]).copy()
    g = x.groupby("code", group_keys=False)
    for h in PATH_H:
        fmax = g["close"].transform(lambda s, h=h: future_roll(s, h, "max"))
        fmin = g["close"].transform(lambda s, h=h: future_roll(s, h, "min"))
        x[f"mfe{h}"] = fmax / x["close"] - 1.0
        x[f"mae{h}"] = fmin / x["close"] - 1.0
        invalid = x["future_bad120"] | x[f"fwd{h}"].isna()
        x.loc[invalid, [f"mfe{h}", f"mae{h}"]] = np.nan
        x[f"rank_end{h}"] = x.groupby("date")[f"fwd{h}"].rank(pct=True)
        x[f"rank_mfe{h}"] = x.groupby("date")[f"mfe{h}"].rank(pct=True)
        x[f"rank_resilience{h}"] = x.groupby("date")[f"mae{h}"].rank(pct=True)
    cols = []
    for h in PATH_H:
        cols += [f"rank_end{h}", f"rank_mfe{h}", f"rank_resilience{h}"]
    x["y_path_quality"] = x[cols].mean(axis=1, skipna=False)
    return x


def fit_model(train: pd.DataFrame):
    med = train[m.FEATURES].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    X = train[m.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    y = train["y_path_quality"].astype(float)
    model = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=160, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=3.0, random_state=917,
    )
    model.fit(X, y)
    return model, med


def main():
    px, inst = m.load_data()
    ds = add_path_labels(m.build_dataset(px, inst))
    dates = np.array(sorted(ds.date.unique()), dtype=np.int64)
    d2i = {int(d): i for i, d in enumerate(dates)}
    picks, rows = [], []

    for year in range(2021, 2026):
        test = ds[(ds.date >= year*10000+101) & (ds.date <= year*10000+1231) & ds.universe_ok].copy()
        if test.empty:
            raise RuntimeError(f"no test rows {year}")
        first = int(test.date.min())
        cutoff_i = d2i[first] - m.PURGE
        if cutoff_i <= 0:
            raise RuntimeError(f"insufficient purge history {year}")
        cutoff = int(dates[cutoff_i])
        train = ds[(ds.date < cutoff) & ds.universe_ok & ds.y_path_quality.notna()].copy()
        if len(train) < 500:
            raise RuntimeError(f"insufficient train {year}: {len(train)}")
        model, med = fit_model(train)
        X = test[m.FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0)
        test["ai_score"] = model.predict(X)
        test["ai_rank"] = test.groupby("date")["ai_score"].rank(method="first", ascending=False)
        top = test[test.ai_rank <= 10].copy(); top["test_year"] = year; top["train_cutoff"] = cutoff
        picks.append(top)

        labeled = top[top.y_path_quality.notna()].copy()
        universe = test[test.y_path_quality.notna()].copy()
        ic = universe.ai_score.corr(universe.y_path_quality, method="spearman") if len(universe) else np.nan
        row = {"test_year": year, "train_cutoff": cutoff, "train_rows": len(train),
               "selected_rows_labeled": len(labeled), "rank_ic": float(ic)}
        for h in m.HORIZONS:
            u = test[test[f"fwd{h}"].notna()].copy()
            medret = u.groupby("date")[f"fwd{h}"].median()
            z = top[top[f"fwd{h}"].notna()].copy(); z[f"alpha{h}"] = z[f"fwd{h}"] - z.date.map(medret)
            row[f"mean_fwd{h}"] = float(z[f"fwd{h}"].mean())
            row[f"mean_alpha{h}"] = float(z[f"alpha{h}"].mean())
        for h in PATH_H:
            z = labeled[labeled[f"mfe{h}"].notna() & labeled[f"mae{h}"].notna()]
            u = universe[universe[f"mfe{h}"].notna() & universe[f"mae{h}"].notna()]
            row[f"mean_mfe{h}"] = float(z[f"mfe{h}"].mean())
            row[f"universe_mean_mfe{h}"] = float(u[f"mfe{h}"].mean())
            row[f"mean_mae{h}"] = float(z[f"mae{h}"].mean())
            row[f"universe_mean_mae{h}"] = float(u[f"mae{h}"].mean())
        rows.append(row)
        print(f"V3 fold={year} train={len(train)} top={len(top)} ic={ic:.4f}", flush=True)

    sel = pd.concat(picks, ignore_index=True)
    diag = pd.DataFrame(rows)
    keep = ["date","code","name","ai_score","ai_rank","test_year","train_cutoff","y_path_quality"]
    keep += [f"fwd{h}" for h in m.HORIZONS]
    for h in PATH_H: keep += [f"mfe{h}", f"mae{h}"]
    sel[keep].to_csv(OUT/"INDEPENDENT_BUY_SELECTOR_V3_TOP10_OOS.csv", index=False)
    diag.to_csv(OUT/"INDEPENDENT_BUY_SELECTOR_V3_YEARLY.csv", index=False)

    overall = {"selected_rows": int(len(sel)), "unique_selected_codes": int(sel.code.nunique()),
               "median_yearly_rank_ic": float(diag.rank_ic.median())}
    positive_years = {}
    oos = ds[(ds.date >= 20210101) & (ds.date <= 20251231) & ds.universe_ok].copy()
    for h in m.HORIZONS:
        u = oos[oos[f"fwd{h}"].notna()].copy(); medret = u.groupby("date")[f"fwd{h}"].median()
        z = sel[sel[f"fwd{h}"].notna()].copy(); z[f"alpha{h}"] = z[f"fwd{h}"] - z.date.map(medret)
        overall[f"mean_fwd{h}"] = float(z[f"fwd{h}"].mean())
        overall[f"mean_alpha{h}"] = float(z[f"alpha{h}"].mean())
        positive_years[str(h)] = int((diag[f"mean_fwd{h}"] > 0).sum())
    for h in PATH_H:
        z = sel[sel[f"mfe{h}"].notna() & sel[f"mae{h}"].notna()]
        u = oos[oos[f"mfe{h}"].notna() & oos[f"mae{h}"].notna()]
        overall[f"mean_mfe{h}"] = float(z[f"mfe{h}"].mean())
        overall[f"universe_mean_mfe{h}"] = float(u[f"mfe{h}"].mean())
        overall[f"mean_mae{h}"] = float(z[f"mae{h}"].mean())
        overall[f"universe_mean_mae{h}"] = float(u[f"mae{h}"].mean())
    overall["positive_absolute_years"] = positive_years

    gate = {
        "absolute_positive_20_60_120": bool(all(overall[f"mean_fwd{h}"] > 0 for h in PATH_H)),
        "positive_alpha_all_horizons": bool(all(overall[f"mean_alpha{h}"] > 0 for h in m.HORIZONS)),
        "four_of_five_positive_years_20_60_120": bool(all(positive_years[str(h)] >= 4 for h in PATH_H)),
        "mfe_better_than_universe_all": bool(all(overall[f"mean_mfe{h}"] > overall[f"universe_mean_mfe{h}"] for h in PATH_H)),
        "mae_no_worse_than_universe_all": bool(all(overall[f"mean_mae{h}"] >= overall[f"universe_mean_mae{h}"] for h in PATH_H)),
        "median_rank_ic_positive": bool(overall["median_yearly_rank_ic"] > 0),
    }
    gate["promising_selector_v3"] = all(gate.values())
    gate["note"] = "Research gate only; path windows are diagnostics/labels, not holding or exit rules."
    (OUT/"INDEPENDENT_BUY_SELECTOR_V3_OVERALL.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    (OUT/"INDEPENDENT_BUY_SELECTOR_V3_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(diag.to_string(index=False)); print(json.dumps(overall, indent=2)); print(json.dumps(gate, indent=2))

if __name__ == "__main__":
    main()
