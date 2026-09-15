#!/usr/bin/env python3
"""Engineering repair for Independent Buy Selector V1.

Same preregistered features, target, model, TOP10 and OOS folds. The only
change is removal of the unnecessary every-fifth-date compute subsample so the
2021 fold can use all causally available 2020 training observations.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "independent_selector_core", ROOT / "scripts" / "backtest_independent_buy_selector_v1.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main() -> None:
    px, inst = m.load_data()
    ds = m.build_dataset(px, inst)
    all_dates = np.array(sorted(ds.date.unique()), dtype=np.int64)
    date_to_i = {int(d): i for i, d in enumerate(all_dates)}
    picks, year_diag = [], []

    for year in range(2021, 2026):
        test = ds[(ds.date >= year * 10000 + 101) & (ds.date <= year * 10000 + 1231) & ds.universe_ok].copy()
        if test.empty:
            raise RuntimeError(f"no test rows for {year}")
        first_date = int(test.date.min())
        cutoff_i = date_to_i[first_date] - m.PURGE
        if cutoff_i <= 0:
            raise RuntimeError(f"insufficient purge history for {year}")
        cutoff = int(all_dates[cutoff_i])
        train = ds[(ds.date < cutoff) & ds.universe_ok & ds.y_quality.notna()].copy()
        if len(train) < 500:
            raise RuntimeError(f"insufficient causal training rows {year}: {len(train)}")
        model, med = m.fit_model(train)
        test["ai_score"] = m.predict(model, med, test)
        test["ai_rank"] = test.groupby("date")["ai_score"].rank(method="first", ascending=False)
        top = test[test.ai_rank <= m.TOPN].copy()
        top["test_year"] = year
        top["train_cutoff"] = cutoff
        picks.append(top)

        labeled = test[test.y_quality.notna()].copy()
        top_lab = top[top.y_quality.notna()].copy()
        ic = labeled.ai_score.corr(labeled.y_quality, method="spearman") if len(labeled) else np.nan
        row = {
            "test_year": year,
            "train_cutoff": cutoff,
            "train_rows": len(train),
            "test_rows": len(test),
            "rank_ic": ic,
            "selected_rows_labeled": len(top_lab),
        }
        for h in m.HORIZONS:
            market_med = labeled.groupby("date")[f"fwd{h}"].median()
            top_lab[f"alpha{h}"] = top_lab[f"fwd{h}"] - top_lab.date.map(market_med)
            row[f"top10_mean_fwd{h}"] = float(top_lab[f"fwd{h}"].mean())
            row[f"top10_mean_alpha{h}"] = float(top_lab[f"alpha{h}"].mean())
            row[f"top10_positive_rate{h}"] = float((top_lab[f"fwd{h}"] > 0).mean())
            row[f"top10_positive_alpha_rate{h}"] = float((top_lab[f"alpha{h}"] > 0).mean())
        year_diag.append(row)
        print(
            f"INDEPENDENT_BUY_SELECTOR fold={year} cutoff={cutoff} "
            f"train={len(train)} test={len(test)} top={len(top)} ic={ic:.4f}",
            flush=True,
        )

    sel = pd.concat(picks, ignore_index=True)
    diag = pd.DataFrame(year_diag)
    keep = ["date", "code", "name", "ai_score", "ai_rank", "test_year", "train_cutoff"] + m.FEATURES + [f"fwd{h}" for h in m.HORIZONS] + ["y_quality"]
    sel[keep].to_csv(m.OUT / "INDEPENDENT_BUY_SELECTOR_TOP10_OOS.csv", index=False)
    diag.to_csv(m.OUT / "INDEPENDENT_BUY_SELECTOR_YEARLY_DIAGNOSTICS.csv", index=False)

    oos = ds[(ds.date >= 20210101) & (ds.date <= 20251231) & ds.universe_ok & ds.y_quality.notna()].copy()
    overall = {
        "selected_rows": int(len(sel)),
        "unique_selected_codes": int(sel.code.nunique()),
        "test_years": [2021, 2022, 2023, 2024, 2025],
        "median_yearly_rank_ic": float(diag.rank_ic.median()),
    }
    robust_years = {}
    for h in m.HORIZONS:
        med = oos.groupby("date")[f"fwd{h}"].median()
        z = sel[sel[f"fwd{h}"].notna()].copy()
        z[f"alpha{h}"] = z[f"fwd{h}"] - z.date.map(med)
        overall[f"top10_mean_fwd{h}"] = float(z[f"fwd{h}"].mean())
        overall[f"top10_mean_alpha{h}"] = float(z[f"alpha{h}"].mean())
        overall[f"top10_positive_rate{h}"] = float((z[f"fwd{h}"] > 0).mean())
        overall[f"top10_positive_alpha_rate{h}"] = float((z[f"alpha{h}"] > 0).mean())
        robust_years[str(h)] = int((diag[f"top10_mean_alpha{h}"] > 0).sum())
    overall["positive_alpha_years"] = robust_years
    (m.OUT / "INDEPENDENT_BUY_SELECTOR_OVERALL.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    gate = {
        "overall_positive_alpha_all_horizons": bool(all(overall[f"top10_mean_alpha{h}"] > 0 for h in m.HORIZONS)),
        "at_least_4_of_5_positive_alpha_years_20_60_120": bool(all(robust_years[str(h)] >= 4 for h in (20, 60, 120))),
        "median_yearly_rank_ic_positive": bool(overall["median_yearly_rank_ic"] > 0),
    }
    gate["promising_selector"] = all(gate.values())
    gate["note"] = "Research gate only. This is not portfolio-strategy success and defines no exit rule."
    (m.OUT / "INDEPENDENT_BUY_SELECTOR_RESEARCH_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("=== YEARLY OOS ===")
    print(diag.to_string(index=False))
    print("=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("=== RESEARCH GATE ===")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
