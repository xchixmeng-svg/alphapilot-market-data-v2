#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_ai_market_reasoning_v6_cpu_quantile as core  # noqa: E402

OUT = ROOT / "ai_market_reasoning_v6_actionability"
OUT.mkdir(exist_ok=True)

YEARS = (2022, 2023, 2024)
HORIZONS = core.AUDIT_HORIZONS
TOPKS = (1, 3, 5)
THRESHOLDS = (0.05, 0.10, 0.20)


def realized_path_metrics(ds: pd.DataFrame, picks: pd.DataFrame, h: int) -> pd.DataFrame:
    by_code = {}
    for code, g in ds.groupby("code", sort=False):
        gg = g.sort_values("date")
        by_code[str(code)] = (
            gg["date"].to_numpy(dtype=np.int64),
            gg["close"].to_numpy(dtype=float),
        )

    rows = []
    for r in picks.itertuples(index=False):
        dates, closes = by_code[str(r.code)]
        pos = int(np.searchsorted(dates, int(r.date)))
        if pos >= len(dates) or int(dates[pos]) != int(r.date):
            continue
        fut = closes[pos + 1 : min(pos + 1 + h, len(closes))]
        if len(fut) == 0 or not np.isfinite(float(r.close)) or float(r.close) <= 0:
            continue
        rets = fut / float(r.close) - 1.0
        out = {
            "date": int(r.date),
            "code": str(r.code).zfill(4),
            "rank_q50": int(r.rank_q50),
            "horizon": int(h),
            "entry_close": float(r.close),
            "pred_q10": float(r.pred_q10),
            "pred_q25": float(r.pred_q25),
            "pred_q50": float(r.pred_q50),
            "pred_q75": float(r.pred_q75),
            "pred_q90": float(r.pred_q90),
            "actual_end_return": float(r.actual_end_return),
            "max_return_within_h": float(np.nanmax(rets)),
            "min_return_within_h": float(np.nanmin(rets)),
            "q50_abs_error": float(abs(float(r.actual_end_return) - float(r.pred_q50))),
        }
        for t in THRESHOLDS:
            idx = np.flatnonzero(rets >= t)
            tag = int(round(t * 100))
            out[f"hit_{tag}pct"] = bool(len(idx))
            out[f"sessions_to_{tag}pct"] = int(idx[0] + 1) if len(idx) else None
        rows.append(out)
    return pd.DataFrame(rows)


def run_year(year: int) -> None:
    if year not in YEARS:
        raise RuntimeError(f"actionability audit limited to completed locked Stage-A years {YEARS}")

    panel, base_feats, _ = core.load_frozen_panel()
    ds, feats = core.add_frozen_observable_transforms(panel, base_feats)
    dates = np.array(sorted(ds["date"].unique()), dtype=np.int64)
    test = ds[
        (ds["date"] >= year * 10000 + 101)
        & (ds["date"] <= year * 10000 + 1231)
        & ds["universe_ok"]
    ].copy()
    first = int(test["date"].min())
    pos = int(np.searchsorted(dates, first))
    cutoff = int(dates[pos - core.PURGE])
    train = ds[(ds["date"] < cutoff) & ds["universe_ok"]].copy()
    models, nfit = core.fit_models(train, feats, core.RNG_SEED + year)

    all_picks = []
    for h in HORIZONS:
        actual = core.fwd_return(test, h)
        valid = np.isfinite(actual.to_numpy(dtype=float))
        eval_df = test.loc[valid, ["date", "code", "close"]].copy()
        y = actual.to_numpy(dtype=float)[valid]
        p = core.predict_quantiles(models, test.loc[valid], feats, h)
        eval_df["pred_q10"] = p[:, 0]
        eval_df["pred_q25"] = p[:, 1]
        eval_df["pred_q50"] = p[:, 2]
        eval_df["pred_q75"] = p[:, 3]
        eval_df["pred_q90"] = p[:, 4]
        eval_df["actual_end_return"] = y
        eval_df["rank_q50"] = eval_df.groupby("date")["pred_q50"].rank(
            method="first", ascending=False
        ).astype(int)
        picks = eval_df[eval_df["rank_q50"] <= max(TOPKS)].copy()
        detailed = realized_path_metrics(ds, picks, h)
        all_picks.append(detailed)
        print(json.dumps({
            "year": year, "horizon": h, "dates": int(detailed["date"].nunique()),
            "top5_rows": int(len(detailed))
        }), flush=True)

    detail = pd.concat(all_picks, ignore_index=True)
    detail.to_csv(OUT / f"V6_ACTIONABILITY_{year}_TOP5_DETAIL.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for h in HORIZONS:
        hdf = detail[detail["horizon"] == h]
        for k in TOPKS:
            x = hdf[hdf["rank_q50"] <= k]
            row = {
                "year": year,
                "horizon": h,
                "top_k": k,
                "dates": int(x["date"].nunique()),
                "picks": int(len(x)),
                "positive_end_rate": float((x["actual_end_return"] > 0).mean()),
                "mean_end_return": float(x["actual_end_return"].mean()),
                "median_end_return": float(x["actual_end_return"].median()),
                "mean_max_return": float(x["max_return_within_h"].mean()),
                "median_max_return": float(x["max_return_within_h"].median()),
                "mean_min_return": float(x["min_return_within_h"].mean()),
                "median_min_return": float(x["min_return_within_h"].median()),
                "median_q50_abs_error": float(x["q50_abs_error"].median()),
            }
            for t in THRESHOLDS:
                tag = int(round(t * 100))
                row[f"hit_{tag}pct_rate"] = float(x[f"hit_{tag}pct"].mean())
                hit = x.loc[x[f"hit_{tag}pct"], f"sessions_to_{tag}pct"].dropna()
                row[f"median_sessions_to_{tag}pct_when_hit"] = (
                    float(hit.median()) if len(hit) else None
                )
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / f"V6_ACTIONABILITY_{year}_SUMMARY.csv", index=False, encoding="utf-8-sig")

    meta = {
        "status": "DIAGNOSTIC_ONLY_NOT_A_NEW_GATE",
        "year": year,
        "source": "locked V6 CPU quantile model and exact frozen Stage0F panel",
        "ranking": "daily descending predicted q50; report Top1/Top3/Top5 without tuning",
        "horizons": list(HORIZONS),
        "thresholds": list(THRESHOLDS),
        "fit_long_rows": int(nfit),
        "feature_count": int(len(feats)),
        "note": "No model parameter, hypothesis, gate, threshold, or 2025 data is changed/opened by this audit."
    }
    (OUT / f"V6_ACTIONABILITY_{year}_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ns = ap.parse_args()
    run_year(ns.year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
