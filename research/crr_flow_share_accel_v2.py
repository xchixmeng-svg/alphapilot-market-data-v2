#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRR Flow-Share Acceleration v2 validation.

Primary hypothesis is preregistered in prereg_crr_flow_share_accel_v2.json.
Reuses v1 data loaders only; v2 features and decision rule are independent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from crr_flow_ratio_v1 import ROOT, load_prices, load_inst, fetch_industry_map, nw_t, period_name

OUT = ROOT / "research" / "out" / "crr-flow-share-accel-v2"
OUT.mkdir(parents=True, exist_ok=True)


def positive_pool_share(ind_daily: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    wide = ind_daily[col].unstack("industry").sort_index()
    if window == 1:
        net = wide
    else:
        net = wide.rolling(window, min_periods=window).sum()
    pos = net.clip(lower=0)
    denom = pos.sum(axis=1).replace(0, np.nan)
    return pos.div(denom, axis=0)


def summarise(panel: pd.DataFrame, feature: str, label: str) -> list[dict]:
    rows: list[dict] = []
    z = panel.dropna(subset=[feature]).copy()
    z["pct"] = z.groupby("date")[feature].rank(pct=True, method="average")
    z["quintile"] = np.select(
        [z.pct <= .2, z.pct <= .4, z.pct <= .6, z.pct <= .8],
        ["Q1", "Q2", "Q3", "Q4"], default="Q5"
    )

    for per in ["dev_2021_2024", "holdout_2025", "stress_2020"]:
        p = z[z["period"] == per]
        if p.empty:
            continue
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            g = p[p["quintile"] == q]
            rec = {
                "feature": label,
                "period": per,
                "bucket": q,
                "n": int(len(g)),
                "days": int(g.date.nunique()),
            }
            for k in (5, 10, 20):
                c = f"ex{k}"
                rec[f"ex{k}_pp"] = float(g[c].mean() * 100) if g[c].notna().any() else np.nan
                daily = g.groupby("date")[c].mean()
                rec[f"ex{k}_t"] = float(nw_t(daily.values, lag=k))
            rows.append(rec)

        for k in (5, 10, 20):
            vals = []
            for _, d in p[[feature, f"ex{k}", "date"]].dropna().groupby("date"):
                if len(d) >= 5 and d[feature].nunique() > 1 and d[f"ex{k}"].nunique() > 1:
                    vals.append(d[feature].rank().corr(d[f"ex{k}"].rank()))
            rows.append({
                "feature": label,
                "period": per,
                "bucket": f"SPEARMAN_{k}",
                "n": len(vals),
                "days": len(vals),
                f"ex{k}_pp": float(np.nanmean(vals) * 100) if vals else np.nan,
                f"ex{k}_t": float(nw_t(vals, lag=k)) if vals else np.nan,
            })
    return rows


def main():
    prereg = json.loads((ROOT / "research" / "prereg_crr_flow_share_accel_v2.json").read_text(encoding="utf-8"))
    print("[LOCKED]", prereg["version"], prereg["pass_bar"], flush=True)

    px = load_prices()
    ins = load_inst()
    print("[DATA] prices", len(px), "institutional", len(ins), flush=True)

    imap = fetch_industry_map()
    print("[INDUSTRY] mapped codes", len(imap), flush=True)

    d = px.merge(ins[["date", "code", "foreign_net", "trust_net"]], on=["date", "code"], how="left")
    d["industry"] = d["code"].map(imap)
    d["turnover"] = d["close"] * d["volume"]
    d["ft_amt"] = (d["foreign_net"] + d["trust_net"]) * d["close"]
    d = d.sort_values(["code", "date"]).reset_index(drop=True)

    # Causal T-close feature; forward outcome starts at T+1 open.
    g = d.groupby("code", sort=False)
    d["entry"] = g["open"].shift(-1)
    for k in (5, 10, 20):
        d[f"ret{k}"] = g["close"].shift(-k) / d["entry"] - 1

    d["avg_turn20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    valid = d["foreign_net"].notna() & d["trust_net"].notna() & d["avg_turn20"].notna() & d["industry"].notna()
    d["turn_rank"] = d["avg_turn20"].where(valid).groupby(d["date"]).rank(ascending=False, method="first")
    u = d[valid & (d["turn_rank"] <= 500)].copy()
    print("[UNIVERSE] rows", len(u), "days", u.date.nunique(), flush=True)

    # Median constituent outcome, identical convention to v1.
    market = u.groupby("date")[[f"ret{k}" for k in (5, 10, 20)]].median().rename(
        columns={f"ret{k}": f"mkt{k}" for k in (5, 10, 20)}
    )
    iout = u.groupby(["date", "industry"])[[f"ret{k}" for k in (5, 10, 20)]].median()
    icount = u.groupby(["date", "industry"])["code"].nunique().rename("constituents")
    iout = iout.join(icount).reset_index().merge(market.reset_index(), on="date", how="left")
    for k in (5, 10, 20):
        iout[f"ex{k}"] = iout[f"ret{k}"] - iout[f"mkt{k}"]

    daily = u.groupby(["date", "industry"], as_index=True).agg(
        ft_amt=("ft_amt", "sum"),
        turnover=("turnover", "sum"),
        members=("code", "nunique"),
    )

    share1 = positive_pool_share(daily, "ft_amt", 1)
    share5 = positive_pool_share(daily, "ft_amt", 5)
    share20 = positive_pool_share(daily, "ft_amt", 20)

    features = {
        "accel_5_20": (share5 - share20),
        "accel_1_5": (share1 - share5),
        "accel_1_20": (share1 - share20),
        "ratio_5_20": share5 / share20.replace(0, np.nan),
        "ratio_1_5": share1 / share5.replace(0, np.nan),
    }

    panel = iout.set_index(["date", "industry"])
    for name, wide in features.items():
        panel = panel.join(wide.stack().rename(name), how="left")
    panel = panel.reset_index()
    panel["period"] = panel["date"].map(period_name)
    panel = panel[panel["period"].notna() & (panel["constituents"] >= 3)].copy()

    tests = [
        ("accel_5_20", "PRIMARY_SHARE5_MINUS_SHARE20"),
        ("accel_1_5", "SECONDARY_SHARE1_MINUS_SHARE5"),
        ("accel_1_20", "SECONDARY_SHARE1_MINUS_SHARE20"),
        ("ratio_5_20", "SECONDARY_SHARE5_DIV_SHARE20"),
        ("ratio_1_5", "SECONDARY_SHARE1_DIV_SHARE5"),
    ]

    rows = []
    for col, label in tests:
        rows.extend(summarise(panel, col, label))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "flow_share_accel_results.csv", index=False, encoding="utf-8-sig")
    panel.to_parquet(OUT / "industry_day_panel.parquet", index=False)

    def primary_q5(period: str):
        q = res[(res.feature == "PRIMARY_SHARE5_MINUS_SHARE20") & (res.period == period) & (res.bucket == "Q5")]
        return q.iloc[0] if len(q) else None

    dev = primary_q5("dev_2021_2024")
    hold = primary_q5("holdout_2025")
    stress = primary_q5("stress_2020")
    passed = bool(
        dev is not None and hold is not None and stress is not None
        and dev.ex20_pp > 1.0 and dev.ex20_t >= 2.0
        and hold.ex20_pp > 0 and stress.ex20_pp > 0
    )

    # For interpretation only: Q1-Q5 spread and complete primary quintiles.
    primary_rows = res[(res.feature == "PRIMARY_SHARE5_MINUS_SHARE20") & res.bucket.isin(["Q1", "Q2", "Q3", "Q4", "Q5"])]
    primary_quintiles = {}
    for per in ["dev_2021_2024", "holdout_2025", "stress_2020"]:
        pp = primary_rows[primary_rows.period == per]
        primary_quintiles[per] = {
            r.bucket: {"ex5_pp": round(float(r.ex5_pp), 3), "ex10_pp": round(float(r.ex10_pp), 3), "ex20_pp": round(float(r.ex20_pp), 3)}
            for _, r in pp.iterrows()
        }

    summary = {
        "study": prereg["version"],
        "pass": passed,
        "primary": {
            "dev_2021_2024": None if dev is None else {"ex20_pp": round(float(dev.ex20_pp), 3), "t": round(float(dev.ex20_t), 3), "n": int(dev.n), "days": int(dev.days)},
            "holdout_2025": None if hold is None else {"ex20_pp": round(float(hold.ex20_pp), 3), "t": round(float(hold.ex20_t), 3), "n": int(hold.n), "days": int(hold.days)},
            "stress_2020": None if stress is None else {"ex20_pp": round(float(stress.ex20_pp), 3), "t": round(float(stress.ex20_t), 3), "n": int(stress.n), "days": int(stress.days)},
        },
        "primary_quintiles": primary_quintiles,
        "data": {
            "price_rows": int(len(px)),
            "inst_rows": int(len(ins)),
            "universe_rows": int(len(u)),
            "industry_map_codes": int(len(imap)),
            "panel_rows": int(len(panel)),
        },
        "notes": [
            "Primary feature is share5-share20, locked before v2 results.",
            "Secondary features are descriptive and cannot rescue a primary FAIL.",
            "Current industry classification snapshot remains the same known limitation as v1."
        ]
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# CRR Flow-Share Acceleration v2",
        "",
        f"**判定：{'PASS' if passed else 'FAIL'}**",
        "",
        "Primary = 5日正向法人資金池占比 - 20日正向法人資金池占比",
        "",
        "| Period | Q5 ex20 (pp) | NW t | n | days |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in [("2021-2024 dev", dev), ("2025 holdout", hold), ("2020 stress", stress)]:
        if row is not None:
            lines.append(f"| {name} | {row.ex20_pp:.3f} | {row.ex20_t:.2f} | {int(row.n)} | {int(row.days)} |")
    lines += [
        "",
        "PASS bar: dev Q5 ex20 > +1.0pp and NW t >= 2, plus positive Q5 ex20 in 2025 holdout and 2020 stress.",
        "",
        "Secondary tests are reported in flow_share_accel_results.csv and cannot rescue a primary FAIL.",
    ]
    (OUT / "CONCLUSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
