#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRR Flow-Ratio v1 historical validation.

Pre-registered in research/prereg_crr_flow_ratio_v1.json.
Reads only the immutable 2020-2025 history archive. Does not modify source data.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "history" / "2020-2025"
OUT = ROOT / "research" / "out" / "crr-flow-ratio-v1"
OUT.mkdir(parents=True, exist_ok=True)
CODE_RE = re.compile(r"^[1-9]\d{3}$")


def norm_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)


def norm_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y%m%d").astype(int)
    x = s.astype(str).str.replace(r"[^0-9]", "", regex=True)
    # ROC yyyMMdd -> Gregorian yyyyMMdd when needed
    def one(v: str):
        if len(v) == 8:
            return int(v)
        if len(v) == 7:
            return int(str(int(v[:3]) + 1911) + v[3:])
        return np.nan
    return x.map(one)


def pick_col(df: pd.DataFrame, names: list[str], required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    if required:
        raise KeyError(f"missing columns {names}; available={list(df.columns)}")
    return None


def load_prices() -> pd.DataFrame:
    parts = []
    for y in range(2020, 2026):
        p = HIST / f"ohlcv_{y}.parquet"
        x = pd.read_parquet(p)
        dc = pick_col(x, ["date", "trade_date"])
        cc = pick_col(x, ["code", "stock_id"])
        oc = pick_col(x, ["open"])
        hc = pick_col(x, ["high"])
        lc = pick_col(x, ["low"])
        cl = pick_col(x, ["close"])
        vc = pick_col(x, ["volume", "trade_volume"])
        x = x[[dc, cc, oc, hc, lc, cl, vc]].copy()
        x.columns = ["date", "code", "open", "high", "low", "close", "volume"]
        x["date"] = norm_date(x["date"])
        x["code"] = norm_code(x["code"])
        for c in ["open", "high", "low", "close", "volume"]:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x[x["code"].map(lambda z: bool(CODE_RE.match(z)))]
        parts.append(x)
    px = pd.concat(parts, ignore_index=True)
    px = px.dropna(subset=["date", "code", "open", "close", "volume"])
    px["date"] = px["date"].astype(int)
    return px.drop_duplicates(["date", "code"], keep="last")


def load_inst() -> pd.DataFrame:
    x = pd.read_parquet(HIST / "institutional_2020_2025.parquet")
    dc = pick_col(x, ["date", "trade_date"])
    cc = pick_col(x, ["code", "stock_id"])
    fc = pick_col(x, ["foreign_net", "foreign_investor_net_buy_sell", "foreign_net_shares"])
    tc = pick_col(x, ["trust_net", "investment_trust_net_buy_sell", "trust_net_shares"])
    dc2 = pick_col(x, ["dealer_net", "dealer_self_net_buy_sell", "dealer_net_shares"], required=False)
    keep = [dc, cc, fc, tc] + ([dc2] if dc2 else [])
    x = x[keep].copy()
    names = ["date", "code", "foreign_net", "trust_net"] + (["dealer_net"] if dc2 else [])
    x.columns = names
    x["date"] = norm_date(x["date"])
    x["code"] = norm_code(x["code"])
    for c in ["foreign_net", "trust_net"] + (["dealer_net"] if "dealer_net" in x else []):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[x["code"].map(lambda z: bool(CODE_RE.match(z)))]
    x["date"] = x["date"].astype("Int64")
    x = x.dropna(subset=["date", "code", "foreign_net", "trust_net"])
    x["date"] = x["date"].astype(int)
    return x.drop_duplicates(["date", "code"], keep="last")


def fetch_industry_map() -> dict[str, str]:
    """Current snapshot only, matching the known limitation of CRR v0."""
    out: dict[str, str] = {}
    headers = {"User-Agent": "Mozilla/5.0 AlphaPilot CRR Flow Ratio Study"}
    for mode in (2, 4):
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "big5"
            soup = BeautifulSoup(r.text, "html.parser")
            for tr in soup.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(tds) < 5:
                    continue
                m = re.match(r"^(\d{4})\s+", tds[0])
                if not m:
                    continue
                code = m.group(1)
                industry = tds[4].strip()
                if industry and industry not in {"-", "--"}:
                    out[code] = industry
        except Exception as e:
            print(f"[WARN] ISIN mode={mode} failed: {e}", flush=True)
    if len(out) < 300:
        # Fallback to FinMind stock info; token is not required for this endpoint in normal use.
        try:
            r = requests.get(
                "https://api.finmindtrade.com/api/v4/data",
                params={"dataset": "TaiwanStockInfo"},
                headers=headers,
                timeout=60,
            )
            j = r.json()
            for row in j.get("data", []):
                code = str(row.get("stock_id", "")).strip()
                ind = str(row.get("industry_category", "")).strip()
                if CODE_RE.match(code) and ind:
                    out[code] = ind
        except Exception as e:
            print(f"[WARN] FinMind industry fallback failed: {e}", flush=True)
    if len(out) < 200:
        raise RuntimeError(f"industry map too small: {len(out)}")
    return out


def nw_t(x, lag=20):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean()
    e = x - mu
    g0 = (e * e).sum() / n
    s = g0
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).sum() / n
    return mu / math.sqrt(s / n) if s > 0 else np.nan


def flow_feature(ind_daily: pd.DataFrame, col: str, kind="in") -> pd.DataFrame:
    # input index date x industry; values daily institutional amount
    net5 = ind_daily[col].unstack("industry").sort_index().rolling(5, min_periods=5).sum()
    turn5 = ind_daily["turnover"].unstack("industry").sort_index().rolling(5, min_periods=5).sum()
    turn_share = turn5.div(turn5.sum(axis=1), axis=0)
    if kind == "in":
        pos = net5.clip(lower=0)
        pool = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0)
        over = pool / turn_share
        base = over.shift(1).rolling(60, min_periods=20).mean()
        ratio = over / base.replace(0, np.nan)
        return pd.concat({"net5": net5, "turn_share": turn_share, "pool_share": pool,
                          "overalloc": over, "flow_ratio60": ratio}, axis=1)
    neg = (-net5.clip(upper=0))
    pool = neg.div(neg.sum(axis=1).replace(0, np.nan), axis=0)
    over = pool / turn_share
    return pd.concat({"net5": net5, "turn_share": turn_share, "pool_share": pool,
                      "overalloc": over}, axis=1)


def period_name(date: int):
    if 20210101 <= date <= 20241231:
        return "dev_2021_2024"
    if 20250101 <= date <= 20251231:
        return "holdout_2025"
    if 20200101 <= date <= 20201231:
        return "stress_2020"
    return None


def summarise(panel: pd.DataFrame, feature: str, label: str) -> list[dict]:
    rows = []
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
            rec = {"feature": label, "period": per, "bucket": q,
                   "n": int(len(g)), "days": int(g.date.nunique())}
            for k in (5, 10, 20):
                c = f"ex{k}"
                rec[f"ex{k}_pp"] = float(g[c].mean() * 100) if g[c].notna().any() else np.nan
                daily = g.groupby("date")[c].mean()
                rec[f"ex{k}_t"] = float(nw_t(daily.values, lag=k))
            rows.append(rec)
        # continuous daily cross-sectional Spearman
        for k in (5, 10, 20):
            vals = []
            for _, d in p[[feature, f"ex{k}", "date"]].dropna().groupby("date"):
                if len(d) >= 5 and d[feature].nunique() > 1 and d[f"ex{k}"].nunique() > 1:
                    vals.append(d[feature].rank().corr(d[f"ex{k}"].rank()))
            rows.append({"feature": label, "period": per, "bucket": "SPEARMAN",
                         "n": len(vals), "days": len(vals), f"ex{k}_pp": float(np.nanmean(vals) * 100) if vals else np.nan,
                         f"ex{k}_t": float(nw_t(vals, lag=k)) if vals else np.nan})
    return rows


def main():
    prereg = json.loads((ROOT / "research" / "prereg_crr_flow_ratio_v1.json").read_text(encoding="utf-8"))
    print("[LOCKED]", prereg["version"], prereg["pass_bar"], flush=True)
    px = load_prices()
    ins = load_inst()
    print("[DATA] prices", len(px), "institutional", len(ins), flush=True)

    imap = fetch_industry_map()
    print("[INDUSTRY] mapped codes", len(imap), flush=True)

    d = px.merge(ins[["date", "code", "foreign_net", "trust_net"]], on=["date", "code"], how="left")
    d["industry"] = d["code"].map(imap)
    d["turnover"] = d["close"] * d["volume"]
    d["foreign_amt"] = d["foreign_net"] * d["close"]
    d["trust_amt"] = d["trust_net"] * d["close"]
    d["ft_amt"] = d["foreign_amt"] + d["trust_amt"]
    d = d.sort_values(["code", "date"]).reset_index(drop=True)

    # T+1 open entry / T+k close outcome at stock level.
    g = d.groupby("code", sort=False)
    d["entry"] = g["open"].shift(-1)
    for k in (5, 10, 20):
        d[f"ret{k}"] = g["close"].shift(-k) / d["entry"] - 1
        # impossible corporate-action-like jumps are not dropped here; industry median outcome limits their influence.

    d["avg_turn20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    valid = d["foreign_net"].notna() & d["trust_net"].notna() & d["avg_turn20"].notna() & d["industry"].notna()
    d["turn_rank"] = d["avg_turn20"].where(valid).groupby(d["date"]).rank(ascending=False, method="first")
    u = d[valid & (d["turn_rank"] <= 500)].copy()
    print("[UNIVERSE] rows", len(u), "days", u.date.nunique(), "mapped share", round(len(u)/max(1, valid.sum()),4), flush=True)

    # Market and industry outcome: median constituent return, to make stock splits/dividend discontinuities non-dominant.
    market = u.groupby("date")[[f"ret{k}" for k in (5,10,20)]].median().rename(columns={f"ret{k}": f"mkt{k}" for k in (5,10,20)})
    iout = u.groupby(["date", "industry"])[[f"ret{k}" for k in (5,10,20)]].median()
    icount = u.groupby(["date", "industry"])["code"].nunique().rename("constituents")
    iout = iout.join(icount).reset_index().merge(market.reset_index(), on="date", how="left")
    for k in (5,10,20):
        iout[f"ex{k}"] = iout[f"ret{k}"] - iout[f"mkt{k}"]

    daily = u.groupby(["date", "industry"], as_index=True).agg(
        turnover=("turnover", "sum"), ft_amt=("ft_amt", "sum"), foreign_amt=("foreign_amt", "sum"), trust_amt=("trust_amt", "sum"), members=("code", "nunique")
    )

    feats = {}
    for c in ["ft_amt", "foreign_amt", "trust_amt"]:
        f = flow_feature(daily, c, "in")
        for field in ["overalloc", "flow_ratio60"]:
            feats[f"{c}_{field}"] = f[field].stack(dropna=False).rename(f"{c}_{field}")
    outf = flow_feature(daily, "ft_amt", "out")
    feats["ft_outflow_overalloc"] = outf["overalloc"].stack(dropna=False).rename("ft_outflow_overalloc")

    panel = iout.set_index(["date", "industry"])
    for s in feats.values():
        panel = panel.join(s, how="left")
    panel = panel.reset_index()
    panel["period"] = panel["date"].map(period_name)
    panel = panel[panel["period"].notna()].copy()
    # Require at least 3 current-universe constituents for an industry outcome.
    panel = panel[panel["constituents"] >= 3]

    tests = [
        ("ft_amt_flow_ratio60", "PRIMARY_FT_FLOW_RATIO60"),
        ("ft_amt_overalloc", "SECONDARY_FT_OVERALLOC5"),
        ("foreign_amt_flow_ratio60", "SECONDARY_FOREIGN_FLOW_RATIO60"),
        ("trust_amt_flow_ratio60", "SECONDARY_TRUST_FLOW_RATIO60"),
        ("ft_outflow_overalloc", "SECONDARY_FT_OUTFLOW_OVERALLOC5"),
    ]
    rows = []
    for c, lab in tests:
        rows.extend(summarise(panel, c, lab))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "flow_ratio_results.csv", index=False, encoding="utf-8-sig")
    panel.to_parquet(OUT / "industry_day_panel.parquet", index=False)

    def q5(period):
        q = res[(res.feature == "PRIMARY_FT_FLOW_RATIO60") & (res.period == period) & (res.bucket == "Q5")]
        return q.iloc[0] if len(q) else None
    dev, hold, stress = q5("dev_2021_2024"), q5("holdout_2025"), q5("stress_2020")
    passed = bool(dev is not None and hold is not None and stress is not None and
                  dev.ex20_pp > 1.0 and dev.ex20_t >= 2.0 and hold.ex20_pp > 0 and stress.ex20_pp > 0)

    summary = {
        "study": prereg["version"], "pass": passed,
        "primary": {
            "dev_2021_2024": None if dev is None else {"ex20_pp": round(float(dev.ex20_pp),3), "t": round(float(dev.ex20_t),3), "n": int(dev.n), "days": int(dev.days)},
            "holdout_2025": None if hold is None else {"ex20_pp": round(float(hold.ex20_pp),3), "t": round(float(hold.ex20_t),3), "n": int(hold.n), "days": int(hold.days)},
            "stress_2020": None if stress is None else {"ex20_pp": round(float(stress.ex20_pp),3), "t": round(float(stress.ex20_t),3), "n": int(stress.n), "days": int(stress.days)},
        },
        "data": {"price_rows": int(len(px)), "inst_rows": int(len(ins)), "universe_rows": int(len(u)), "industry_map_codes": int(len(imap)), "panel_rows": int(len(panel))},
        "notes": ["Current industry classification snapshot is used, same known limitation as CRR v0.", "Primary PASS is determined only by preregistered FT flow_ratio60 Q5 rule."]
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# CRR Flow-Ratio v1 驗證結果", "", f"**判定：{'PASS' if passed else 'FAIL'}**", "", "## Primary: FT 資金量比 Q5", "",
             "| Period | ex20 (pp) | NW t | n | days |", "|---|---:|---:|---:|---:|"]
    for name, row in [("2021-2024 dev", dev), ("2025 holdout", hold), ("2020 stress", stress)]:
        if row is not None:
            lines.append(f"| {name} | {row.ex20_pp:.3f} | {row.ex20_t:.2f} | {int(row.n)} | {int(row.days)} |")
    lines += ["", "Pass bar: dev ex20 > +1.0pp and NW t>=2, with positive ex20 in both 2025 holdout and 2020 stress.", "", "## Quintiles / secondary tests", "", "See `flow_ratio_results.csv`. No secondary result can rescue a primary FAIL."]
    (OUT / "CONCLUSION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
