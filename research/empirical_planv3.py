#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaPilot empirical probability research for plan-v3.

Research-only. It does NOT modify AlphaPilot Forward, R10-MAX, or main data files.

Target:
- decision at T close
- T+1 open entry; cancel no-data / gap >= 1*ATR20 / locked limit-up
- no intraday exit on T+1
- days 2..30: dynamic stop (1.5*ATR20/T close, clipped 5..8%) has priority over +15%
- if +5% never reached by day 10 -> failure
- if +15% not reached by day 30 -> failure

Validation:
- 2020-2023 base fit
- 2024 Platt calibration
- 2025 untouched validation
- final 2026 shadow scorer: 2020-2024 base fit, 2025 calibration
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

HOLD = 30
TARGET = 0.15
EARLY = 0.05
EARLY_DAYS = 10
STOP_ATR = 1.5
STOP_MIN = 0.05
STOP_MAX = 0.08
GAP_ATR = 1.0
LIMIT_UP = 1.095
UNIV_TOP = 500
UNIV_MIN_AMOUNT20 = 30_000_000.0

FEATURES = [
    "r5",
    "r20",
    "atr_pct",
    "amount_ratio",
    "ma20_gap",
    "ma60_gap",
    "inst5_ratio_pct",
    "cost_dist_atr",
]

ALIASES = {
    "date": ["date", "trade_date", "日期", "資料日期", "datetime"],
    "code": ["code", "stock_id", "symbol", "ticker", "sid", "證券代號", "股票代號", "代號", "公司代號", "stock_no", "stockid"],
    "open": ["open", "開盤價", "open_price"],
    "high": ["high", "最高價", "high_price", "max"],
    "low": ["low", "最低價", "low_price", "min"],
    "close": ["close", "收盤價", "close_price"],
    "volume": ["volume", "成交股數", "trading_volume", "vol"],
    "amount": ["amount", "trading_value", "成交金額", "turnover", "trading_money", "value"],
}


def norm(s):
    return str(s).strip().lower().replace(" ", "")


def pick_col(cols, names):
    lookup = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in lookup:
            return lookup[norm(n)]
    return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for k, names in ALIASES.items():
        c = pick_col(df.columns, names)
        if c is not None:
            ren[c] = k
    df = df.rename(columns=ren).copy()
    need = {"date", "code", "open", "high", "low", "close", "volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"OHLCV missing columns: {sorted(miss)}")
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace(r"\.0$", "", regex=True), errors="coerce")
    df["code"] = (
        df["code"].astype(str).str.strip()
        .str.replace(r"\.(TW|TWO)$", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )
    for c in ["open", "high", "low", "close", "volume"] + (["amount"] if "amount" in df.columns else []):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    return df.dropna(subset=["date", "code", "open", "high", "low", "close", "volume"])


def institutional_cols(cols) -> Dict[str, Tuple[str, ...]]:
    low = {c: norm(c) for c in cols}
    cats = {
        "foreign": (["foreign", "外資", "外陸資"], ["自營", "dealer"]),
        "trust": (["trust", "投信"], []),
        "dealer": (["dealer", "自營"], ["foreign", "外資", "外陸資"]),
    }
    out = {}
    for cat, (keys, excl) in cats.items():
        cand = [c for c in cols if any(k in low[c] for k in keys) and not any(x in low[c] for x in excl)]
        nets = [c for c in cand if any(x in low[c] for x in ["net", "買賣超", "diff", "buysell", "買賣"])]
        if nets:
            out[cat] = ("net", sorted(nets, key=lambda x: len(str(x)))[0])
            continue
        buys = [c for c in cand if any(x in low[c] for x in ["buy", "買進"])]
        sells = [c for c in cand if any(x in low[c] for x in ["sell", "賣出"])]
        if buys and sells:
            out[cat] = ("bs", sorted(buys, key=lambda x: len(str(x)))[0], sorted(sells, key=lambda x: len(str(x)))[0])
    return out


def normalize_inst(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for k in ["date", "code"]:
        c = pick_col(df.columns, ALIASES[k])
        if c is not None:
            ren[c] = k
    df = df.rename(columns=ren).copy()
    if not {"date", "code"} <= set(df.columns):
        raise ValueError("institutional file missing date/code")
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace(r"\.0$", "", regex=True), errors="coerce")
    df["code"] = (
        df["code"].astype(str).str.strip()
        .str.replace(r"\.(TW|TWO)$", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )
    spec = institutional_cols(df.columns)
    out = df[["date", "code"]].copy()
    for cat in ["foreign", "trust", "dealer"]:
        sp = spec.get(cat)
        if sp is None:
            out[cat] = np.nan
        elif sp[0] == "net":
            out[cat] = pd.to_numeric(df[sp[1]].astype(str).str.replace(",", "", regex=False), errors="coerce")
        else:
            b = pd.to_numeric(df[sp[1]].astype(str).str.replace(",", "", regex=False), errors="coerce")
            s = pd.to_numeric(df[sp[2]].astype(str).str.replace(",", "", regex=False), errors="coerce")
            out[cat] = b - s
    return (
        out.dropna(subset=["date", "code"])
        .groupby(["date", "code"], as_index=False)
        .agg(foreign=("foreign", "sum"), trust=("trust", "sum"), dealer=("dealer", "sum"))
    )


def load_history(root: Path):
    h = root / "data" / "history" / "2020-2025"
    pxs = []
    hashes = {}
    for y in range(2020, 2026):
        p = h / f"ohlcv_{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        pxs.append(normalize_ohlcv(pd.read_parquet(p)))
    ip = h / "institutional_2020_2025.parquet"
    if not ip.exists():
        raise FileNotFoundError(ip)
    hashes[ip.name] = hashlib.sha256(ip.read_bytes()).hexdigest()
    inst = normalize_inst(pd.read_parquet(ip))
    px = pd.concat(pxs, ignore_index=True)
    px = px[px["code"].str.fullmatch(r"[1-9]\d{3}|0050", na=False)]
    px = px.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")
    return px, inst, hashes


def make_matrices(px: pd.DataFrame, inst: pd.DataFrame):
    dates = np.array(sorted(px["date"].unique()), dtype="datetime64[ns]")
    ids = np.array(sorted(px["code"].unique()))

    def wide(df, col):
        return (
            df.pivot(index="date", columns="code", values=col)
            .reindex(index=dates, columns=ids).to_numpy(float)
        )

    O, H, L, C, V, A = (wide(px, c) for c in ["open", "high", "low", "close", "volume", "amount"])
    medv = np.nanmedian(V)
    if np.isfinite(medv) and medv < 20_000:
        V = V * 1000.0

    inst = inst[inst["code"].isin(ids)]

    def iwide(col):
        return (
            inst.pivot(index="date", columns="code", values=col)
            .reindex(index=dates, columns=ids).to_numpy(float)
        )

    IF, IT, ID = (iwide(c) for c in ["foreign", "trust", "dealer"])
    medi = np.nanmedian(np.abs(IF))
    if np.isfinite(medi) and medi < 50 and medv >= 20_000:
        IF, IT, ID = IF * 1000.0, IT * 1000.0, ID * 1000.0
    return dates, ids, O, H, L, C, V, A, IF, IT, ID


def shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=float)
    if k < a.shape[0]:
        out[:-k] = a[k:]
    return out


def build_dataset(dates, ids, O, H, L, C, V, A, IF, IT, ID):
    D = lambda x: pd.DataFrame(x)
    cdf, vdf, adf = D(C), D(V), D(A)
    prev = cdf.shift(1).to_numpy()
    tr = np.fmax(H, prev) - np.fmin(L, prev)
    tr = np.where(np.isnan(prev), H - L, tr)

    atr = D(tr).rolling(20, min_periods=15).mean().to_numpy()
    ma20 = cdf.rolling(20, min_periods=15).mean().to_numpy()
    ma60 = cdf.rolling(60, min_periods=45).mean().to_numpy()
    amt20 = adf.rolling(20, min_periods=15).mean().to_numpy()

    feats = {
        "r5": (cdf / cdf.shift(5) - 1).to_numpy(),
        "r20": (cdf / cdf.shift(20) - 1).to_numpy(),
        "atr_pct": atr / C,
        "amount_ratio": A / amt20,
        "ma20_gap": C / ma20 - 1,
        "ma60_gap": C / ma60 - 1,
    }

    is_stock = np.array([bool(pd.Series([s]).str.fullmatch(r"[1-9]\d{3}").iloc[0]) for s in ids])
    rank = pd.DataFrame(np.where(is_stock[None, :], amt20, np.nan)).rank(axis=1, ascending=False).to_numpy()
    hist_ok = pd.DataFrame(~np.isnan(C)).rolling(60, min_periods=1).sum().to_numpy() >= 55
    U = (
        is_stock[None, :]
        & (amt20 >= UNIV_MIN_AMOUNT20)
        & (rank <= UNIV_TOP)
        & hist_ok
        & np.isfinite(C)
        & np.isfinite(atr)
    )

    total = IF + IT + ID
    tot5 = D(total).rolling(5, min_periods=3).sum().to_numpy()
    vol5 = vdf.rolling(5, min_periods=3).sum().to_numpy()
    feats["inst5_ratio_pct"] = tot5 / vol5 * 100.0

    ft = IF + IT
    typ = (H + L + C) / 3.0
    pos = np.clip(ft, 0, None)
    den60 = D(pos).rolling(60, min_periods=55).sum().to_numpy()
    num60 = D(typ * pos).rolling(60, min_periods=55).sum().to_numpy()
    cost60 = num60 / np.where(den60 > 0, den60, np.nan)
    feats["cost_dist_atr"] = (C - cost60) / atr

    entry = shift(O, 1)
    eL = shift(L, 1)
    eV = shift(V, 1)
    gap_cancel = entry >= (C + GAP_ATR * atr)
    locked = eL >= C * LIMIT_UP

    # Match the production baserate safeguards:
    # (a) do not label dates that do not yet have the full future horizon;
    # (b) exclude paths containing >10.5% close-to-close jumps, which are usually
    #     unadjusted corporate actions / par-value changes rather than tradable moves.
    row_no = np.arange(C.shape[0])[:, None]
    future_ok = row_no <= (C.shape[0] - HOLD - 2)
    prev_close = D(C).shift(1).to_numpy()
    jump = np.abs(C / prev_close - 1.0) > 0.105
    future_bad = np.zeros(C.shape, dtype=bool)
    jf = jump.astype(float)
    for kk in range(1, HOLD + 1):
        future_bad |= np.nan_to_num(shift(jf, kk), nan=0.0) > 0.5

    valid = U & future_ok & ~future_bad & np.isfinite(entry) & (eV > 0) & ~gap_cancel & ~locked

    stop_pct = np.clip(STOP_ATR * atr / C, STOP_MIN, STOP_MAX)
    stop = entry * (1.0 - stop_pct)
    tgt = entry * (1.0 + TARGET)
    early = entry * (1.0 + EARLY)

    status = np.zeros(C.shape, dtype=np.int8)
    reached5 = np.zeros(C.shape, dtype=bool)
    for k in range(2, HOLD + 1):
        Hk, Lk = shift(H, k), shift(L, k)
        act = valid & (status == 0)
        stop_hit = act & np.isfinite(Lk) & (Lk <= stop)
        status[stop_hit] = 2

        act = valid & (status == 0)
        hit = act & np.isfinite(Hk) & (Hk >= tgt)
        status[hit] = 1

        act = valid & (status == 0)
        reached5 |= act & np.isfinite(Hk) & (Hk >= early)
        if k == EARLY_DAYS:
            status[act & ~reached5] = 2

    status[valid & (status == 0)] = 2

    ui, uj = np.nonzero(U)
    X = np.column_stack([feats[n][ui, uj] for n in FEATURES]).astype(np.float32)
    y = (status[ui, uj] == 1).astype(np.int8)
    executed = valid[ui, uj]
    day = ui.astype(np.int32)
    return X, y, executed, day


def fit_base(X, y, mask):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=260,
        max_leaf_nodes=31,
        min_samples_leaf=500,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=20261002,
    )
    model.fit(X[mask], y[mask])
    return model


def fit_platt(raw_p, y):
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    p = np.clip(raw_p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(C=10.0, solver="lbfgs", random_state=20261002)
    cal.fit(z, y)
    return cal


def apply_platt(cal, raw_p):
    eps = 1e-6
    p = np.clip(raw_p, eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1]


def calc_metrics(y, p):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    out = {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "mean_p": float(np.mean(p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
    }
    q = pd.DataFrame({"y": y, "p": p})
    try:
        q["bin"] = pd.qcut(q["p"], 5, duplicates="drop")
        out["calibration"] = [
            {"n": int(len(g)), "p": float(g.p.mean()), "hit": float(g.y.mean())}
            for _, g in q.groupby("bin", observed=True)
        ]
    except Exception:
        out["calibration"] = []
    return out


# Frozen from the production v1.0 baserate computed on 2020-2024.
# This is deliberately NOT recomputed from 2025 outcomes.
ATR_BASE_EDGES = np.array([0.02446, 0.03323, 0.04176, 0.05321], dtype=float)
ATR_BASE_P30 = np.array([0.0795, 0.1636, 0.2220, 0.2755, 0.3495], dtype=float)


def atr_base_p30(atr_pct):
    idx = np.searchsorted(ATR_BASE_EDGES, np.asarray(atr_pct, dtype=float), side="right")
    return ATR_BASE_P30[idx]


def wilson95(k, n):
    if n <= 0:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [float(cen - half), float(cen + half)]


def threshold_sweep_2025(y, p, X, day, years, executed):
    atr = X[:, FEATURES.index("atr_pct")].astype(float)
    base = atr_base_p30(atr)
    edge = p - base
    rows = []
    for pp in [0, 2, 3, 4, 5, 6, 8]:
        th = pp / 100.0
        mask = executed & (years == 2025) & np.isfinite(edge) & (edge >= th)
        idx = np.where(mask)[0]

        def summarize(sel, mode):
            n = int(len(sel))
            if not n:
                return {
                    "threshold_pp": pp, "mode": mode, "n": 0, "days": 0,
                    "hit_rate": None, "hit_ci95": [None, None], "mean_p": None,
                    "mean_atr_base": None, "realized_lift_pp": None,
                }
            yy = y[sel].astype(float)
            hit = float(yy.mean())
            mb = float(base[sel].mean())
            return {
                "threshold_pp": pp,
                "mode": mode,
                "n": n,
                "days": int(len(np.unique(day[sel]))),
                "hit_rate": hit,
                "hit_ci95": wilson95(int(yy.sum()), n),
                "mean_p": float(p[sel].mean()),
                "mean_atr_base": mb,
                "realized_lift_pp": float((hit - mb) * 100.0),
                "mean_predicted_edge_pp": float((p[sel].mean() - mb) * 100.0),
            }

        rows.append(summarize(idx, "all_qualifying"))

        # Historical data does not contain Claude's final ranking, so this is an
        # empirical-only capacity check: at most five candidates per decision day,
        # ranked by empirical edge. It is NOT a backtest of the complete Claude system.
        top = []
        if len(idx):
            dvals = day[idx]
            for d0 in np.unique(dvals):
                z = idx[dvals == d0]
                order = np.argsort(-edge[z], kind="mergesort")
                top.extend(z[order[:5]].tolist())
        rows.append(summarize(np.asarray(top, dtype=int), "empirical_top5_by_edge_per_day"))

        # More natural production ordering: after passing the edge gate, rank the
        # remaining names by absolute empirical success probability.
        top_p = []
        if len(idx):
            dvals = day[idx]
            for d0 in np.unique(dvals):
                z = idx[dvals == d0]
                order = np.argsort(-p[z], kind="mergesort")
                top_p.extend(z[order[:5]].tolist())
        rows.append(summarize(np.asarray(top_p, dtype=int), "empirical_top5_by_p_per_day"))
    return rows


def score_candidates(model, cal, path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    X = np.asarray(
        [[np.nan if r.get(f) is None else float(r.get(f)) for f in FEATURES] for r in obj["candidates"]],
        dtype=np.float32,
    )
    raw = model.predict_proba(X)[:, 1]
    p = apply_platt(cal, raw)
    rows = []
    for r, pr, pc in zip(obj["candidates"], raw, p):
        q = dict(r)
        base = float(q.get("base30") or 0)
        q["empirical_raw_p30"] = round(float(pr), 6)
        q["empirical_p30"] = round(float(pc), 6)
        q["empirical_edge_pp"] = round((float(pc) - base) * 100, 2)
        q["passes_original_edge8_shadow"] = bool(
            float(pc) >= base + 0.08 and float(q.get("score100") or 0) >= 60
        )
        rows.append(q)
    return obj, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--candidate-file", default="research/input/forward_20261002_candidates.json")
    ap.add_argument("--out", default="research/empirical_out")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    px, inst, hashes = load_history(root)
    mats = make_matrices(px, inst)
    dates = mats[0]
    X, y, executed, day = build_dataset(*mats)

    years = pd.DatetimeIndex(dates[day]).year.to_numpy()
    m2020_23 = executed & (years <= 2023)
    m2024 = executed & (years == 2024)
    m2025 = executed & (years == 2025)

    if min(m2020_23.sum(), m2024.sum(), m2025.sum()) < 1000:
        raise RuntimeError(
            f"insufficient samples: fit={m2020_23.sum()}, cal={m2024.sum()}, test={m2025.sum()}"
        )

    # Untouched validation.
    base = fit_base(X, y, m2020_23)
    cal = fit_platt(base.predict_proba(X[m2024])[:, 1], y[m2024])
    p25 = apply_platt(cal, base.predict_proba(X[m2025])[:, 1])
    val = calc_metrics(y[m2025], p25)

    # Put the 2025 probabilities back into sample alignment for leakage-free
    # threshold testing against the frozen 2020-2024 ATR-matched baseline.
    p25_all = np.full(len(y), np.nan, dtype=float)
    p25_all[m2025] = p25
    sweep = threshold_sweep_2025(y, p25_all, X, day, years, executed)
    val.update(
        {
            "target": "plan-v3-hit15-h30",
            "features": FEATURES,
            "fit_2020_2023": int(m2020_23.sum()),
            "calibrate_2024": int(m2024.sum()),
            "validate_2025": int(m2025.sum()),
            "data_hashes": hashes,
            "note": "2025 was not used for fit, calibration, feature selection, or hyper-parameter search.",
        }
    )

    # 2026 shadow scorer.
    m2020_24 = executed & (years <= 2024)
    prod = fit_base(X, y, m2020_24)
    prod_cal = fit_platt(prod.predict_proba(X[m2025])[:, 1], y[m2025])
    cobj, scored = score_candidates(prod, prod_cal, root / args.candidate_file)

    score_obj = {
        "schema": "AlphaPilot empirical probability shadow v1",
        "decision_date": cobj["decision_date"],
        "target": "plan-v3-hit15-h30",
        "formal_forward_modified": False,
        "model": {"base_fit": "2020-2024", "calibration": "2025 Platt", "features": FEATURES},
        "candidates": sorted(scored, key=lambda r: r["empirical_p30"], reverse=True),
    }

    (outdir / "validation_2025.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "threshold_sweep_2025.json").write_text(
        json.dumps(
            {
                "scope": "2025 untouched OOS",
                "baseline_source": "production v1.0 ATR buckets computed from 2020-2024",
                "warning": "empirical_top5_per_day ranks by empirical edge because historical Claude final ranks do not exist; this is not a full Claude backtest.",
                "rows": sweep,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "score_20261002.json").write_text(
        json.dumps(score_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# AlphaPilot plan-v3 empirical shadow — 2026-10-02",
        "",
        "> Research branch only; live Claude Forward and R10-MAX are unchanged.",
        "",
        "## 2025 out-of-sample validation",
        "",
        f"- N: {val['n']:,}",
        f"- Base rate: {val['base_rate']:.2%}",
        f"- AUC: {val['auc']:.4f}" if val["auc"] is not None else "- AUC: NA",
        f"- Brier: {val['brier']:.5f}",
        f"- Mean predicted: {val['mean_p']:.2%}",
        "",
        "## 10/2 finalists",
        "",
        "| Rank | Code | Name | AI p | ATR base | empirical p | edge | old +8pp shadow |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(score_obj["candidates"], 1):
        lines.append(
            f"| {i} | {r['code']} | {r.get('name','')} | "
            f"{float(r.get('ai_p30') or 0):.0f}% | {float(r.get('base30') or 0):.1%} | "
            f"{r['empirical_p30']:.1%} | {r['empirical_edge_pp']:+.1f}pp | "
            f"{'PASS' if r['passes_original_edge8_shadow'] else 'NO'} |"
        )
    lines += [
        "",
        "## 2025 OOS empirical-edge threshold sweep",
        "",
        "| Edge gate | Mode | N | Days | Hit rate | Mean ATR base | Realized lift | Mean predicted edge |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sweep:
        if r["n"]:
            lines.append(
                f"| +{r['threshold_pp']}pp | {r['mode']} | {r['n']:,} | {r['days']} | "
                f"{r['hit_rate']:.1%} | {r['mean_atr_base']:.1%} | "
                f"{r['realized_lift_pp']:+.1f}pp | {r['mean_predicted_edge_pp']:+.1f}pp |"
            )
        else:
            lines.append(f"| +{r['threshold_pp']}pp | {r['mode']} | 0 | 0 | — | — | — | — |")
    lines += [
        "",
        "## Notes",
        "",
        "- Claude/AI p is kept only for comparison; it is not a feature or gate.",
        "- The old +8pp rule is applied only as a shadow comparison.",
        "- The 2026-10-02 future outcome is not used in training or calibration.",
    ]
    (outdir / "report_20261002.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"validation": val, "threshold_sweep": sweep, "candidates": score_obj["candidates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
