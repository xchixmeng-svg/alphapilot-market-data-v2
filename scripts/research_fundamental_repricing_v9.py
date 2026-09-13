#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fundamental_v9_out"
OUT.mkdir(parents=True, exist_ok=True)
FUND = ROOT / "data" / "history" / "full-market-fundamental-2024-2026"
PRICE = ROOT / "formal_run" / "ohlcv_causal_2020_2025.csv.gz"
INST = ROOT / "formal_run" / "institutional_2020_2025.parquet"

FEE = 0.000855
TAX = 0.003
BUY_SLIP = 0.005
SELL_SLIP = 0.005
HOLD = 20
MIN_HOLDOUT = 40
MIN_DEV = 40
MIN_WIN = 0.52
MIN_PF = 1.15
MIN_DEV_PF = 1.00


def ncol(df: pd.DataFrame, names: list[str], default=np.nan):
    for c in names:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def scol(df: pd.DataFrame, names: list[str], default=""):
    for c in names:
        if c in df.columns:
            return df[c].astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def pf(x: pd.Series) -> float:
    if len(x) == 0:
        return 0.0
    g = float(x[x > 0].sum())
    l = float(-x[x < 0].sum())
    if l <= 1e-12:
        return 99.0 if g > 0 else 0.0
    return g / l


def stats(t: pd.DataFrame, label: str) -> dict:
    if t.empty:
        return {"slice": label, "trades": 0, "win_rate": 0.0, "mean_net": 0.0, "median_net": 0.0,
                "pf": 0.0, "median_mae": np.nan, "median_mfe": np.nan}
    r = t["return_net"].astype(float)
    return {
        "slice": label,
        "trades": int(len(t)),
        "win_rate": float((r > 0).mean()),
        "mean_net": float(r.mean()),
        "median_net": float(r.median()),
        "pf": float(pf(r)),
        "median_mae": float(t["mae"].median()),
        "median_mfe": float(t["mfe"].median()),
    }


def read_fundamentals():
    rp = FUND / "month_revenue.csv.gz"
    if not rp.exists():
        raise FileNotFoundError(rp)
    r = pd.read_csv(rp, dtype={"stock_id": str}, low_memory=False)
    r["code"] = scol(r, ["stock_id", "code"]).str.zfill(4)
    r["source_date"] = pd.to_datetime(scol(r, ["date"]), errors="coerce")
    r["revenue"] = ncol(r, ["revenue", "Revenue"])
    r["yoy"] = ncol(r, ["revenue_yoy", "yoy", "RevenueYoY"])
    r["mom"] = ncol(r, ["revenue_mom", "mom", "RevenueMoM"])
    r["revenue_year"] = ncol(r, ["revenue_year", "year"])
    r["revenue_month"] = ncol(r, ["revenue_month", "month"])
    r = r[r.code.str.fullmatch(r"[1-9]\d{3}", na=False) & r.source_date.notna()].copy()
    if r["revenue_year"].isna().all():
        r["revenue_year"] = r.source_date.dt.year
    if r["revenue_month"].isna().all():
        r["revenue_month"] = r.source_date.dt.month
    r["period"] = (r.revenue_year.fillna(r.source_date.dt.year).astype(int) * 100 +
                   r.revenue_month.fillna(r.source_date.dt.month).astype(int))
    r = r.sort_values(["code", "period", "source_date"]).drop_duplicates(["code", "period"], keep="last")
    g = r.groupby("code", group_keys=False)
    if r["yoy"].isna().mean() > 0.8:
        r["yoy"] = g.revenue.pct_change(12) * 100
    if r["mom"].isna().mean() > 0.8:
        r["mom"] = g.revenue.pct_change(1) * 100
    r["yoy_l1"] = g.yoy.shift(1)
    r["yoy_l2"] = g.yoy.shift(2)
    r["yoy_l3"] = g.yoy.shift(3)
    r["accel"] = r.yoy - r.yoy_l1
    r["accel2"] = r.yoy - r.yoy_l2
    r["rev_rank"] = r.groupby("period")["yoy"].rank(pct=True)
    r["accel_rank"] = r.groupby("period")["accel2"].rank(pct=True)
    # Conservative anti-leak rule: revenue becomes usable no earlier than 10 calendar days after source date.
    r["available_date"] = r.source_date + pd.Timedelta(days=10)

    epsp = FUND / "eps_actual_extract.csv.gz"
    eps = pd.DataFrame(columns=["code", "eps_available", "eps", "eps_prev4", "eps_improve"])
    if epsp.exists():
        e = pd.read_csv(epsp, dtype={"stock_id": str}, low_memory=False)
        e["code"] = scol(e, ["stock_id", "code"]).str.zfill(4)
        e["source_date"] = pd.to_datetime(scol(e, ["date"]), errors="coerce")
        e["eps"] = ncol(e, ["value", "amount", "eps"])
        e = e[e.code.str.fullmatch(r"[1-9]\d{3}", na=False) & e.source_date.notna() & e.eps.notna()].copy()
        e = e.groupby(["code", "source_date"], as_index=False)["eps"].last().sort_values(["code", "source_date"])
        e["eps_prev4"] = e.groupby("code").eps.shift(4)
        e["eps_improve"] = ((e.eps > e.eps_prev4) & (e.eps > 0)) | ((e.eps > 0) & (e.eps_prev4 <= 0))
        # Conservative statement lag. It deliberately trades timeliness for anti-leak safety.
        e["eps_available"] = e.source_date + pd.Timedelta(days=45)
        eps = e[["code", "eps_available", "eps", "eps_prev4", "eps_improve"]].copy()
    return r, eps


def build_daily():
    px = pd.read_csv(PRICE, dtype={"code": str}, low_memory=False)
    px["code"] = px.code.astype(str).str.zfill(4)
    px["date"] = pd.to_numeric(px.date, errors="coerce").astype("Int64")
    px = px[px.date.notna()].copy(); px["date"] = px.date.astype(int)
    px["date_dt"] = pd.to_datetime(px.date.astype(str), format="%Y%m%d", errors="coerce")
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px["amount"] = px.close * px.volume
    g = px.groupby("code", group_keys=False)
    for w in (1, 5, 20, 60):
        px[f"r{w}"] = g.aclose.transform(lambda s, w=w: s.pct_change(w))
    for w in (20, 60):
        px[f"ma{w}"] = g.aclose.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    px["amount20"] = g.amount.transform(lambda s: s.rolling(20, min_periods=20).mean())
    px["vol20"] = g.aclose.transform(lambda s: s.pct_change().rolling(20, min_periods=20).std())
    px["dist_ma20"] = px.aclose / px.ma20 - 1

    inst = pd.read_parquet(INST)
    inst["code"] = inst.code.astype(str).str.zfill(4)
    if not np.issubdtype(inst.date.dtype, np.integer):
        inst["date"] = pd.to_datetime(inst.date).dt.strftime("%Y%m%d").astype(int)
    else:
        inst["date"] = inst.date.astype(int)
    inst = inst.sort_values(["code", "date"]).copy()
    gi = inst.groupby("code", group_keys=False)
    inst["inst5"] = gi.foreign_net.transform(lambda s: s.rolling(5, min_periods=5).sum()) + gi.trust_net.transform(lambda s: s.rolling(5, min_periods=5).sum())
    inst["inst20"] = gi.foreign_net.transform(lambda s: s.rolling(20, min_periods=20).sum()) + gi.trust_net.transform(lambda s: s.rolling(20, min_periods=20).sum())
    inst["prev5"] = gi.foreign_net.transform(lambda s: s.shift(5).rolling(5, min_periods=5).sum()) + gi.trust_net.transform(lambda s: s.shift(5).rolling(5, min_periods=5).sum())
    inst["inst_accel"] = inst.inst5 - inst.prev5
    px = px.merge(inst[["date", "code", "inst5", "inst20", "inst_accel"]], on=["date", "code"], how="left")
    px[["inst5", "inst20", "inst_accel"]] = px[["inst5", "inst20", "inst_accel"]].fillna(0)
    px["flow5"] = (px.inst5 * px.close) / (px.amount20 * 5).replace(0, np.nan)
    px["flow20"] = (px.inst20 * px.close) / (px.amount20 * 20).replace(0, np.nan)
    px["flow_accel"] = (px.inst_accel * px.close) / (px.amount20 * 5).replace(0, np.nan)

    valid = (px.code.str.fullmatch(r"[1-9]\d{3}", na=False) & (px.close >= 10) & (px.amount20 >= 50_000_000))
    if "name" in px:
        valid &= ~px.name.astype(str).str.contains("KY", case=False, na=False)
    q = px[valid].copy()
    for c in ["r20", "r60", "vol20", "flow5", "flow20", "flow_accel", "amount20"]:
        q[c + "_pr"] = q.groupby("date")[c].rank(pct=True)
    ctx = q.groupby("date").agg(
        breadth60=("aclose", lambda s: 0.0),
    ).reset_index().drop(columns="breadth60")
    b = q.assign(above60=q.aclose > q.ma60).groupby("date").above60.mean().rename("breadth60").reset_index()
    q = q.merge(b, on="date", how="left")
    return px, q


def map_events_to_daily(rev: pd.DataFrame, eps: pd.DataFrame, daily: pd.DataFrame):
    events = []
    bycode = {c: d.sort_values("date_dt") for c, d in daily.groupby("code")}
    for code, er in rev.groupby("code"):
        d = bycode.get(code)
        if d is None or d.empty:
            continue
        dates = d.date_dt.to_numpy(dtype="datetime64[ns]")
        for x in er.itertuples(index=False):
            if pd.isna(x.available_date):
                continue
            j = int(np.searchsorted(dates, np.datetime64(x.available_date), side="left"))
            if j >= len(d):
                continue
            row = d.iloc[j]
            z = x._asdict(); z["signal_date"] = int(row.date)
            events.append(z)
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev
    keep_daily = ["date", "code", "close", "aclose", "open", "high", "low", "r20", "r60", "ma20", "ma60", "dist_ma20",
                  "amount20", "vol20", "r20_pr", "r60_pr", "vol20_pr", "flow5", "flow20", "flow_accel",
                  "flow5_pr", "flow20_pr", "flow_accel_pr", "amount20_pr", "breadth60"]
    ev = ev.merge(daily[keep_daily], left_on=["signal_date", "code"], right_on=["date", "code"], how="left").drop(columns=["date"], errors="ignore")
    ev["signal_dt"] = pd.to_datetime(ev.signal_date.astype(str), format="%Y%m%d")

    if not eps.empty:
        pieces = []
        for code, d in ev.groupby("code"):
            e = eps[eps.code == code].sort_values("eps_available")
            d = d.sort_values("signal_dt").copy()
            if e.empty:
                d["eps"] = np.nan; d["eps_prev4"] = np.nan; d["eps_improve"] = False
            else:
                d = pd.merge_asof(d, e.drop(columns="code"), left_on="signal_dt", right_on="eps_available", direction="backward")
                d["eps_improve"] = d.eps_improve.fillna(False)
            pieces.append(d)
        ev = pd.concat(pieces, ignore_index=True)
    else:
        ev["eps"] = np.nan; ev["eps_prev4"] = np.nan; ev["eps_improve"] = False
    return ev


def simulate_events(name: str, sig: pd.DataFrame, px_all: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame()
    bycode = {c: d.sort_values("date").reset_index(drop=True) for c, d in px_all.groupby("code")}
    out = []; last_exit = {}
    for s in sig.sort_values(["signal_date", "score"], ascending=[True, False]).itertuples(index=False):
        d = bycode.get(s.code)
        if d is None: continue
        arr = d.date.to_numpy()
        k = int(np.searchsorted(arr, int(s.signal_date), side="right"))
        if k >= len(d) or k + HOLD >= len(d): continue
        entry = d.iloc[k]; exitr = d.iloc[k + HOLD]
        if int(entry.date) <= last_exit.get(s.code, 0): continue
        if not np.isfinite(entry.open) or entry.open <= 0 or not np.isfinite(exitr.open) or exitr.open <= 0: continue
        ep = float(entry.open) * (1 + BUY_SLIP); xp = float(exitr.open) * (1 - SELL_SLIP)
        ret = (xp * (1 - FEE - TAX)) / (ep * (1 + FEE)) - 1
        path = d.iloc[k:k + HOLD + 1]
        mae = float(path.low.min() / ep - 1) if "low" in path and path.low.notna().any() else np.nan
        mfe = float(path.high.max() / ep - 1) if "high" in path and path.high.notna().any() else np.nan
        out.append({"hypothesis": name, "code": s.code, "signal_date": int(s.signal_date), "entry_date": int(entry.date),
                    "exit_date": int(exitr.date), "entry_price": ep, "exit_price": xp, "return_net": ret, "mae": mae, "mfe": mfe,
                    "yoy": float(s.yoy) if np.isfinite(s.yoy) else np.nan, "mom": float(s.mom) if np.isfinite(s.mom) else np.nan,
                    "accel": float(s.accel) if np.isfinite(s.accel) else np.nan, "score": float(s.score)})
        last_exit[s.code] = int(exitr.date)
    return pd.DataFrame(out)


def main():
    rev, eps = read_fundamentals()
    px_all, daily = build_daily()
    ev = map_events_to_daily(rev, eps, daily)
    ev = ev[(ev.signal_date >= 20240101) & (ev.signal_date <= 20251231)].copy()
    ev = ev[ev.amount20 >= 50_000_000].copy()

    hypotheses = {
        "revenue_acceleration_flow": (
            lambda d: (d.yoy >= 10) & (d.accel >= 5) & (d.yoy_l1 > 0) & (d.flow5 > 0) & d.r20.between(-0.05, 0.15),
            lambda d: .35*d.accel_rank + .30*d.rev_rank + .20*d.flow5_pr + .15*(1-d.r60_pr)),
        "revenue_inflection_price_lag": (
            lambda d: (d.yoy >= 8) & ((d.yoy_l1 <= 0) | (d.yoy_l2 <= 0)) & (d.flow_accel > 0) & (d.r20 < 0.10) & (d.aclose > d.ma20),
            lambda d: .35*d.accel_rank + .25*d.rev_rank + .25*d.flow_accel_pr + .15*(1-d.r20_pr)),
        "persistent_revenue_quiet_base": (
            lambda d: (d.yoy >= 10) & (d.yoy_l1 >= 10) & (d.yoy_l2 >= 5) & (d.vol20_pr <= .65) & (d.dist_ma20.abs() <= .08) & (d.flow20 > 0),
            lambda d: .30*d.rev_rank + .25*d.accel_rank + .20*d.flow20_pr + .25*(1-d.vol20_pr)),
        "revenue_eps_confirmation": (
            lambda d: (d.yoy >= 8) & d.eps_improve.astype(bool) & (d.flow5 > 0) & (d.r60_pr <= .85),
            lambda d: .35*d.rev_rank + .25*d.accel_rank + .20*d.flow5_pr + .20*(1-d.r60_pr)),
        "fundamental_price_gap": (
            lambda d: (d.rev_rank >= .80) & (d.accel_rank >= .60) & (d.r60_pr <= .60) & (d.flow5 > 0),
            lambda d: .40*d.rev_rank + .25*d.accel_rank + .20*(1-d.r60_pr) + .15*d.flow5_pr),
        "cycle_early_reacceleration": (
            lambda d: (d.yoy >= 5) & (d.mom > 0) & (d.accel >= 3) & (d.r20 < .12) & (d.flow5 > 0),
            lambda d: .30*d.accel_rank + .25*d.rev_rank + .20*d.flow5_pr + .15*(1-d.r20_pr) + .10*d.breadth60.clip(0,1)),
    }

    summary = []; survivor_names = []
    for name, (cond, scoref) in hypotheses.items():
        raw = ev[cond(ev).fillna(False)].copy()
        if raw.empty:
            tr = pd.DataFrame()
        else:
            raw["score"] = scoref(raw).replace([np.inf, -np.inf], np.nan).fillna(0)
            # One compact cross-sectional selection per disclosure/market day. No threshold grid search.
            sig = raw.sort_values(["signal_date", "score", "amount20"], ascending=[True, False, False]).groupby("signal_date", as_index=False).head(10)
            tr = simulate_events(name, sig, px_all)
        dev = tr[(tr.signal_date >= 20240101) & (tr.signal_date <= 20241231)].copy() if not tr.empty else pd.DataFrame()
        ho = tr[(tr.signal_date >= 20250101) & (tr.signal_date <= 20251231)].copy() if not tr.empty else pd.DataFrame()
        ds = stats(dev, "2024_dev"); hs = stats(ho, "2025_holdout")
        gate = (hs["trades"] >= MIN_HOLDOUT and ds["trades"] >= MIN_DEV and hs["win_rate"] >= MIN_WIN and
                hs["mean_net"] > 0 and hs["pf"] >= MIN_PF and ds["pf"] >= MIN_DEV_PF and ds["mean_net"] >= 0)
        reasons = []
        if hs["trades"] < MIN_HOLDOUT: reasons.append("holdout_sample")
        if ds["trades"] < MIN_DEV: reasons.append("dev_sample")
        if hs["win_rate"] < MIN_WIN: reasons.append("holdout_win")
        if hs["mean_net"] <= 0: reasons.append("holdout_mean")
        if hs["pf"] < MIN_PF: reasons.append("holdout_pf")
        if ds["pf"] < MIN_DEV_PF or ds["mean_net"] < 0: reasons.append("dev_direction")
        row = {"hypothesis": name, "pass_stage1": bool(gate), "reject_reason": ";".join(reasons),
               **{"dev_" + k: v for k, v in ds.items() if k != "slice"},
               **{"holdout_" + k: v for k, v in hs.items() if k != "slice"}}
        summary.append(row)
        if gate:
            survivor_names.append(name)
            ddir = OUT / "survivors" / name; ddir.mkdir(parents=True, exist_ok=True)
            tr.to_csv(ddir / "event_trades.csv", index=False)
            (ddir / "stage1_gate.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    sm = pd.DataFrame(summary).sort_values(["pass_stage1", "holdout_win_rate", "holdout_pf", "holdout_mean_net"], ascending=[False, False, False, False])
    sm.to_csv(OUT / "screen_summary.csv", index=False)
    audit = {
        "version": "fundamental-repricing-v9",
        "research_scope": "quick gate only; no minute optimization",
        "fundamental_sources": ["FinMind TaiwanStockMonthRevenue", "FinMind TaiwanStockFinancialStatements EPS extract"],
        "anti_leak": {"monthly_revenue_calendar_lag_days": 10, "financial_statement_calendar_lag_days": 45,
                      "decision": "T close after conservative availability date", "entry": "next trading day open +0.5% adverse slippage"},
        "costs": {"buy_fee": FEE, "sell_fee": FEE, "sell_tax": TAX, "buy_slippage": BUY_SLIP, "sell_slippage": SELL_SLIP},
        "hold_days": HOLD,
        "gates": {"dev_min_trades": MIN_DEV, "holdout_min_trades": MIN_HOLDOUT, "holdout_min_win": MIN_WIN,
                  "holdout_min_pf": MIN_PF, "dev_min_pf": MIN_DEV_PF, "positive_net_mean_both": True},
        "hypotheses": list(hypotheses),
        "survivors": survivor_names,
        "survivor_count": len(survivor_names),
        "note": "Failures intentionally produce only compact summary rows. Detailed trades are emitted only for Stage-1 survivors."
    }
    (OUT / "research_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if survivor_names:
        (OUT / "survivors_present.flag").write_text("\n".join(survivor_names) + "\n", encoding="utf-8")
    print(sm.to_string(index=False))
    print(json.dumps({"survivors": survivor_names, "count": len(survivor_names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
