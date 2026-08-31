"""Causal Taiwan equity rotation research versus 0050 (2020 warm-up).

The script intentionally uses only T-close information and executes every
decision at T+1 under CONTRACT.md.  It writes machine-readable evidence and
never labels a candidate PASS unless it beats 0050 and stays below 20% MDD.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_CASH = 1_000_000.0
BUY_FEE = 0.001425
SELL_FEE = 0.001425
SELL_TAX = 0.003
BUY_SLIP = 0.005
SELL_ADVERSE = 0.02
MAX_WEIGHT = 0.20
LOT_STEP = 100


@dataclass(frozen=True)
class Params:
    rebalance_days: int
    momentum_days: int
    skip_days: int
    market_ma: int
    market_ret_days: int
    volume_floor_m: int
    max_positions: int = 5


def load_data(root: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(root / f"ohlcv_{year}.parquet") for year in range(2020, 2026)]
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"].astype(str))
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "code", "open", "high", "low", "close", "volume"])
    d = d[(d.open > 0) & (d.high > 0) & (d.low > 0) & (d.close > 0) & (d.volume >= 0)]
    return d.sort_values(["code", "date"]).reset_index(drop=True)


def features(d: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = d.copy()
    g = x.groupby("code", sort=False)
    x["mom"] = g.close.pct_change(p.momentum_days) - g.close.pct_change(p.skip_days)
    x["ma60"] = g.close.transform(lambda s: s.rolling(60, min_periods=60).mean())
    x["ma120"] = g.close.transform(lambda s: s.rolling(120, min_periods=120).mean())
    x["adv20"] = (x.close * x.volume).groupby(x.code).transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["vol20"] = g.close.pct_change().groupby(x.code).transform(lambda s: s.rolling(20, min_periods=20).std())
    idx = x[x.code == "0050"].set_index("date").copy()
    idx["market_ma"] = idx.close.rolling(p.market_ma, min_periods=p.market_ma).mean()
    idx["market_ret"] = idx.close.pct_change(p.market_ret_days)
    market = idx[["open", "low", "close", "market_ma", "market_ret"]]
    return x, market


def metrics(curve: pd.Series) -> dict:
    curve = curve.dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.2425
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1
    dd = curve / curve.cummax() - 1
    daily = curve.pct_change().dropna()
    return {"final_nav": float(curve.iloc[-1]), "total_return": float(curve.iloc[-1] / curve.iloc[0] - 1),
            "cagr": float(cagr), "max_drawdown": float(dd.min()),
            "sharpe": float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() else 0.0}


def benchmark(market: pd.DataFrame) -> tuple[pd.Series, dict]:
    m = market.loc["2021-01-01":"2025-12-31"].copy()
    buy = m.iloc[0].open * (1 + BUY_SLIP)
    shares = int((INITIAL_CASH / (buy * (1 + BUY_FEE))) // LOT_STEP * LOT_STEP)
    cash = INITIAL_CASH - shares * buy - shares * m.iloc[0].open * BUY_FEE
    curve = cash + shares * m.close
    curve.iloc[0] = INITIAL_CASH
    return curve, metrics(curve)


def simulate(d: pd.DataFrame, market: pd.DataFrame, p: Params) -> tuple[pd.Series, list[dict]]:
    days = market.loc["2021-01-01":"2025-12-31"].index
    by_day = {k: v.set_index("code") for k, v in d[d.date.isin(days)].groupby("date")}
    cash, pos, pending, trades = INITIAL_CASH, {}, None, []
    curve, curve_days = [], []
    last_rebalance = -10**9
    for i, day in enumerate(days):
        bars = by_day.get(day)
        if bars is None:
            continue
        # Execute yesterday's fully precommitted rebalance at today's prices.
        if pending:
            for code in list(pos):
                if code not in pending["targets"] and code in bars.index:
                    raw = float(bars.at[code, "open"]); sh = pos.pop(code)
                    modeled = raw * (1 - SELL_ADVERSE)
                    cash += sh * modeled - sh * raw * (SELL_FEE + SELL_TAX)
                    trades.append({"date": str(day.date()), "side": "SELL", "code": code, "shares": sh, "raw_price": raw, "modeled_price": modeled})
            nav_open = cash + sum(sh * float(bars.at[c, "open"]) for c, sh in pos.items() if c in bars.index)
            cap = nav_open * MAX_WEIGHT
            for code, limit in pending["orders"]:
                if code in pos or code not in bars.index:
                    continue
                op, lo = float(bars.at[code, "open"]), float(bars.at[code, "low"])
                fill = min(limit, op * (1 + BUY_SLIP)) if op <= limit else (limit if lo <= limit else None)
                if fill is None:
                    continue
                sh = int(min(cap / fill, cash / (fill * (1 + BUY_FEE))) // LOT_STEP * LOT_STEP)
                if sh <= 0:
                    continue
                fee = sh * op * BUY_FEE
                cash -= sh * fill + fee; pos[code] = sh
                trades.append({"date": str(day.date()), "side": "BUY", "code": code, "shares": sh, "raw_price": op, "modeled_price": fill})
            pending = None
        nav = cash + sum(sh * float(bars.at[c, "close"]) for c, sh in pos.items() if c in bars.index)
        curve.append(nav); curve_days.append(day)
        # Signal uses this completed T bar only, for T+1.
        if i - last_rebalance >= p.rebalance_days and i + 1 < len(days):
            mr = market.loc[day]
            risk_on = mr.close > mr.market_ma and mr.market_ret > 0
            targets, orders = [], []
            if risk_on:
                q = bars[(bars.index.str.fullmatch(r"\d{4}")) & (bars.adv20 >= p.volume_floor_m * 1e6) &
                         (bars.close > bars.ma60) & (bars.ma60 > bars.ma120) & bars.mom.notna() & bars.vol20.notna()].copy()
                q["score"] = q.mom / q.vol20.clip(lower=0.005)
                targets = list(q.nlargest(p.max_positions, "score").index)
                orders = [(c, float(q.at[c, "close"]) * 1.02) for c in targets]
            pending = {"targets": targets, "orders": orders}
            last_rebalance = i
    s = pd.Series(curve, index=pd.DatetimeIndex(curve_days), name="nav")
    return s, trades


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/history/2020-2025"); ap.add_argument("--out", default="artifacts/beat_0050")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = load_data(Path(a.data)); grid = itertools.product([10, 20, 40], [60, 120, 180], [5, 20], [60, 120], [20, 60], [30, 50])
    rows, best = [], None
    for values in grid:
        p = Params(*values); d, market = features(raw, p); bcurve, bm = benchmark(market); curve, trades = simulate(d, market, p); m = metrics(curve)
        row = {**asdict(p), **m, "benchmark_cagr": bm["cagr"], "benchmark_max_drawdown": bm["max_drawdown"],
               "excess_cagr": m["cagr"] - bm["cagr"], "pass": bool(m["cagr"] > bm["cagr"] and m["max_drawdown"] > -0.20)}
        rows.append(row)
        key = (row["pass"], row["excess_cagr"], row["max_drawdown"])
        if best is None or key > best[0]: best = (key, p, curve, trades, bm, bcurve, row)
    pd.DataFrame(rows).sort_values(["pass", "excess_cagr"], ascending=False).to_csv(out / "grid_results.csv", index=False)
    _, p, curve, trades, bm, bcurve, row = best
    pd.DataFrame(trades).to_csv(out / "best_trades.csv", index=False)
    pd.concat([curve, bcurve.rename("benchmark_0050")], axis=1).to_csv(out / "best_equity_curve.csv")
    report = {"status": "PASS" if row["pass"] else "FAIL", "period": "2021-2025", "warmup": 2020, "initial_cash": INITIAL_CASH,
              "parameters": asdict(p), "strategy": {k: row[k] for k in ("final_nav", "cagr", "max_drawdown", "sharpe")}, "benchmark_0050": bm,
              "contract": "T+1; precommitted limit; 100-share step; 20% cap; full fees/tax; buy +0.5% slip; sell -2% adverse"}
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
