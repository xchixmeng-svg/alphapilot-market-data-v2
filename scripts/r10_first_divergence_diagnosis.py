#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forensic comparison of legacy ledger vs clean R10 output.

IMPORTANT: legacy fixtures are comparison-only. They are never imported by the
trading engine. This script runs only after r10_true_validation.py has produced
its own orders/trades/NAV from raw OHLCV + institutional data.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

import r10_true_validation as true

bt = true.bt
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stress_results" / "latest" / "true_validation2021_2025"
GOLD = ROOT / "tests" / "fixtures" / "golden_2021_2025"


def load_gold(prefix: str) -> pd.DataFrame:
    parts = sorted(GOLD.glob(f"{prefix}_part*.csv"))
    return pd.concat([pd.read_csv(p, dtype={"code": str}) for p in parts], ignore_index=True)


def held_actual(orders: pd.DataFrame, di: int) -> list[dict]:
    pos: dict[tuple[str, str], dict] = {}
    q = orders[orders.execute_date <= di].sort_values(["execute_date", "side"], kind="stable")
    for r in q.itertuples(index=False):
        key = (str(r.strategy), str(r.code).zfill(4))
        if str(r.side) == "BUY" and bool(r.filled):
            pos[key] = {"strategy": key[0], "code": key[1], "shares": int(r.shares), "entry_price": float(r.fill_price)}
        elif str(r.side) == "SELL" and bool(r.filled):
            pos.pop(key, None)
    return list(pos.values())


def main() -> None:
    actual_nav = pd.read_csv(OUT / "daily_nav.csv")
    actual_orders = pd.read_csv(OUT / "orders.csv", dtype={"code": str})
    actual_trades = pd.read_csv(OUT / "trades.csv", dtype={"code": str})
    legacy_nav = load_gold("nav")
    legacy_orders = load_gold("orders")
    legacy_exits = load_gold("exits")

    for q in (actual_orders, actual_trades, legacy_orders, legacy_exits):
        if "code" in q:
            q["code"] = q.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)

    # Reload the exact raw-data pipeline used by the clean engine.
    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = bt.load_scenario_ohlcv(cfg)
    feat, _, _ = bt.build_features(raw)
    feat_idx = feat.set_index(["date", "code"]).sort_index()

    # 1) Earliest NAV divergence.
    an = actual_nav[["date", "nav", "cash", "stock_mv"]].copy()
    gn = legacy_nav[["date", "nav", "cash", "market_value"]].copy()
    m = an.merge(gn, on="date", suffixes=("_actual", "_legacy"))
    m["nav_delta"] = m.nav_actual - m.nav_legacy
    first = m.loc[m.nav_delta.abs() > 0.01].iloc[0]
    di = int(first.date)

    positions = held_actual(actual_orders, di)
    valuation_rows = []
    raw_mv = 0.0
    adj_mv = 0.0
    for p in positions:
        try:
            r = feat_idx.loc[(di, p["code"])]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[-1]
        except KeyError:
            continue
        rmv = p["shares"] * float(r.close)
        amv = p["shares"] * float(r.aclose)
        raw_mv += rmv
        adj_mv += amv
        valuation_rows.append({
            **p,
            "raw_close": float(r.close),
            "adjusted_close": float(r.aclose),
            "adjustment_factor": float(r.aclose / r.close) if r.close else np.nan,
            "raw_mv": rmv,
            "adjusted_mv": amv,
        })

    # 2) First BUY divergence by sequence.
    ao = actual_orders[actual_orders.side.eq("BUY")].reset_index(drop=True)
    go = legacy_orders.reset_index(drop=True)
    order_diff = None
    n = min(len(ao), len(go))
    for i in range(n):
        a, g = ao.iloc[i], go.iloc[i]
        ak = (int(a.decision_date), str(a.strategy), str(a.code))
        gk = (int(g.decision_date), str(g.strategy), str(g.code))
        if ak != gk:
            order_diff = {"index": i, "type": "identity", "actual": ak, "legacy": gk}
            break
        for fld, av, gv, tol in [
            ("execute_date", int(a.execute_date), int(g.execute_date), 0),
            ("limit", float(a.order_price), float(g.limit), 1e-8),
            ("shares", int(a.shares), int(g.shares), 0),
            ("target_cash", float(a.target_cash), float(g.target_cash), 0.05),
        ]:
            if (av != gv if tol == 0 else abs(av-gv) > tol):
                order_diff = {"index": i, "key": ak, "type": fld, "actual": av, "legacy": gv, "delta": av-gv}
                break
        if order_diff:
            break

    # 3) First exit-date divergence and exact source rows around it.
    amap = {(str(r.strategy), str(r.code), int(r.entry_date)): r for r in actual_trades.itertuples(index=False)}
    exit_diff = None
    exit_source = []
    for i, g in enumerate(legacy_exits.itertuples(index=False)):
        key = (str(g.strategy), str(g.code), int(g.entry_date))
        a = amap.get(key)
        if a is None or int(a.exit_date) != int(g.exit_date):
            exit_diff = {
                "index": i, "key": key,
                "legacy_sell_decision_date": int(g.sell_decision_date),
                "legacy_exit_date": int(g.exit_date),
                "actual_exit_date": None if a is None else int(a.exit_date),
                "legacy_exit_price": float(g.exit_price),
                "actual_exit_price": None if a is None else float(a.exit_price),
            }
            if a is not None and key[0] == "R05":
                er = feat_idx.loc[(key[2], key[1])]
                if isinstance(er, pd.DataFrame): er = er.iloc[-1]
                entry_price = float(a.entry_price)
                entry_factor = float(er.aclose / er.close) if er.close else 1.0
                entry_adj = entry_price * entry_factor
                hard = entry_adj * 0.90
                dates = sorted(d for d in feat.date.unique() if key[2] <= int(d) <= int(a.exit_date))
                for d in dates:
                    try:
                        r = feat_idx.loc[(int(d), key[1])]
                        if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                    except KeyError:
                        continue
                    factor = float(r.aclose / r.close) if r.close else 1.0
                    low_adj = float(r.low) * factor
                    exit_source.append({
                        "date": int(d), "open": float(r.open), "low": float(r.low), "close": float(r.close),
                        "aclose": float(r.aclose), "factor": factor, "low_adj": low_adj,
                        "hard_threshold_adj": hard,
                        "close_hard_trigger": bool(float(r.aclose) <= hard + 1e-12),
                        "intraday_low_hard_trigger": bool(low_adj <= hard + 1e-12),
                    })
            break

    report = {
        "reference_is_engine_input": False,
        "first_nav_divergence": {
            "date": di,
            "actual_nav": float(first.nav_actual), "legacy_nav": float(first.nav_legacy),
            "nav_delta": float(first.nav_delta),
            "actual_cash": float(first.cash_actual), "legacy_cash": float(first.cash_legacy),
            "cash_delta": float(first.cash_actual-first.cash_legacy),
            "actual_stock_mv": float(first.stock_mv), "legacy_market_value": float(first.market_value),
            "market_value_delta": float(first.stock_mv-first.market_value),
            "recomputed_raw_close_mv": raw_mv,
            "recomputed_adjusted_close_mv": adj_mv,
            "legacy_minus_raw_mv": float(first.market_value-raw_mv),
            "legacy_minus_adjusted_mv": float(first.market_value-adj_mv),
            "positions": valuation_rows,
        },
        "first_buy_divergence": order_diff,
        "first_exit_divergence": exit_diff,
        "first_exit_source_rows": exit_source,
    }

    p = OUT / "first_divergence_diagnosis.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
