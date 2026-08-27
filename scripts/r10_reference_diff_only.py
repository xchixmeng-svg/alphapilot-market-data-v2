#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-run forensic diff only.

This script is intentionally separate from r10_true_validation.py. It may read
legacy reference ledgers only AFTER the clean engine has completed. It never
feeds a reference row back into signal, order, sizing, fill, exit, or NAV logic.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stress_results" / "latest" / "true_validation2021_2025"
GOLD = ROOT / "tests" / "fixtures" / "golden_2021_2025"


def _date_int(v):
    return int(str(v).replace("-", "")[:8])


def main() -> None:
    actual_orders = pd.read_csv(OUT / "orders.csv", dtype={"code": str})
    actual_trades = pd.read_csv(OUT / "trades.csv", dtype={"code": str})
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))

    op = sorted(GOLD.glob("orders_part*.csv"))
    xp = sorted(GOLD.glob("exits_part*.csv"))
    legacy_orders = pd.concat([pd.read_csv(p, dtype={"code": str}) for p in op], ignore_index=True) if op else pd.DataFrame()
    legacy_exits = pd.concat([pd.read_csv(p, dtype={"code": str}) for p in xp], ignore_index=True) if xp else pd.DataFrame()

    ao = actual_orders[actual_orders.side.eq("BUY")].copy()
    for q in (ao, legacy_orders, actual_trades, legacy_exits):
        if not q.empty and "code" in q:
            q["code"] = q.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)

    report = {
        "reference_is_engine_input": False,
        "comparison_purpose": "forensic-first-divergence-only",
        "true_result": summary,
        "legacy_reference": {
            "end_nav": 9888538.413551485,
            "cagr": 0.5019270634416155,
            "max_dd": -0.12258760312884043,
            "completed_trades": 241,
            "status": "QUARANTINED_LOOKAHEAD_CONTAMINATION",
        },
        "actual_buy_orders": int(len(ao)),
        "legacy_buy_orders": int(len(legacy_orders)),
        "actual_trades": int(len(actual_trades)),
        "legacy_trades": int(len(legacy_exits)),
    }

    first_order_diff = None
    if not legacy_orders.empty:
        for c in ("decision_date", "execute_date"):
            legacy_orders[c] = legacy_orders[c].map(_date_int)
        ao["decision_date"] = ao.decision_date.map(_date_int)
        ao["execute_date"] = ao.execute_date.map(_date_int)
        akeys = [(int(r.decision_date), str(r.strategy), str(r.code)) for r in ao.itertuples(index=False)]
        gkeys = [(int(r.decision_date), str(r.strategy), str(r.code)) for r in legacy_orders.itertuples(index=False)]
        n = max(len(akeys), len(gkeys))
        for i in range(n):
            ak = akeys[i] if i < len(akeys) else None
            gk = gkeys[i] if i < len(gkeys) else None
            if ak != gk:
                first_order_diff = {"index": i, "actual": ak, "legacy": gk, "type": "identity"}
                break
            ar = ao.iloc[i]
            gr = legacy_orders.iloc[i]
            fields = [
                ("execute_date", int(ar.execute_date), int(gr.execute_date), 0.0),
                ("limit", float(ar.order_price), float(gr.limit), 1e-8),
                ("shares", int(ar.shares), int(gr.shares), 0.0),
                ("target_cash", float(ar.target_cash), float(gr.target_cash), 0.05),
            ]
            for name, av, gv, tol in fields:
                mismatch = (av != gv) if tol == 0 else abs(av - gv) > tol
                if mismatch:
                    first_order_diff = {
                        "index": i, "key": ak, "type": name,
                        "actual": av, "legacy": gv, "delta": av - gv,
                    }
                    break
            if first_order_diff:
                break
            exp_filled = str(gr.status).upper() == "FILLED"
            if bool(ar.filled) != exp_filled:
                first_order_diff = {
                    "index": i, "key": ak, "type": "fill_status",
                    "actual": bool(ar.filled), "legacy": exp_filled,
                }
                break
            if exp_filled and pd.notna(gr.fill_price):
                av = float(ar.fill_price)
                gv = float(gr.fill_price)
                if abs(av - gv) > 1e-8:
                    first_order_diff = {
                        "index": i, "key": ak, "type": "fill_price",
                        "actual": av, "legacy": gv, "delta": av - gv,
                    }
                    break
    report["first_legacy_order_divergence"] = first_order_diff

    first_exit_diff = None
    if not legacy_exits.empty:
        for c in ("entry_date", "exit_date"):
            legacy_exits[c] = legacy_exits[c].map(_date_int)
        if not actual_trades.empty:
            actual_trades["entry_date"] = actual_trades.entry_date.map(_date_int)
            actual_trades["exit_date"] = actual_trades.exit_date.map(_date_int)
        amap = {
            (str(r.strategy), str(r.code), int(r.entry_date)): r
            for r in actual_trades.itertuples(index=False)
        }
        for i, gr in enumerate(legacy_exits.itertuples(index=False)):
            key = (str(gr.strategy), str(gr.code), int(gr.entry_date))
            ar = amap.get(key)
            if ar is None:
                first_exit_diff = {"index": i, "key": key, "type": "missing_trade_in_true_engine"}
                break
            if int(ar.exit_date) != int(gr.exit_date):
                first_exit_diff = {
                    "index": i, "key": key, "type": "exit_date",
                    "actual": int(ar.exit_date), "legacy": int(gr.exit_date),
                }
                break
            if hasattr(gr, "exit_price") and pd.notna(gr.exit_price):
                av, gv = float(ar.exit_price), float(gr.exit_price)
                if abs(av - gv) > 1e-8:
                    first_exit_diff = {
                        "index": i, "key": key, "type": "exit_price",
                        "actual": av, "legacy": gv, "delta": av - gv,
                    }
                    break
    report["first_legacy_exit_divergence"] = first_exit_diff

    p = OUT / "reference_diff_only.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
