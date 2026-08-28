#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-run forensic comparison of the formal R10 ledger vs the clean engine.

IMPORTANT:
- The trading engine NEVER imports the reference ledger.
- This script runs only after r10_true_validation.py has independently finished.
- Historical NAV is NOT trusted as an input. It is mechanically rebuilt from
  the formal 410-order / 241-exit ledger + the same raw OHLCV used by the clean
  engine. This prevents a stale/corrupt NAV export from creating a fake diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import r10_true_validation as true

bt = true.bt
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stress_results" / "latest" / "true_validation2021_2025"
REF = ROOT / "tests" / "fixtures" / "golden_2021_2025"


def load_reference(prefix: str) -> pd.DataFrame:
    parts = sorted(REF.glob(f"{prefix}_part*.csv"))
    if not parts:
        raise RuntimeError(f"missing post-run reference parts: {prefix}")
    return pd.concat([pd.read_csv(p, dtype={"code": str}) for p in parts], ignore_index=True)


def norm_code(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    if "code" in q:
        q["code"] = q.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    return q


def row_at(feat_idx, di: int, code: str):
    try:
        r = feat_idx.loc[(int(di), str(code).zfill(4))]
        return r.iloc[-1] if isinstance(r, pd.DataFrame) else r
    except KeyError:
        return None


def rebuild_reference_nav(
    eval_dates: list[int], feat_idx, orders: pd.DataFrame, exits: pd.DataFrame
) -> pd.DataFrame:
    """Rebuild the formal historical account without reading a NAV answer file."""
    cash = float(bt.INITIAL_CAPITAL)
    positions: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []

    buys_by_day = {
        int(di): g.copy()
        for di, g in orders[orders.status.astype(str).str.upper().eq("FILLED")].groupby("execute_date")
    }
    exits_by_day = {int(di): g.copy() for di, g in exits.groupby("exit_date")}

    for di in eval_dates:
        # Formal/common-pool execution ordering: T+1 sells first, then T+1 buys.
        for x in exits_by_day.get(int(di), pd.DataFrame()).itertuples(index=False):
            key = (str(x.strategy), str(x.code).zfill(4))
            p = positions.get(key)
            if p is None:
                raise RuntimeError(f"reference exit has no open position {di} {key}")
            shares = int(x.shares)
            if shares != int(p["shares"]):
                raise RuntimeError(
                    f"reference exit share mismatch {di} {key}: exit={shares} held={p['shares']}"
                )
            gross = float(x.exit_price) * shares
            proceeds = gross * (1.0 - bt.SELL_FEE - bt.SELL_TAX)
            # Ledger's proceeds column is an audit field only; recompute it here.
            if hasattr(x, "proceeds") and pd.notna(x.proceeds):
                if abs(proceeds - float(x.proceeds)) > 0.05:
                    raise RuntimeError(
                        f"reference proceeds formula mismatch {di} {key}: calc={proceeds} ledger={x.proceeds}"
                    )
            cash += proceeds
            del positions[key]

        for o in buys_by_day.get(int(di), pd.DataFrame()).itertuples(index=False):
            key = (str(o.strategy), str(o.code).zfill(4))
            if key in positions:
                raise RuntimeError(f"reference duplicate open position {di} {key}")
            shares = int(o.fill_shares) if hasattr(o, "fill_shares") and pd.notna(o.fill_shares) else int(o.shares)
            fill = float(o.fill_price)
            cost = fill * shares * (1.0 + bt.BUY_FEE)
            if cost > cash + 1e-6:
                raise RuntimeError(f"reference ledger overdraft {di} {key}: cost={cost} cash={cash}")
            cash -= cost
            positions[key] = {
                "strategy": key[0], "code": key[1], "name": str(o.name),
                "shares": shares, "entry_date": int(di), "entry_price": fill,
            }

        mv = 0.0
        for p in positions.values():
            r = row_at(feat_idx, int(di), p["code"])
            px = float(r.close) if r is not None and np.isfinite(r.close) else float(p["entry_price"])
            mv += int(p["shares"]) * px
        rows.append({
            "date": int(di), "nav": cash + mv, "cash": cash,
            "market_value": mv, "positions": len(positions),
        })

    return pd.DataFrame(rows)


def held_actual(orders: pd.DataFrame, di: int) -> list[dict]:
    pos: dict[tuple[str, str], dict] = {}
    q = orders[orders.execute_date <= di].copy()
    # Preserve actual emitted execution ordering. When same day exists, SELL first.
    q["_side_order"] = q.side.map({"SELL": 0, "BUY": 1}).fillna(9)
    q = q.sort_values(["execute_date", "_side_order"], kind="stable")
    for r in q.itertuples(index=False):
        key = (str(r.strategy), str(r.code).zfill(4))
        if str(r.side) == "BUY" and bool(r.filled):
            pos[key] = {
                "strategy": key[0], "code": key[1], "shares": int(r.shares),
                "entry_price": float(r.fill_price),
            }
        elif str(r.side) == "SELL" and bool(r.filled):
            pos.pop(key, None)
    return list(pos.values())


def main() -> None:
    actual_nav = pd.read_csv(OUT / "daily_nav.csv")
    actual_orders = norm_code(pd.read_csv(OUT / "orders.csv", dtype={"code": str}))
    actual_trades = norm_code(pd.read_csv(OUT / "trades.csv", dtype={"code": str}))
    reference_orders = norm_code(load_reference("orders"))
    reference_exits = norm_code(load_reference("exits"))

    for c in ("decision_date", "execute_date"):
        reference_orders[c] = pd.to_numeric(reference_orders[c], errors="raise").astype(int)
    for c in ("entry_date", "sell_decision_date", "exit_date"):
        reference_exits[c] = pd.to_numeric(reference_exits[c], errors="raise").astype(int)

    # Reload the exact raw-data pipeline independently used by the clean engine.
    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = bt.load_scenario_ohlcv(cfg)
    feat, _, _ = bt.build_features(raw)
    feat_idx = feat.set_index(["date", "code"]).sort_index()
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= int(x) <= eval_end)

    reference_nav = rebuild_reference_nav(eval_dates, feat_idx, reference_orders, reference_exits)

    # 1) Earliest NAV/cash/market-value divergence, using REBUILT reference NAV.
    an = actual_nav[["date", "nav", "cash", "stock_mv"]].copy()
    gn = reference_nav[["date", "nav", "cash", "market_value"]].copy()
    m = an.merge(gn, on="date", suffixes=("_actual", "_reference"))
    m["nav_delta"] = m.nav_actual - m.nav_reference
    bad = m.loc[m.nav_delta.abs() > 0.01]
    first = bad.iloc[0] if not bad.empty else None

    first_nav = None
    if first is not None:
        di = int(first.date)
        first_nav = {
            "date": di,
            "actual_nav": float(first.nav_actual), "reference_nav": float(first.nav_reference),
            "nav_delta_actual_minus_reference": float(first.nav_delta),
            "actual_cash": float(first.cash_actual), "reference_cash": float(first.cash_reference),
            "cash_delta": float(first.cash_actual - first.cash_reference),
            "actual_stock_mv": float(first.stock_mv),
            "reference_market_value": float(first.market_value),
            "market_value_delta": float(first.stock_mv - first.market_value),
        }

    # 2) First BUY divergence by emitted sequence.
    ao = actual_orders[actual_orders.side.eq("BUY")].reset_index(drop=True)
    go = reference_orders.reset_index(drop=True)
    order_diff = None
    n = max(len(ao), len(go))
    for i in range(n):
        if i >= len(ao) or i >= len(go):
            order_diff = {
                "index": i, "type": "sequence_length",
                "actual": None if i >= len(ao) else ao.iloc[i].to_dict(),
                "reference": None if i >= len(go) else go.iloc[i].to_dict(),
            }
            break
        a, g = ao.iloc[i], go.iloc[i]
        ak = (int(a.decision_date), str(a.strategy), str(a.code))
        gk = (int(g.decision_date), str(g.strategy), str(g.code))
        if ak != gk:
            order_diff = {"index": i, "type": "identity", "actual": ak, "reference": gk}
            break
        for fld, av, gv, tol in [
            ("execute_date", int(a.execute_date), int(g.execute_date), 0),
            ("limit", float(a.order_price), float(g.limit), 1e-8),
            ("shares", int(a.shares), int(g.shares), 0),
            ("target_cash", float(a.target_cash), float(g.target_cash), 0.05),
        ]:
            mismatch = (av != gv) if tol == 0 else abs(av - gv) > tol
            if mismatch:
                order_diff = {
                    "index": i, "key": ak, "type": fld,
                    "actual": av, "reference": gv, "delta": av - gv,
                }
                break
        if order_diff:
            break
        exp_filled = str(g.status).upper() == "FILLED"
        if bool(a.filled) != exp_filled:
            order_diff = {
                "index": i, "key": ak, "type": "fill_status",
                "actual": bool(a.filled), "reference": exp_filled,
            }
            break
        if exp_filled and pd.notna(g.fill_price) and abs(float(a.fill_price) - float(g.fill_price)) > 1e-8:
            order_diff = {
                "index": i, "key": ak, "type": "fill_price",
                "actual": float(a.fill_price), "reference": float(g.fill_price),
                "delta": float(a.fill_price) - float(g.fill_price),
            }
            break

    # 3) First exit-date divergence and exact T-known source rows around it.
    amap = {
        (str(r.strategy), str(r.code), int(r.entry_date)): r
        for r in actual_trades.itertuples(index=False)
    }
    exit_diff = None
    exit_source = []
    first_exit_explanation = None
    for i, g in enumerate(reference_exits.itertuples(index=False)):
        key = (str(g.strategy), str(g.code), int(g.entry_date))
        a = amap.get(key)
        if a is None or int(a.exit_date) != int(g.exit_date):
            exit_diff = {
                "index": i, "key": key,
                "reference_sell_decision_date": int(g.sell_decision_date),
                "reference_exit_date": int(g.exit_date),
                "actual_exit_date": None if a is None else int(a.exit_date),
                "reference_exit_price": float(g.exit_price),
                "actual_exit_price": None if a is None else float(a.exit_price),
            }
            if a is not None and key[0] == "R05":
                er = row_at(feat_idx, key[2], key[1])
                entry_price = float(a.entry_price)
                entry_factor = float(er.aclose / er.close) if er is not None and er.close else 1.0
                entry_adj = entry_price * entry_factor
                hard = entry_adj * 0.90
                dates = [d for d in eval_dates if key[2] <= int(d) <= int(a.exit_date)]
                for d in dates:
                    r = row_at(feat_idx, int(d), key[1])
                    if r is None:
                        continue
                    factor = float(r.aclose / r.close) if r.close else 1.0
                    low_adj = float(r.low) * factor
                    exit_source.append({
                        "date": int(d), "open": float(r.open), "low": float(r.low),
                        "close": float(r.close), "aclose": float(r.aclose),
                        "factor": factor, "low_adj": low_adj,
                        "hard_threshold_adj": hard,
                        "close_hard_trigger": bool(float(r.aclose) <= hard + 1e-12),
                        "intraday_low_hard_trigger": bool(low_adj <= hard + 1e-12),
                    })
                ref_day = row_at(feat_idx, int(g.exit_date), key[1])
                if ref_day is not None:
                    formal_proceeds = float(g.exit_price) * int(g.shares) * (1.0 - bt.SELL_FEE - bt.SELL_TAX)
                    held_close_mv = float(ref_day.close) * int(g.shares)
                    first_exit_explanation = {
                        "reference_exit_proceeds_recomputed": formal_proceeds,
                        "value_if_position_held_to_same_day_close": held_close_mv,
                        "same_day_nav_lift_from_early_exit": formal_proceeds - held_close_mv,
                    }
            break

    report = {
        "reference_is_engine_input": False,
        "reference_nav_source": "mechanically rebuilt post-run from formal 410-order/241-exit ledger; no historical NAV answer file read",
        "reference_orders": int(len(reference_orders)),
        "reference_exits": int(len(reference_exits)),
        "first_nav_divergence": first_nav,
        "first_buy_divergence": order_diff,
        "first_exit_divergence": exit_diff,
        "first_exit_source_rows": exit_source,
        "first_exit_nav_explanation": first_exit_explanation,
    }

    p = OUT / "first_divergence_diagnosis.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
