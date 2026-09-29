"""Prospective, auditable paper account for the locked R10 forward scanner.

The seed is the 2026-09-29 confirmed-position *model snapshot*.  It is not a
broker reconciliation.  Only orders emitted by this account are subsequently
settled; older scanner reports are never retrospectively assumed filled.
"""
import json
import math
from pathlib import Path

import numpy as np

SEED_DATE = "2026-09-29"
BUY_FEE, SELL_FEE, SELL_TAX = 0.000855, 0.000855, 0.003


def tick(price):
    return 0.01 if price < 10 else 0.05 if price < 50 else 0.1 if price < 100 else 0.5 if price < 500 else 1.0 if price < 1000 else 5.0


def floor_tick(price):
    return round(math.floor((price + 1e-10) / tick(price)) * tick(price), 4)


def ceil_tick(price):
    return round(math.ceil((price - 1e-10) / tick(price)) * tick(price), 4)


def buy_fill(open_price, low, limit):
    if open_price <= limit:
        return min(ceil_tick(open_price * 1.005), limit)
    return limit if low <= limit else None


def _previous(repo, trade_date):
    files = sorted((Path(repo) / "forward_runtime").glob("????-??-??/paper_state.json"))
    older = [p for p in files if p.parent.name < trade_date]
    return (older[-1], json.loads(older[-1].read_text(encoding="utf-8"))) if older else (None, None)


def _seed(trade_date, confirmed, baseline):
    if trade_date != SEED_DATE:
        raise RuntimeError(f"PAPER_GATE_FAIL no prior paper state for {trade_date}; seed requires {SEED_DATE}")
    positions = {}
    for p in confirmed.get("confirmed_forward_positions", []):
        code = str(p["code"])
        positions[code] = {**p, "code": code, "shares": int(p["shares"]),
                           "entry_shares": int(p["shares"]), "corp_income": 0.0}
    if {str(p["code"]) for p in baseline["positions"]} != set(positions):
        raise RuntimeError("PAPER_GATE_FAIL seed differs from confirmed model positions")
    return {"schema_version": 1, "account": "PAPER_SIMULATION_NOT_BROKER",
            "trade_date": trade_date, "cash": float(baseline["cash"]),
            "hwm": float(baseline["hwm"]), "positions": positions,
            "pending_orders": [], "fills": [], "seed": "confirmed model snapshot on 2026-09-29"}


def advance(repo, trade_date, feat, events, confirmed, baseline,
            compute_live_state, next_session):
    prev_path, prev = _previous(repo, trade_date)
    if prev is None:
        return _seed(trade_date, confirmed, baseline)
    if prev.get("account") != "PAPER_SIMULATION_NOT_BROKER":
        raise RuntimeError("PAPER_GATE_FAIL invalid account provenance")
    if next_session(prev["trade_date"]) != trade_date:
        raise RuntimeError(f"PAPER_GATE_FAIL missing sequential session: {prev['trade_date']} -> {trade_date}")
    if any(o["scheduled_date"] != trade_date for o in prev["pending_orders"]):
        raise RuntimeError("PAPER_GATE_FAIL pending order date mismatch")
    today = feat[feat.date == int(trade_date.replace("-", ""))].set_index("code")
    positions = {str(k): dict(v) for k, v in prev["positions"].items()}
    cash = float(prev["cash"])
    fills = []

    # Official ex-date adjustment is applied before the open, as in the locked engine.
    daily_events = events[events.date == int(trade_date.replace("-", ""))]
    for _, e in daily_events.iterrows():
        code = str(e.code)
        if code not in positions:
            continue
        if code not in today.index:
            raise RuntimeError(f"PAPER_GATE_FAIL held event has no T price: {code}")
        p = positions[code]
        old = int(p["shares"])
        dividend = e.get("cash_dividend_per_share", np.nan)
        stock = e.get("stock_shares_per_1000", np.nan)
        factor = 1 + float(stock) / 1000 if np.isfinite(stock) else 1.0
        if "RIGHT" in str(e.get("event_type", "")) and str(e.get("market", "")) == "TWSE" and not np.isfinite(stock):
            raise RuntimeError(f"PAPER_GATE_FAIL rights factor missing: {code}")
        exact = old * factor
        p["shares"] = int(math.floor(exact + 1e-10))
        reference = e.get("reference_price", np.nan)
        if exact != p["shares"] and not np.isfinite(reference):
            raise RuntimeError(f"PAPER_GATE_FAIL cash-in-lieu reference missing: {code}")
        credit = (old * float(dividend) if np.isfinite(dividend) else 0.0) + (exact - p["shares"]) * (float(reference) if np.isfinite(reference) else 0.0)
        cash += credit
        p["corp_income"] = float(p.get("corp_income", 0.0)) + credit
        fills.append({"event": "CORPORATE_ACTION", "code": code, "date": trade_date,
                      "old_shares": old, "new_shares": p["shares"], "cash_credit": credit})

    for order in prev["pending_orders"]:
        o = dict(order)
        code = str(o["code"])
        o.update({"event": "ORDER", "fill_date": trade_date, "status": None,
                  "raw_fill_price": None, "gross_value": None, "fee": None,
                  "tax": None, "net_cash": None})
        if code not in today.index:
            o["status"] = "CANCELED_NO_T1_PRICE"
        elif o["side"] == "SELL":
            p = positions.get(code)
            if p is None:
                o["status"] = "CANCELED_NO_POSITION"
            else:
                price = floor_tick(float(today.loc[code, "open"]) * 0.995)
                shares = int(p["shares"])
                gross = price * shares
                fee, tax = gross * SELL_FEE, gross * SELL_TAX
                proceeds = gross - fee - tax
                cash += proceeds
                del positions[code]
                o.update(status="FILLED", shares=shares, raw_fill_price=price,
                         gross_value=gross, fee=fee, tax=tax, net_cash=proceeds)
        elif o["side"] == "BUY":
            price = buy_fill(float(today.loc[code, "open"]), float(today.loc[code, "low"]), float(o["limit_price"]))
            if price is None:
                o["status"] = "UNFILLED_LIMIT_NOT_TOUCHED"
            else:
                shares = int(o["shares"])
                gross = price * shares
                fee = gross * BUY_FEE
                cost = gross + fee
                if cost > cash + 1e-7 or code in positions:
                    o["status"] = "REJECTED_CASH_OR_DUPLICATE"
                else:
                    cash -= cost
                    positions[code] = {"code": code, "name": o.get("name", ""),
                                       "strategy": o["strategy"], "shares": shares,
                                       "entry_shares": shares, "entry_date": trade_date,
                                       "entry_raw_price": price, "corp_income": 0.0}
                    o.update(status="FILLED", raw_fill_price=price, gross_value=gross,
                             fee=fee, tax=0.0, net_cash=-cost)
        else:
            raise RuntimeError(f"PAPER_GATE_FAIL unknown side {o['side']}")
        fills.append(o)

    position_input = [{**p, "shares": p["entry_shares"]} for p in positions.values()]
    calculated = compute_live_state({"initial_capital": confirmed["initial_capital"],
                                     "confirmed_forward_positions": position_input}, feat, events, trade_date)
    for p in calculated["positions"]:
        if int(p["shares"]) != int(positions[p["code"]]["shares"]):
            raise RuntimeError(f"PAPER_GATE_FAIL corporate shares mismatch: {p['code']}")
    market_value = sum(float(p["market_value"]) for p in calculated["positions"])
    nav = cash + market_value
    hwm = max(float(prev["hwm"]), nav)
    return {"schema_version": 1, "account": "PAPER_SIMULATION_NOT_BROKER",
            "trade_date": trade_date, "cash": cash, "hwm": hwm,
            "positions": positions, "pending_orders": [], "fills": fills,
            "previous_state": str(prev_path.relative_to(repo))}


def paper_live(paper, feat, events, confirmed, compute_live_state):
    positions = [{**p, "shares": p["entry_shares"]} for p in paper["positions"].values()]
    calculated = compute_live_state({"initial_capital": confirmed["initial_capital"],
                                     "confirmed_forward_positions": positions},
                                    feat, events, paper["trade_date"])
    mv = sum(float(p["market_value"]) for p in calculated["positions"])
    cash, hwm = float(paper["cash"]), float(paper["hwm"])
    nav = cash + mv
    if nav <= 0:
        raise RuntimeError("PAPER_GATE_FAIL nonpositive NAV")
    if abs(nav - calculated["nav"]) > 1e-6 and paper["trade_date"] == SEED_DATE:
        raise RuntimeError("PAPER_GATE_FAIL seed NAV mismatch")
    return {"cash": cash, "market_value": mv, "nav": nav,
            "hwm": hwm, "dd": nav / hwm - 1, "exposure": mv / nav,
            "positions": calculated["positions"]}


def store_orders(paper, buys, sells):
    paper["pending_orders"] = [dict(o) for o in sells + buys]
    return paper
