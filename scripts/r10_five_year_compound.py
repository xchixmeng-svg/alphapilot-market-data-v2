#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaPilot R10-MAX 2008-now five-year capital-path test.

Rules of this research path:
- Start with NT$1,000,000 in 2008.
- Run locked FULL R10 causally.
- Every completed five-calendar-year block is fully liquidated on T+1 using
  the locked 0.5% adverse sell-slippage + fee/tax convention.
- If settled NAV > block starting capital, withdraw 50% of the profit.
  The other 50% plus original capital starts the next block.
- If the block loses money, withdraw nothing and carry all remaining capital.
- The current incomplete block (2023 -> last completed market date) is marked
  to market only and does NOT withdraw profit yet.

This script intentionally does not scale precomputed return percentages. Each
block is re-simulated with its actual path-dependent starting capital so board-
lot sizing, high-price odd-lot exceptions, cash, liquidity, and exposure limits
remain realistic.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import r10_full_battery as full

bt = full.bt
ROOT = Path(__file__).resolve().parent.parent
OUTROOT = ROOT / "stress_results" / "five_year_compound"
CACHE = bt.CACHE_ROOT / "five_year_compound"
CACHE.mkdir(parents=True, exist_ok=True)

# Reuse the rolling scanner's 2026 weekly+official builder because an unfinished
# calendar year does not yet have a yearly release archive.
_ORIG_LOAD_YEAR = bt.load_year

def _last_complete_trade_date() -> date:
    state = ROOT / "data" / "r10_state.json"
    if state.exists():
        try:
            s = json.loads(state.read_text(encoding="utf-8"))
            x = s.get("last_scan_date")
            if x:
                return date.fromisoformat(x)
        except Exception:
            pass
    tf = ROOT / ".alphapilot_trade_date"
    if tf.exists():
        try: return date.fromisoformat(tf.read_text(encoding="utf-8").strip())
        except Exception: pass
    raise RuntimeError("cannot determine last completed market date from repository")

LAST_TRADE = _last_complete_trade_date()

def _load_year_extended(year: int) -> pd.DataFrame:
    if year != LAST_TRADE.year:
        return _ORIG_LOAD_YEAR(year)
    # Only use rolling history for the current unfinished year. If a completed
    # yearly archive later exists, the ordinary loader remains preferred.
    try:
        return _ORIG_LOAD_YEAR(year)
    except Exception:
        q = bt.core.history(LAST_TRADE)
        q = bt.normalize_ohlcv(q)
        return q

bt.load_year = _load_year_extended

# Cache expensive official institutional history per exact interval.
_BASE_INST_DISPATCH = bt.fetch_institutional

def _cached_inst(feat, eval_start: int, eval_end: int):
    # Pre-T86 FULL R10 must go through the already calibrated FinMind fallback.
    if full._CURRENT_SCENARIO in {"gfc2008", "euro2011"}:
        return _BASE_INST_DISPATCH(feat, eval_start, eval_end)
    p = CACHE / f"official_inst_{eval_start}_{eval_end}.csv"
    if p.exists():
        q = pd.read_csv(p, dtype={"code": str})
        for c in ["date","foreign_net","trust_net","dealer_net","Foreign3D","Foreign10D","Trust5D"]:
            if c in q.columns: q[c] = pd.to_numeric(q[c], errors="coerce")
        q["date"] = q["date"].astype(int)
        q["code"] = q["code"].astype(str).str.zfill(4)
        bt.log(f"[INST CACHE] {p.name}: {len(q)} rows")
        return q
    q = _BASE_INST_DISPATCH(feat, eval_start, eval_end)
    q.to_csv(p, index=False, encoding="utf-8-sig")
    return q

bt.fetch_institutional = _cached_inst

CYCLES = {
    "2008_2012": {"warmup": 2007, "start": "2008-01-02", "end": "2012-12-31", "finmind": True, "complete": True},
    "2013_2017": {"warmup": 2012, "start": "2013-01-01", "end": "2017-12-31", "finmind": False, "complete": True},
    "2018_2022": {"warmup": 2017, "start": "2018-01-01", "end": "2022-12-31", "finmind": False, "complete": True},
    "2023_NOW":  {"warmup": 2022, "start": "2023-01-01", "end": LAST_TRADE.isoformat(), "finmind": False, "complete": False},
}


def _years_between(a: int, b: int) -> list[int]:
    return list(range(a, b + 1))


def _scenario(cycle: str, actual_start: str) -> dict:
    c = CYCLES[cycle]
    end = date.fromisoformat(c["end"])
    start = date.fromisoformat(actual_start)
    # Include warm-up year through evaluation end year.
    return {
        "label": f"R10 five-year capital path {cycle}",
        "warmup_start": f"{c['warmup']}-01-01",
        "eval_start": start.isoformat(),
        "eval_end": end.isoformat(),
        "years": _years_between(c["warmup"], end.year),
        "r05": True,
        "mode": "FULL_R10_FIVE_YEAR_CAPITAL_PATH",
        "reason": "Locked FULL R10. Starting capital is inherited from the prior cycle after actual settlement and 50% profit withdrawal.",
    }


def _open_positions(outdir: Path) -> list[dict]:
    op = outdir / "orders.csv"
    tr = outdir / "trades.csv"
    if not op.exists(): return []
    o = pd.read_csv(op, dtype={"code": str})
    if o.empty: return []
    filled = o[(o["side"].astype(str) == "BUY") & (o["filled"].astype(str).str.lower().isin(["true","1"]))].copy()
    if filled.empty: return []
    filled["execute_date"] = pd.to_numeric(filled["execute_date"], errors="coerce").astype("Int64")
    filled["shares"] = pd.to_numeric(filled["shares"], errors="coerce").astype("Int64")
    closed = Counter()
    if tr.exists():
        t = pd.read_csv(tr, dtype={"code": str})
        if not t.empty:
            for r in t.itertuples(index=False):
                closed[(str(r.strategy), str(r.code).zfill(4), int(r.entry_date), int(r.shares))] += 1
    opens = []
    for r in filled.itertuples(index=False):
        key = (str(r.strategy), str(r.code).zfill(4), int(r.execute_date), int(r.shares))
        if closed[key] > 0:
            closed[key] -= 1
            continue
        opens.append({"strategy": key[0], "code": key[1], "entry_date": key[2], "shares": key[3], "name": str(getattr(r,"name",""))})
    return opens


def _settle_completed_cycle(cycle: str, initial_capital: float, result: dict) -> dict:
    c = CYCLES[cycle]
    outdir = bt.OUT_ROOT / "latest" / cycle
    nav = pd.read_csv(outdir / "daily_nav.csv")
    cash_end = float(nav.iloc[-1]["cash"])
    opens = _open_positions(outdir)
    eval_end = bt.intdate(c["end"])
    end_year = date.fromisoformat(c["end"]).year
    # Need the following calendar year for T+1 liquidation. Load the whole year
    # so a temporary suspension is handled by the first genuinely tradable open.
    nxt = _load_year_extended(end_year + 1)
    nxt = nxt[pd.to_numeric(nxt["date"], errors="coerce") > eval_end].copy()
    if nxt.empty:
        raise RuntimeError(f"{cycle}: no next-year data available for settlement")
    proceeds = 0.0
    settle_rows = []
    last_sell_date = int(nxt["date"].min())
    for p in opens:
        q = nxt[nxt["code"].astype(str).str.zfill(4).eq(p["code"])].sort_values("date")
        q = q[pd.to_numeric(q["open"], errors="coerce") > 0]
        if q.empty:
            raise RuntimeError(f"{cycle}: cannot liquidate open position {p['strategy']} {p['code']} after cycle end")
        r = q.iloc[0]
        di = int(r["date"]); opx = float(r["open"])
        sell = bt.legal_sell_price(opx)
        gross = sell * int(p["shares"])
        net = gross * (1.0 - bt.SELL_FEE - bt.SELL_TAX)
        proceeds += net; last_sell_date = max(last_sell_date, di)
        settle_rows.append({**p,"settle_date":di,"next_open":opx,"sell_price_after_0p5_slippage":sell,"net_proceeds":net})
    settled_nav = cash_end + proceeds
    profit = settled_nav - initial_capital
    withdrawal = max(0.0, profit * 0.50)
    next_capital = settled_nav - withdrawal
    # Next cycle can start with cash on the day the last settlement sale clears
    # intraday; signals formed at that day's close can only execute T+1.
    next_start_date = datetime.strptime(str(last_sell_date), "%Y%m%d").date().isoformat()
    pd.DataFrame(settle_rows).to_csv(outdir / "cycle_settlement.csv", index=False, encoding="utf-8-sig")
    return {
        "cycle": cycle,
        "initial_capital": initial_capital,
        "mark_to_market_end_nav": float(result["end_nav"]),
        "settled_nav_after_forced_T1_liquidation": settled_nav,
        "cycle_profit_after_settlement": profit,
        "withdrawal_50pct_profit": withdrawal,
        "next_cycle_capital": next_capital,
        "next_cycle_start_date": next_start_date,
        "open_positions_forced_out": len(opens),
        "max_dd": float(result["max_dd"]),
        "completed_trades": int(result["completed_trades"]),
        "avg_exposure": float(result["avg_exposure"]),
        "max_exposure": float(result["max_exposure"]),
        "min_cash": float(result["min_cash"]),
        "annual_returns": result.get("annual_returns", {}),
        "status": "PASS",
    }


def run_cycle(cycle: str, initial_capital: float, actual_start: str) -> dict:
    c = CYCLES[cycle]
    if initial_capital <= 0: raise RuntimeError("initial capital must be positive")
    bt.INITIAL_CAPITAL = float(initial_capital)
    # Activates calibrated FinMind dispatch only for the pre-T86 block.
    full._CURRENT_SCENARIO = "gfc2008" if c["finmind"] else cycle
    cfg = _scenario(cycle, actual_start)
    bt.SCENARIOS[cycle] = cfg
    result = bt.simulate(cycle, cfg)
    if not result.get("r05_enabled") or str(result.get("mode","")).startswith("PARTIAL"):
        raise RuntimeError(f"{cycle}: FULL R10 requirement failed")
    if c["finmind"]:
        audit = dict(full._LAST_FINMIND_AUDIT)
        if audit.get("day_coverage", 0) < 1.0 or audit.get("pair_coverage", 0) < 0.985:
            raise RuntimeError(f"{cycle}: FinMind FULL-R10 coverage gate failed: {audit}")
        result["institutional_audit"] = audit
    if c["complete"]:
        summary = _settle_completed_cycle(cycle, initial_capital, result)
    else:
        summary = {
            "cycle": cycle,
            "initial_capital": initial_capital,
            "actual_start": actual_start,
            "through": c["end"],
            "current_account_nav": float(result["end_nav"]),
            "unrealized_cycle_profit": float(result["end_nav"]) - initial_capital,
            "withdrawal_50pct_profit": 0.0,
            "withdrawal_reason": "Current cycle has not completed five years; no withdrawal yet.",
            "max_dd": float(result["max_dd"]),
            "completed_trades": int(result["completed_trades"]),
            "avg_exposure": float(result["avg_exposure"]),
            "max_exposure": float(result["max_exposure"]),
            "min_cash": float(result["min_cash"]),
            "annual_returns": result.get("annual_returns", {}),
            "status": "PASS",
        }
    summary["engine_version"] = bt.VERSION
    summary["full_r10"] = True
    summary["actual_start"] = actual_start
    out = OUTROOT / cycle
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True, choices=list(CYCLES))
    ap.add_argument("--initial-capital", required=True, type=float)
    ap.add_argument("--actual-start", default="")
    args = ap.parse_args()
    actual_start = args.actual_start or CYCLES[args.cycle]["start"]
    run_cycle(args.cycle, args.initial_capital, actual_start)

if __name__ == "__main__": main()
