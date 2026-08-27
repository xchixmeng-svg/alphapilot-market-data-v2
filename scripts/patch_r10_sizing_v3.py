#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministically repair R10 sizing semantics in backtest_r10_stress.py.

Locked rule being repaired:
- R7 base target = 22% of Portfolio NAV; R0.5 = 20% of Portfolio NAV.
- Strategic target is clipped by DD/single-name/global-exposure constraints.
- Cash availability may reduce executable quantity, but MUST NOT change
  board-lot vs odd-lot eligibility.
- Odd lots are allowed only when ONE board lot itself exceeds the strategic
  target. Residual cash must never turn an otherwise board-lot order into
  a tiny odd-lot order.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "backtest_r10_stress.py"
s = P.read_text(encoding="utf-8")

old_size = '''def size_shares(target_cash: float, limit: float, avg_vol20: float, cash_available: float) -> Tuple[int, str]:
    if target_cash <= 0 or limit <= 0 or cash_available <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    max_by_cash = min(target_cash, cash_available)
    one_lot_cost = limit * 1000 * (1 + BUY_FEE)
    if max_by_cash >= one_lot_cost:
        shares = int(math.floor(max_by_cash / (limit * (1 + BUY_FEE)) / 1000) * 1000)
        mode = "BOARD_LOT"
        liq = int(math.floor(float(avg_vol20) * ADV_CAP / 1000) * 1000)
    else:
        shares = int(math.floor(max_by_cash / (limit * (1 + BUY_FEE))))
        mode = "HIGH_PRICE_ODDLOT"
        liq = int(math.floor(float(avg_vol20) * ADV_CAP))
    shares = min(shares, max(0, liq))
    if mode == "BOARD_LOT": shares = (shares // 1000) * 1000
    return max(0, int(shares)), mode
'''

new_size = '''def size_shares(target_cash: float, limit: float, avg_vol20: float, cash_available: float) -> Tuple[int, str]:
    if target_cash <= 0 or limit <= 0 or cash_available <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    # IMPORTANT: board-lot eligibility is determined by the STRATEGIC target,
    # not by residual cash. Cash may reduce quantity, but cannot convert a
    # normal board-lot order into an odd-lot order.
    one_lot_cost = limit * 1000 * (1 + BUY_FEE)
    executable_cash = min(target_cash, cash_available)
    if target_cash + 1e-9 >= one_lot_cost:
        mode = "BOARD_LOT"
        shares = int(math.floor(executable_cash / (limit * (1 + BUY_FEE)) / 1000) * 1000)
        liq = int(math.floor(float(avg_vol20) * ADV_CAP / 1000) * 1000)
        shares = min(shares, max(0, liq))
        shares = (shares // 1000) * 1000
        return max(0, int(shares)), mode

    # Odd lots are legal only when one board lot itself exceeds the strategic
    # target. This is the documented high-price exception.
    mode = "HIGH_PRICE_ODDLOT"
    shares = int(math.floor(executable_cash / (limit * (1 + BUY_FEE))))
    liq = int(math.floor(float(avg_vol20) * ADV_CAP))
    shares = min(shares, max(0, liq))
    return max(0, int(shares)), mode
'''

old_target = '''                target=nav*base_pct*dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global,rem_cash)
                if target<=0:return
                shares,_=size_shares(target,limit,float(row.avgvol20),rem_cash)
                if shares<=0:return
                reserve=shares*limit*(1+BUY_FEE); notional=shares*limit
'''

new_target = '''                # Keep the strategic 22%/20% target separate from cash.
                # Cash shortage may reduce executable shares, but it must not
                # redefine the strategy target or odd-lot eligibility.
                target=nav*base_pct*dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global)
                if target<=0:return
                shares,_=size_shares(target,limit,float(row.avgvol20),rem_cash)
                if shares<=0:return
                reserve=shares*limit*(1+BUY_FEE); notional=shares*limit
'''

if old_size not in s:
    raise SystemExit("size_shares source block not found; refusing blind patch")
if old_target not in s:
    raise SystemExit("try_order target source block not found; refusing blind patch")

s = s.replace(old_size, new_size, 1).replace(old_target, new_target, 1)
P.write_text(s, encoding="utf-8")
print("PATCHED", P)
