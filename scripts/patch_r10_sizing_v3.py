#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair R10 sizing/common-pool semantics to the user-confirmed locked rules.

Locked rules:
- R7 new position = 22% of current Portfolio NAV.
- R0.5 new position = 20% of current Portfolio NAV.
- Existing DD / single-name / total-exposure / R7-regime / ADV controls may
  constrain the strategic target, but current residual cash must NOT redefine
  the T-day target quantity.
- Normal-price stocks are BOARD LOT ONLY and use the nearest whole 1,000-share
  lot to the strategic cash target; if the computed quantity is below one lot,
  it is still one board lot (subject to hard liquidity availability).
- Odd lots are allowed only for the high-price exception: one board lot itself
  costs more than the strategy's unthrottled base allocation (22% or 20% NAV).
- T+1 execution already processes sells before buys. Actual sale proceeds enter
  the single common cash pool first; buy affordability is checked only then.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "backtest_r10_stress.py"
s = P.read_text(encoding="utf-8")

old_size = '''def size_shares(target_cash: float, limit: float, avg_vol20: float, cash_available: float) -> Tuple[int, str]:
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

new_size = '''def size_shares(target_cash: float, base_target_cash: float, limit: float, avg_vol20: float) -> Tuple[int, str]:
    if target_cash <= 0 or base_target_cash <= 0 or limit <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    per_share_cost = limit * (1.0 + BUY_FEE)
    one_lot_cost = per_share_cost * 1000.0

    # HIGH-PRICE EXCEPTION ONLY: one full board lot is already larger than the
    # strategy's normal 22%/20% base allocation. Only here may odd lots exist.
    if one_lot_cost > base_target_cash + 1e-9:
        mode = "HIGH_PRICE_ODDLOT"
        shares = max(1, int(math.floor(target_cash / per_share_cost + 0.5)))
        liq = int(math.floor(float(avg_vol20) * ADV_CAP))
        shares = min(shares, max(0, liq))
        return max(0, int(shares)), mode

    # Normal-price stocks: whole board lots only. Use the nearest board lot;
    # even a sub-lot computed target becomes one full lot, never 578/867 shares.
    mode = "BOARD_LOT"
    raw_lots = target_cash / one_lot_cost
    lots = max(1, int(math.floor(raw_lots + 0.5)))
    liq_lots = int(math.floor(float(avg_vol20) * ADV_CAP / 1000.0))
    lots = min(lots, max(0, liq_lots))
    return max(0, lots * 1000), mode
'''

old_target = '''                current_code=value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0.0)
                rem_single=nav*MAX_SINGLE-current_code; rem_global=nav*MAX_TOTAL-base_exposure-reserved_exposure; rem_cash=cash-reserved_cash
                # Keep the strategic 22%/20% target separate from cash.
                # Cash shortage may reduce executable shares, but it must not
                # redefine the strategy target or odd-lot eligibility.
                target=nav*base_pct*dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global)
                if target<=0:return
                shares,_=size_shares(target,limit,float(row.avgvol20),rem_cash)
                if shares<=0:return
                reserve=shares*limit*(1+BUY_FEE); notional=shares*limit
                if reserve>rem_cash+1e-6:return
'''

new_target = '''                current_code=value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0.0)
                rem_single=nav*MAX_SINGLE-current_code; rem_global=nav*MAX_TOTAL-base_exposure-reserved_exposure

                # Exact locked allocation: R7=22% NAV, R0.5=20% NAV. This base
                # amount also determines whether the stock qualifies for the
                # high-price odd-lot exception. Current T-day cash is NOT used to
                # shrink the precommitted quantity; T+1 sells execute first and
                # their actual proceeds enter the common pool before buys.
                base_target=nav*base_pct
                target=base_target*dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global)
                if target<=0:return
                shares,_=size_shares(target,base_target,limit,float(row.avgvol20))
                if shares<=0:return
                reserve=shares*limit*(1+BUY_FEE); notional=shares*limit
'''

if old_size not in s:
    raise SystemExit("current size_shares block not found; refusing blind patch")
if old_target not in s:
    raise SystemExit("current try_order sizing block not found; refusing blind patch")

s = s.replace(old_size, new_size, 1).replace(old_target, new_target, 1)
P.write_text(s, encoding="utf-8")
print("PATCHED", P)
