#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / "r10_true_validation.py"
s = p.read_text(encoding="utf-8")

old = '''            reserved_exposure = 0.0
            reserved_r7 = 0.0
            reserved_code: dict[str, float] = {}

            def try_order(strategy: str, row, rank: int):
                nonlocal reserved_exposure, reserved_r7, codes_after
'''
new = '''            reserved_exposure = 0.0
            reserved_r7 = 0.0
            reserved_cash = 0.0
            reserved_code: dict[str, float] = {}

            def try_order(strategy: str, row, rank: int):
                nonlocal reserved_exposure, reserved_r7, reserved_cash, codes_after
'''
if old not in s:
    raise SystemExit("cash reservation declaration anchor not found")
s = s.replace(old, new, 1)

old = '''                rem_single = nav * bt.MAX_SINGLE - current_code
                rem_global = nav * bt.MAX_TOTAL - base_exposure - reserved_exposure
                base_target = nav * base_pct
                target = base_target * bt.dd_multiplier(dd)
                if strategy == "R7":
                    r7_cap = nav * float(r7_state["exposure"])
                    target = min(target, r7_cap - base_r7 - reserved_r7)
                target = min(target, rem_single, rem_global)
                if target <= 0:
                    return

                shares, share_mode = size_shares_true(target, limit, float(row.avgvol20))
                if shares <= 0:
                    return
                notional = shares * limit
                reserve = notional * (1.0 + bt.BUY_FEE)
'''
new = '''                rem_single = nav * bt.MAX_SINGLE - current_code
                rem_global = nav * bt.MAX_TOTAL - base_exposure - reserved_exposure
                # Formal R10 cash semantics: T-day orders precommit and reserve
                # T-day known cash. Planned T+1 sells may release tomorrow's
                # slot/projected exposure, but their unknown proceeds cannot
                # finance an order decided the previous night.
                rem_cash = max(0.0, cash - reserved_cash)
                base_target = nav * base_pct
                target = base_target * bt.dd_multiplier(dd)
                if strategy == "R7":
                    r7_cap = nav * float(r7_state["exposure"])
                    target = min(target, r7_cap - base_r7 - reserved_r7)
                target = min(target, rem_single, rem_global, rem_cash)
                if target <= 0:
                    return

                shares, share_mode = size_shares_true(target, limit, float(row.avgvol20))
                if shares <= 0:
                    return
                notional = shares * limit
                reserve = notional * (1.0 + bt.BUY_FEE)
                # Quantity rule floors notional, while cash safety includes fee.
                # If fees push the reservation over remaining cash, reduce only
                # downward; never borrow future T+1 sale proceeds.
                if reserve > rem_cash + 1e-6:
                    if share_mode == "BOARD_LOT":
                        lots = int(math.floor(rem_cash / (limit * (1.0 + bt.BUY_FEE) * 1000.0) + 1e-12))
                        shares = min(shares, max(0, lots) * 1000)
                    else:
                        affordable = int(math.floor(rem_cash / (limit * (1.0 + bt.BUY_FEE)) + 1e-12))
                        shares = min(shares, max(0, affordable))
                    if shares <= 0:
                        return
                    notional = shares * limit
                    reserve = notional * (1.0 + bt.BUY_FEE)
                if reserve > rem_cash + 1e-6:
                    raise RuntimeError(f"T-day cash reservation breach {di} {strategy} {code}: reserve={reserve} rem_cash={rem_cash}")
'''
if old not in s:
    raise SystemExit("cash reservation sizing anchor not found")
s = s.replace(old, new, 1)

old = '''                created.append(bt.BuyOrder(
                    di, exdate, strategy, code, name0, limit, shares, target, reserve, rank
                ))
                reserved_exposure += notional
'''
new = '''                created.append(bt.BuyOrder(
                    di, exdate, strategy, code, name0, limit, shares, target, reserve, rank
                ))
                reserved_cash += reserve
                reserved_exposure += notional
'''
if old not in s:
    raise SystemExit("cash reservation accumulation anchor not found")
s = s.replace(old, new, 1)

# Provenance: make the cash rule explicit in every result artifact.
s = s.replace(
    '"causality": "T-close decision; T+1 sells before T+1 buys; raw T+1 Open/Low execution only",',
    '"causality": "T-close decision; T-day buy cash precommitted; T+1 sells before T+1 buys; raw T+1 Open/Low execution only",',
    1,
)

p.write_text(s, encoding="utf-8")
print("PATCHED", p)
