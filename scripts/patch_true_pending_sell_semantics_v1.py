#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / "r10_true_validation.py"
s = p.read_text(encoding="utf-8")

# Formal R10 semantics inferred directly from the locked 410-order ledger:
# - a preplanned T+1 sell may free a tomorrow slot;
# - it does NOT erase the position from T-day current exposure / single-name MV.
old = '''            codes_after = {p.code for k, p in positions.items() if k not in sell_keys}
            base_exposure = bt.value_of(positions, feat_idx, di, exclude=sell_keys)
            base_r7 = bt.value_of(positions, feat_idx, di, strategy="R7", exclude=sell_keys)
'''
new = '''            # A scheduled T+1 sell may release a T+1 slot, but its T-day
            # market value still occupies T-day exposure. This matches the
            # formal order ledger's target_cash construction.
            codes_after = {p.code for k, p in positions.items() if k not in sell_keys}
            base_exposure = bt.value_of(positions, feat_idx, di)
            base_r7 = bt.value_of(positions, feat_idx, di, strategy="R7")
'''
if old not in s:
    raise SystemExit("pending-sell exposure anchor not found")
s = s.replace(old, new, 1)

old = '''                k = bt.pos_key(strategy, code)
                if k in positions and k not in sell_keys:
                    return
'''
new = '''                k = bt.pos_key(strategy, code)
                if k in positions and k not in sell_keys:
                    return
                # R7 manual explicitly forbids an entry in a stock that is
                # simultaneously prepared for exit. Do not generalize this
                # restriction to R0.5 without a documented source rule.
                if strategy == "R7" and k in sell_keys:
                    return
'''
if old not in s:
    raise SystemExit("R7 same-day reentry anchor not found")
s = s.replace(old, new, 1)

# Per-code cap must also include T-day MV of a position scheduled to sell.
old = '''                current_code = bt.value_of(positions, feat_idx, di, code=code, exclude=sell_keys)
                current_code += reserved_code.get(code, 0.0)
'''
new = '''                current_code = bt.value_of(positions, feat_idx, di, code=code)
                current_code += reserved_code.get(code, 0.0)
'''
if old not in s:
    raise SystemExit("single-name pending-sell anchor not found")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("PATCHED", p)
