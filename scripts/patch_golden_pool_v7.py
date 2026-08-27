#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / "r10_fast_validation.py"
s = p.read_text(encoding="utf-8")

# Golden workbook timing semantics:
# a T-close sell decision is still an OPEN position until the T+1 sell actually
# executes. Do not pre-release its slot, exposure or cash when creating other
# T+1 buy orders on the same T decision date.
old_head = '''            exdate=next_date[di]; sell_keys=set(sell_map); codes_after={p.code for k,p in positions.items() if k not in sell_keys}
            base_exposure=bt.value_of(positions,feat_idx,di,exclude=sell_keys); base_r7=bt.value_of(positions,feat_idx,di,strategy="R7",exclude=sell_keys)
'''
new_head = '''            exdate=next_date[di]; sell_keys=set(sell_map); codes_after={p.code for p in positions.values()}
            base_exposure=bt.value_of(positions,feat_idx,di); base_r7=bt.value_of(positions,feat_idx,di,strategy="R7")
'''
if old_head not in s:
    raise SystemExit("golden pool head anchor not found")
s = s.replace(old_head, new_head, 1)

s = s.replace('''                if k in positions and k not in sell_keys:return
''', '''                if k in positions:return
''', 1)
s = s.replace('''                    n=sum(1 for kk,p in positions.items() if p.strategy=="R05" and kk not in sell_keys)+sum(1 for o in created if o.strategy=="R05")
''', '''                    n=sum(1 for p in positions.values() if p.strategy=="R05")+sum(1 for o in created if o.strategy=="R05")
''', 1)
s = s.replace('''                    n=sum(1 for kk,p in positions.items() if p.strategy=="R7" and kk not in sell_keys)+sum(1 for o in created if o.strategy=="R7")
''', '''                    n=sum(1 for p in positions.values() if p.strategy=="R7")+sum(1 for o in created if o.strategy=="R7")
''', 1)

old_size = '''                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
                rem_single=nav*bt.MAX_SINGLE-current_code; rem_global=nav*bt.MAX_TOTAL-base_exposure-reserved_exposure
                base_target=nav*base_pct
                target=base_target*bt.dd_multiplier(dd)
                r7_cap=nav*float(r7_state["exposure"]) if strategy=="R7" else float("inf")
                if strategy=="R7": target=min(target,r7_cap-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global)
                if target<=0:return
                shares,_=bt.size_shares(target,base_target,limit,float(row.avgvol20))
                if shares<=0:return
                reserve=shares*limit*(1+bt.BUY_FEE); notional=shares*limit
                if notional>rem_single+1e-6:return
                if notional>rem_global+1e-6:return
                if strategy=="R7" and base_r7+reserved_r7+notional>r7_cap*1.03+1:return
'''
new_size = '''                current_code=bt.value_of(positions,feat_idx,di,code=code)+reserved_code.get(code,0)
                rem_single=nav*bt.MAX_SINGLE-current_code; rem_global=nav*bt.MAX_TOTAL-base_exposure-reserved_exposure; rem_cash=cash-reserved_cash
                base_target=nav*base_pct
                target=base_target*bt.dd_multiplier(dd)
                r7_cap=nav*float(r7_state["exposure"]) if strategy=="R7" else float("inf")
                if strategy=="R7": target=min(target,r7_cap-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global,rem_cash)
                if target<=0:return
                shares,share_mode=bt.size_shares(target,base_target,limit,float(row.avgvol20))
                if shares<=0:return
                reserve=shares*limit*(1+bt.BUY_FEE); notional=shares*limit
                if reserve>rem_cash+1e-6:return
                if notional>rem_single+1e-6:return
                if notional>rem_global+1e-6:return
                if strategy=="R7" and base_r7+reserved_r7+notional>r7_cap*1.03+1:return
'''
if old_size not in s:
    raise SystemExit("golden pool sizing anchor not found")
s = s.replace(old_size, new_size, 1)

s = s.replace('AlphaPilot-R10-FastValidation-v6-GOLDEN-EXIT-SCHEDULE', 'AlphaPilot-R10-FastValidation-v7-GOLDEN-POOL-TIMING', 1)
p.write_text(s, encoding="utf-8")
print("PATCHED", p)
