#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bring the optimized FAST validation engine back to exact locked-engine parity.

Repairs three known divergences:
1) size_shares() argument order / base-target semantics;
2) T-day cash must not resize precommitted orders; T+1 sells fund the common pool first;
3) FORCE_DD must actually liquidate weakest positions to the locked target exposure,
   not merely start cooldown/no-buy timers.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "r10_fast_validation.py"
s = P.read_text(encoding="utf-8")

old_force = '''            if dd<=bt.FORCE_DD and i>=force_cooldown_until:
                force_cooldown_until=i+bt.FORCE_COOLDOWN_DAYS; no_buy_until=max(no_buy_until,i+bt.FORCE_NO_BUY_DAYS); forced_count+=1
            if sell_map: pending_sells.setdefault(exdate,[]).extend(sell_map.values())
'''
new_force = '''            if dd<=bt.FORCE_DD and i>=force_cooldown_until:
                force_cooldown_until=i+bt.FORCE_COOLDOWN_DAYS; no_buy_until=max(no_buy_until,i+bt.FORCE_NO_BUY_DAYS); forced_count+=1
                exclude=set(sell_map); projected=bt.value_of(positions,feat_idx,di,exclude=exclude); force_target=nav*bt.FORCE_TARGET_EXPOSURE
                remain=[(k,p) for k,p in positions.items() if k not in exclude]
                def weakness(item):
                    _,p=item
                    if p.strategy=="R7": return (0,-r7_rank.get(p.code,10**9),r7_score.get(p.code,-1e9))
                    rr=bt.row_lookup(feat_idx,di,p.code); ret=(float(rr.aclose)/p.entry_adj-1) if rr is not None and np.isfinite(rr.aclose) else -9
                    return (1,r05_score.get(p.code,-1e9),ret)
                remain.sort(key=weakness)
                for k,p in remain:
                    if projected<=force_target: break
                    rr=bt.row_lookup(feat_idx,di,p.code); mv=p.shares*(float(rr.close) if rr is not None else p.entry_price)
                    sell_map[k]=bt.SellOrder(di,exdate,p.strategy,p.code,"FORCE_DD"); projected-=mv
                event_rows.append({"date":di,"event":"FORCE_DD","dd":dd,"target_exposure":bt.FORCE_TARGET_EXPOSURE})
            if sell_map: pending_sells.setdefault(exdate,[]).extend(sell_map.values())
'''

old_size = '''                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
                rem_single=nav*bt.MAX_SINGLE-current_code; rem_global=nav*bt.MAX_TOTAL-base_exposure-reserved_exposure; rem_cash=cash-reserved_cash
                target=nav*base_pct*bt.dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global,rem_cash)
                if target<=0:return
                shares,_=bt.size_shares(target,limit,float(row.avgvol20),rem_cash)
                if shares<=0:return
                reserve=shares*limit*(1+bt.BUY_FEE); notional=shares*limit
                if reserve>rem_cash+1e-6:return
'''
new_size = '''                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
                rem_single=nav*bt.MAX_SINGLE-current_code; rem_global=nav*bt.MAX_TOTAL-base_exposure-reserved_exposure
                base_target=nav*base_pct
                target=base_target*bt.dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global)
                if target<=0:return
                shares,_=bt.size_shares(target,base_target,limit,float(row.avgvol20))
                if shares<=0:return
                reserve=shares*limit*(1+bt.BUY_FEE); notional=shares*limit
'''

if old_force not in s:
    raise SystemExit("FAST FORCE_DD block not found; refusing blind patch")
if old_size not in s:
    raise SystemExit("FAST sizing/common-pool block not found; refusing blind patch")

s = s.replace(old_force, new_force, 1)
s = s.replace(old_size, new_size, 1)
P.write_text(s, encoding="utf-8")
print("PATCHED", P)
