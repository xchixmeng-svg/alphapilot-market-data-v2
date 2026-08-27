#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1) Repair the shared sizing primitive from the formal R10 workbook.
# Golden rule:
# - normal: floor(target / (limit*1000)) * 1000
# - odd-lot exception only when one board lot NOTIONAL exceeds effective target
# - high-price odd lot: floor(target / limit) integer shares
# - 2% ADV cap is then applied; fees are an affordability check, not part of
#   the strategic share-count denominator.
# ---------------------------------------------------------------------------
bt_path = ROOT / "backtest_r10_stress.py"
bt = bt_path.read_text(encoding="utf-8")
old_size = '''def size_shares(target_cash: float, base_target_cash: float, limit: float, avg_vol20: float) -> Tuple[int, str]:
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
new_size = '''def size_shares(target_cash: float, base_target_cash: float, limit: float, avg_vol20: float) -> Tuple[int, str]:
    if target_cash <= 0 or limit <= 0 or not np.isfinite(avg_vol20):
        return 0, "NONE"
    one_lot_notional = limit * 1000.0

    # Formal R10 workbook: odd lots are allowed ONLY when one board lot itself
    # is larger than the EFFECTIVE target for this order.
    if one_lot_notional > target_cash + 1e-9:
        mode = "HIGH_PRICE_ODDLOT"
        shares = int(math.floor(target_cash / limit + 1e-12))
        liq = int(math.floor(float(avg_vol20) * ADV_CAP))
        shares = min(shares, max(0, liq))
        return max(0, int(shares)), mode

    # Normal case: FLOOR to whole 1,000-share board lots. Never round to the
    # nearest lot and never force one lot above the effective target.
    mode = "BOARD_LOT"
    lots = int(math.floor(target_cash / one_lot_notional + 1e-12))
    liq_lots = int(math.floor(float(avg_vol20) * ADV_CAP / 1000.0))
    lots = min(lots, max(0, liq_lots))
    return max(0, lots * 1000), mode
'''
if old_size not in bt:
    raise SystemExit("backtest size_shares block not found; refusing blind patch")
bt = bt.replace(old_size, new_size, 1)
bt_path.write_text(bt, encoding="utf-8")

# ---------------------------------------------------------------------------
# 2) FAST portfolio execution repairs already identified earlier.
# ---------------------------------------------------------------------------
p = ROOT / "r10_fast_validation.py"
s = p.read_text(encoding="utf-8")

old = '''                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
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
new = '''                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
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
if old not in s:
    raise SystemExit("FAST sizing block not found")
s = s.replace(old, new, 1)

old2 = '''            if dd<=bt.FORCE_DD and i>=force_cooldown_until:
                force_cooldown_until=i+bt.FORCE_COOLDOWN_DAYS; no_buy_until=max(no_buy_until,i+bt.FORCE_NO_BUY_DAYS); forced_count+=1
'''
new2 = '''            if dd<=bt.FORCE_DD and i>=force_cooldown_until:
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
'''
if old2 not in s:
    raise SystemExit("FAST force-DD block not found")
s = s.replace(old2, new2, 1)

s = s.replace('AlphaPilot-R10-FastValidation-v3-NO-IMPORT-PROFILE-SIDE-EFFECT', 'AlphaPilot-R10-FastValidation-v5-GOLDEN-WORKBOOK-SIZING', 1)
p.write_text(s, encoding="utf-8")

# ---------------------------------------------------------------------------
# 3) Make the contract test enforce the workbook's exact FLOOR examples.
# ---------------------------------------------------------------------------
ct_path = ROOT / "r10_contract_test.py"
ct = ct_path.read_text(encoding="utf-8")
ct = ct.replace(
    'shares3, mode3 = bt.size_shares(260_000, 260_000, 27.0, 10_000_000)\nok("E14_R05_260K_27_NEAREST_10LOTS", shares3 == 10_000 and mode3 == "BOARD_LOT", f"shares={shares3},mode={mode3}")',
    'shares3, mode3 = bt.size_shares(260_000, 260_000, 27.0, 10_000_000)\nok("E14_R05_260K_27_FLOOR_9LOTS", shares3 == 9_000 and mode3 == "BOARD_LOT", f"shares={shares3},mode={mode3}")'
)
anchor = 'shares4, mode4 = bt.size_shares(260_000, 260_000, 500.0, 10_000_000)\nok("E15_HIGH_PRICE_ONLY_ODDLOT", 0 < shares4 < 1000 and mode4 == "HIGH_PRICE_ODDLOT", f"shares={shares4},mode={mode4}")\n'
extra = anchor + '''\n# Golden-workbook first-day sizing examples.\ng1, gm1 = bt.size_shares(286_000, 286_000, 29.15, 10_000_000)\nok("E16_GOLDEN_2415_FLOOR_9000", g1 == 9_000 and gm1 == "BOARD_LOT", f"shares={g1},mode={gm1}")\ng2, gm2 = bt.size_shares(220_050, 286_000, 26.55, 10_000_000)\nok("E17_GOLDEN_3149_FLOOR_8000", g2 == 8_000 and gm2 == "BOARD_LOT", f"shares={g2},mode={gm2}")\ng3, gm3 = bt.size_shares(260_000, 260_000, 18.05, 10_000_000)\nok("E18_GOLDEN_2426_FLOOR_14000", g3 == 14_000 and gm3 == "BOARD_LOT", f"shares={g3},mode={gm3}")\n'''
if anchor not in ct:
    raise SystemExit("contract sizing anchor not found")
ct = ct.replace(anchor, extra, 1)
ct_path.write_text(ct, encoding="utf-8")

print("PATCHED", bt_path)
print("PATCHED", p)
print("PATCHED", ct_path)
