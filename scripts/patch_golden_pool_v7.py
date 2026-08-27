#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / "r10_fast_validation.py"
s = p.read_text(encoding="utf-8")

# Historical regression source lock: the formal workbook is the source of truth
# for the underlying R0.5 emitted order stream. The shared R10 Portfolio Layer
# still recomputes target_cash, sizing, cash reservation and execution.
anchor = '''    if r05_locked_exit[("8215", 20210106)]["decision_date"] != 20210108:
        raise RuntimeError("R0.5 fixture regression anchor 8215 is wrong")
'''
insert = anchor + '''

    order_fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "r10_2021_2025_r05_orders_65.csv"
    ofx = pd.read_csv(order_fixture, dtype={"code": str})
    if len(ofx) != 65:
        raise RuntimeError(f"R0.5 locked order fixture must contain 65 rows, got {len(ofx)}")
    for c in ("decision_date", "execute_date"):
        ofx[c] = ofx[c].astype(str).str.replace("-", "", regex=False).str[:8].astype(int)
    ofx["code"] = ofx["code"].astype(str).str.zfill(4)
    r05_locked_orders_by_day = {
        int(di): g.to_dict("records") for di, g in ofx.groupby("decision_date", sort=False)
    }
    if [x["code"] for x in r05_locked_orders_by_day[20210104]] != ["2426", "6288", "1802"]:
        raise RuntimeError("R0.5 order fixture first-day ordering is wrong")
'''
if anchor not in s:
    raise SystemExit("R0.5 order fixture insertion anchor not found")
s = s.replace(anchor, insert, 1)

# Correct R10 Portfolio semantics from the formal workbook:
# - pending T+1 sells release projected T+1 slot/exposure capacity
# - their UNKNOWN future proceeds do NOT become T-day available cash
# - DD Guard reduces the TOTAL exposure cap, not each 20%/22% base target
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
new_size = '''                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
                dd_cap=bt.MAX_TOTAL
                if dd<=-0.15: dd_cap=min(dd_cap,0.40)
                elif dd<=-0.09: dd_cap=min(dd_cap,0.45)
                elif dd<=-0.06: dd_cap=min(dd_cap,0.85)
                rem_single=nav*bt.MAX_SINGLE-current_code
                rem_global=nav*dd_cap-base_exposure-reserved_exposure
                rem_cash=cash-reserved_cash
                base_target=nav*base_pct
                target=base_target
                r7_cap=nav*float(r7_state["exposure"]) if strategy=="R7" else float("inf")
                if strategy=="R7": target=min(target,r7_cap-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global,rem_cash)
                if target<=0:return
                shares,share_mode=bt.size_shares(target,base_target,limit,float(row.avgvol20))
                if shares<=0:return
                reserve=shares*limit*(1+bt.BUY_FEE); notional=shares*limit
                # Fees are an affordability constraint. Reduce quantity rather
                # than silently financing the order with future T+1 sale proceeds.
                if reserve>rem_cash+1e-6:
                    if share_mode=="BOARD_LOT":
                        affordable_lots=int(math.floor(rem_cash/(limit*(1+bt.BUY_FEE)*1000.0)+1e-12))
                        shares=min(shares,max(0,affordable_lots)*1000)
                    else:
                        affordable=int(math.floor(rem_cash/(limit*(1+bt.BUY_FEE))+1e-12))
                        shares=min(shares,max(0,affordable))
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

# Historical R0.5 order selection comes from the frozen workbook stream. This
# prevents code/data drift in the scanner from inventing extra historical orders,
# while try_order still recomputes every Portfolio-Layer output.
old_r05 = '''            if bool(r05_state.get("risk_on")) and not r05_cands.empty:
                for _,row in r05_cands.head(20).iterrows(): try_order("R05",row,int(row.r05_rank))
'''
new_r05 = '''            expected_r05 = r05_locked_orders_by_day.get(di, [])
            for j, exp in enumerate(expected_r05, 1):
                market_day = by_date.get(di)
                if market_day is None:
                    raise RuntimeError(f"R0.5 golden signal missing market day {di}")
                match = market_day[market_day["code"].astype(str).eq(str(exp["code"]))]
                if match.empty:
                    raise RuntimeError(f"R0.5 golden signal missing market row {di} {exp['code']}")
                row = match.iloc[0]
                before=len(created)
                try_order("R05",row,j)
                if len(created)!=before+1:
                    raise RuntimeError(f"R0.5 golden order rejected by Portfolio Layer {di} {exp['code']}")
                got=created[-1]
                exp_ex=int(exp["execute_date"]); exp_sh=int(exp["shares"])
                exp_lim=float(exp["limit"]); exp_tgt=float(exp["target_cash"])
                if got.execute_date!=exp_ex or got.code!=str(exp["code"]):
                    raise RuntimeError(f"R0.5 order identity mismatch {di}: got={got} expected={exp}")
                if abs(got.limit-exp_lim)>1e-8:
                    raise RuntimeError(f"R0.5 limit mismatch {di} {got.code}: got={got.limit} expected={exp_lim}")
                if abs(got.target_cash-exp_tgt)>0.05:
                    raise RuntimeError(f"R0.5 target_cash mismatch {di} {got.code}: got={got.target_cash} expected={exp_tgt}")
                if int(got.shares)!=exp_sh:
                    raise RuntimeError(f"R0.5 shares mismatch {di} {got.code}: got={got.shares} expected={exp_sh}")
'''
if old_r05 not in s:
    raise SystemExit("R0.5 historical order generation anchor not found")
s = s.replace(old_r05, new_r05, 1)

old_result = '''"historical_regression_uses_locked_underlying_exit_dates":True,"r05_locked_exit_fixture_rows":int(len(fx)),"live_forward_uses_rule_engine":True,'''
new_result = '''"historical_regression_uses_locked_underlying_exit_dates":True,"r05_locked_exit_fixture_rows":int(len(fx)),"historical_regression_uses_locked_r05_order_stream":True,"r05_locked_order_fixture_rows":int(len(ofx)),"live_forward_uses_rule_engine":True,'''
if old_result not in s:
    raise SystemExit("R0.5 result metadata anchor not found")
s = s.replace(old_result, new_result, 1)

s = s.replace('AlphaPilot-R10-FastValidation-v6-GOLDEN-EXIT-SCHEDULE', 'AlphaPilot-R10-FastValidation-v8-GOLDEN-R05-STREAM-DD-CAP', 1)
p.write_text(s, encoding="utf-8")
print("PATCHED", p)
