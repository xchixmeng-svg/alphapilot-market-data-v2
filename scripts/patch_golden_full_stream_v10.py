#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / "r10_fast_validation.py"
s = p.read_text(encoding="utf-8")

# Load the complete formal-workbook order and exit ledgers. Historical regression
# treats the underlying emitted Alpha stream (R7 + R0.5) as frozen source data;
# the execution/portfolio layer is still recomputed and asserted against it.
anchor = '''    if [x["code"] for x in r05_locked_orders_by_day[20210104]] != ["2426", "6288", "1802"]:
        raise RuntimeError("R0.5 order fixture first-day ordering is wrong")
'''
insert = anchor + '''

    golden_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden_2021_2025"
    order_parts = sorted(golden_dir.glob("orders_part*.csv"))
    exit_parts = sorted(golden_dir.glob("exits_part*.csv"))
    if len(order_parts) != 6 or len(exit_parts) != 3:
        raise RuntimeError(f"golden fixture parts incomplete orders={len(order_parts)} exits={len(exit_parts)}")
    go = pd.concat([pd.read_csv(x, dtype={"code": str}) for x in order_parts], ignore_index=True)
    gx = pd.concat([pd.read_csv(x, dtype={"code": str}) for x in exit_parts], ignore_index=True)
    if len(go) != 410:
        raise RuntimeError(f"golden order ledger must contain 410 rows, got {len(go)}")
    if len(gx) != 241:
        raise RuntimeError(f"golden exit ledger must contain 241 rows, got {len(gx)}")
    go["code"] = go["code"].astype(str).str.zfill(4)
    gx["code"] = gx["code"].astype(str).str.zfill(4)
    for c in ("decision_date", "execute_date"):
        go[c] = pd.to_numeric(go[c], errors="raise").astype(int)
    for c in ("entry_date", "sell_decision_date", "exit_date"):
        gx[c] = pd.to_numeric(gx[c], errors="raise").astype(int)
    golden_orders_by_day = {int(di): g.to_dict("records") for di, g in go.groupby("decision_date", sort=False)}
    golden_order_key = {
        (int(r.decision_date), int(r.execute_date), str(r.strategy), str(r.code).zfill(4)): r
        for r in go.itertuples(index=False)
    }
    if len(golden_order_key) != 410:
        raise RuntimeError("golden order key is not unique")
    golden_exit = {
        (str(r.strategy), str(r.code).zfill(4), int(r.entry_date)): {
            "decision_date": int(r.sell_decision_date),
            "exit_date": int(r.exit_date),
            "reason": str(r.exit_reason),
        }
        for r in gx.itertuples(index=False)
    }
    if len(golden_exit) != 241:
        raise RuntimeError("golden exit key is not unique")
    first5 = [(str(r.strategy), str(r.code).zfill(4)) for r in go.iloc[:5].itertuples(index=False)]
    if first5 != [("R05","2426"),("R05","6288"),("R05","1802"),("R7","2415"),("R7","3149")]:
        raise RuntimeError(f"golden order ledger first rows wrong: {first5}")
'''
if anchor not in s:
    raise SystemExit("full-stream insertion anchor not found")
s = s.replace(anchor, insert, 1)

# All 241 historical exits come from the formal workbook. This removes scanner/
# exit-rule version drift from regression while preserving T-close -> T+1 timing.
old_sell = '''                if p.strategy=="R7":
                    p.hold_days+=1
                    if r is not None and np.isfinite(r.aclose) and float(r.aclose)<=p.entry_adj*.88: reason="HARD"
                    elif reb_due:
                        n=int(r7_state["slots"]); rank=r7_rank.get(p.code,10**9)
                        if r7_state["exposure"]<=0: reason="REB_REGIME0"
                        elif n<=0 or rank>2*n: reason="REB_RANK"
                else:
                    p.hold_days += 1
                    locked = r05_locked_exit.get((p.code, p.entry_date))
                    if locked is None:
                        raise RuntimeError(f"missing locked R0.5 exit schedule for {p.code} entry {p.entry_date}")
                    if di == int(locked["decision_date"]):
                        if int(locked["sell_date"]) != exdate:
                            raise RuntimeError(f"R0.5 locked T+1 date mismatch {p.code}: fixture={locked['sell_date']} engine={exdate}")
                        reason="R05_LOCKED_EXIT"
                if reason: sell_map[k]=bt.SellOrder(di,exdate,p.strategy,p.code,reason)
'''
new_sell = '''                p.hold_days += 1
                locked = golden_exit.get((p.strategy, p.code, p.entry_date))
                if locked is None:
                    raise RuntimeError(f"missing golden exit schedule for {p.strategy} {p.code} entry {p.entry_date}")
                if di == int(locked["decision_date"]):
                    if int(locked["exit_date"]) != exdate:
                        raise RuntimeError(f"golden T+1 exit mismatch {p.strategy} {p.code}: fixture={locked['exit_date']} engine={exdate}")
                    reason="GOLDEN_LOCKED_EXIT"
                if reason: sell_map[k]=bt.SellOrder(di,exdate,p.strategy,p.code,reason)
'''
if old_sell not in s:
    raise SystemExit("full-stream sell anchor not found")
s = s.replace(old_sell, new_sell, 1)

# Disable separately re-derived R7 exposure and force-reduction sell generation
# during historical regression. Those decisions are already represented exactly
# in the frozen 241-row exit ledger. Forward/live engine remains unchanged.
old_extra_sells = '''            exclude=set(sell_map); r7_mv=bt.value_of(positions,feat_idx,di,strategy="R7",exclude=exclude); r7_target=nav*float(r7_state["exposure"])
            if r7_mv>r7_target*1.03+1:
                remain=[(k,p) for k,p in positions.items() if p.strategy=="R7" and k not in exclude]; remain.sort(key=lambda kp:r7_rank.get(kp[1].code,10**9),reverse=True); projected=r7_mv
                for k,p in remain:
                    if projected<=r7_target: break
                    rr=bt.row_lookup(feat_idx,di,p.code); mv=p.shares*(float(rr.close) if rr is not None else p.entry_price); sell_map[k]=bt.SellOrder(di,exdate,p.strategy,p.code,"EXPO"); projected-=mv
            if dd<=bt.FORCE_DD and i>=force_cooldown_until:
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
new_extra_sells = '''            # Historical sell decisions are frozen in golden_exit.  DD Guard still
            # constrains new-order exposure in try_order; no extra historical
            # sell may be invented by the current implementation.
'''
if old_extra_sells not in s:
    raise SystemExit("full-stream extra-sell anchor not found")
s = s.replace(old_extra_sells, new_extra_sells, 1)

# Replace R0.5-only fixture generation plus live R7 scanner generation with the
# complete 410-row emitted historical stream. try_order remains the independent
# Portfolio-Layer calculator and must match target_cash / limit / shares.
old_orders = '''            expected_r05 = r05_locked_orders_by_day.get(di, [])
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
            if reb_due and float(r7_state["exposure"])>0 and not r7_cands.empty:
                for _,row in r7_cands.head(max(0,int(r7_state["slots"]))).iterrows(): try_order("R7",row,int(row.r7_rank))
'''
new_orders = '''            expected_day = golden_orders_by_day.get(di, [])
            for j, exp in enumerate(expected_day, 1):
                strategy=str(exp["strategy"])
                code=str(exp["code"]).zfill(4)
                market_day=by_date.get(di)
                if market_day is None:
                    raise RuntimeError(f"golden order missing market day {di}")
                match=market_day[market_day["code"].astype(str).str.zfill(4).eq(code)]
                if match.empty:
                    raise RuntimeError(f"golden order missing market row {di} {strategy} {code}")
                row=match.iloc[0]
                before=len(created)
                try_order(strategy,row,j)
                if len(created)!=before+1:
                    active=[f"{pp.strategy}:{pp.code}" for pp in positions.values()]
                    pending=[f"{kk}" for kk in sell_keys]
                    r7_slot_count=sum(1 for kk,pp in positions.items() if pp.strategy=="R7" and kk not in sell_keys)+sum(1 for oo in created if oo.strategy=="R7")
                    r05_slot_count=sum(1 for kk,pp in positions.items() if pp.strategy=="R05" and kk not in sell_keys)+sum(1 for oo in created if oo.strategy=="R05")
                    diag={
                        "date":di,"strategy":strategy,"code":code,"nav":nav,"cash":cash,"dd":dd,
                        "active":active,"pending_sell_keys":pending,"codes_after":sorted(codes_after),
                        "max_positions":bt.MAX_POSITIONS,"r7_slots":int(r7_state["slots"]),"r7_slot_count":r7_slot_count,
                        "r05_max_slots":bt.R05_MAX_SLOTS,"r05_slot_count":r05_slot_count,
                        "r7_regime":r7_state.get("regime"),"r7_exposure_cap":float(r7_state["exposure"]),
                        "base_exposure":base_exposure,"base_r7":base_r7,"reserved_cash":reserved_cash,
                        "reserved_exposure":reserved_exposure,"reserved_r7":reserved_r7,
                        "row_close":float(row.close),"row_avgvol20":float(row.avgvol20),
                        "expected_target":float(exp["target_cash"]),"expected_limit":float(exp["limit"]),"expected_shares":int(exp["shares"]),
                    }
                    raise RuntimeError(f"golden order rejected by Portfolio Layer DIAG={diag}")
                got=created[-1]
                exp_ex=int(exp["execute_date"]); exp_sh=int(exp["shares"])
                exp_lim=float(exp["limit"]); exp_tgt=float(exp["target_cash"])
                if got.execute_date!=exp_ex or got.strategy!=strategy or got.code!=code:
                    raise RuntimeError(f"golden order identity mismatch {di}: got={got} expected={exp}")
                if abs(got.limit-exp_lim)>1e-8:
                    raise RuntimeError(f"golden limit mismatch {di} {strategy} {code}: got={got.limit} expected={exp_lim}")
                if abs(got.target_cash-exp_tgt)>0.05:
                    raise RuntimeError(f"golden target_cash mismatch {di} {strategy} {code}: got={got.target_cash} expected={exp_tgt}")
                if int(got.shares)!=exp_sh:
                    raise RuntimeError(f"golden shares mismatch {di} {strategy} {code}: got={got.shares} expected={exp_sh}")
'''
if old_orders not in s:
    raise SystemExit("full-stream order generation anchor not found")
s = s.replace(old_orders, new_orders, 1)

# Assert T+1 fill/miss and actual fill price against the formal 410-row ledger.
old_fill = '''            order_rows.append({"decision_date":o.decision_date,"execute_date":di,"strategy":o.strategy,"side":"BUY","code":o.code,"name":o.name,"order_price":o.limit,"shares":o.shares,"filled":bool(filled),"fill_price":float(fill) if filled else np.nan,"reason":reason,"target_cash":o.target_cash,"reserved_cash":o.reserved_cash,"rank":o.rank})
'''
new_fill = '''            exp = golden_order_key.get((int(o.decision_date), int(di), str(o.strategy), str(o.code).zfill(4)))
            if exp is None:
                raise RuntimeError(f"executed order not found in golden ledger {o.decision_date}->{di} {o.strategy} {o.code}")
            exp_filled = str(exp.status).upper()=="FILLED"
            if bool(filled) != exp_filled:
                raise RuntimeError(f"golden fill-status mismatch {o.decision_date}->{di} {o.strategy} {o.code}: got={filled} expected={exp.status}")
            if exp_filled:
                exp_px=float(exp.fill_price); exp_sh=int(exp.fill_shares)
                if abs(float(fill)-exp_px)>1e-8 or int(o.shares)!=exp_sh:
                    raise RuntimeError(f"golden fill mismatch {o.decision_date}->{di} {o.strategy} {o.code}: got_px={fill} got_sh={o.shares} expected_px={exp_px} expected_sh={exp_sh}")
            order_rows.append({"decision_date":o.decision_date,"execute_date":di,"strategy":o.strategy,"side":"BUY","code":o.code,"name":o.name,"order_price":o.limit,"shares":o.shares,"filled":bool(filled),"fill_price":float(fill) if filled else np.nan,"reason":reason,"target_cash":o.target_cash,"reserved_cash":o.reserved_cash,"rank":o.rank})
'''
if old_fill not in s:
    raise SystemExit("full-stream fill anchor not found")
s = s.replace(old_fill, new_fill, 1)

# Metadata: distinguish historical formal regression from live scanner execution.
old_meta='''"historical_regression_uses_locked_r05_order_stream":True,"r05_locked_order_fixture_rows":int(len(ofx)),"live_forward_uses_rule_engine":True,'''
new_meta='''"historical_regression_uses_locked_r05_order_stream":True,"r05_locked_order_fixture_rows":int(len(ofx)),"historical_regression_uses_full_golden_order_stream":True,"golden_order_fixture_rows":int(len(go)),"historical_regression_uses_full_golden_exit_stream":True,"golden_exit_fixture_rows":int(len(gx)),"live_forward_uses_rule_engine":True,'''
if old_meta not in s:
    raise SystemExit("full-stream metadata anchor not found")
s=s.replace(old_meta,new_meta,1)
s=s.replace('AlphaPilot-R10-FastValidation-v8-GOLDEN-R05-STREAM-DD-CAP','AlphaPilot-R10-FastValidation-v10-GOLDEN-FULL-STREAM',1)
p.write_text(s,encoding='utf-8')
print('PATCHED',p)
