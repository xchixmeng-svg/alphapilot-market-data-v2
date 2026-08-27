#!/usr/bin/env python3
from pathlib import Path

p=Path(__file__).resolve().parent/'r10_fast_validation.py'
s=p.read_text(encoding='utf-8')

# Historical regression signal schedule comes from the formal workbook's
# R05掛單_65. Live/forward scanner remains untouched; this patch is applied only
# by the 2021-2025 regression workflow.
anchor='''    if r05_locked_exit[("8215", 20210106)]["decision_date"] != 20210108:
        raise RuntimeError("R0.5 fixture regression anchor 8215 is wrong")
'''
insert='''    if r05_locked_exit[("8215", 20210106)]["decision_date"] != 20210108:
        raise RuntimeError("R0.5 fixture regression anchor 8215 is wrong")

    r05_order_fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "r10_2021_2025_r05_orders_65.csv"
    r05fx = pd.read_csv(r05_order_fixture, dtype={"code": str})
    if len(r05fx) != 65:
        raise RuntimeError(f"R0.5 locked order fixture must contain 65 rows, got {len(r05fx)}")
    r05fx["decision_int"] = r05fx.decision_date.astype(str).str.replace("-", "", regex=False).astype(int)
    r05_order_by_date = {int(d): g.reset_index(drop=True) for d,g in r05fx.groupby("decision_int", sort=False)}
'''
if anchor not in s: raise SystemExit('R05 order fixture init anchor missing')
s=s.replace(anchor,insert,1)

anchor2='''        r7_state=r7_states[di]; r7_cands=r7_cands_map[di]; r05_state=r05_states[di]; r05_cands=r05_cands_map[di]
        regime=r7_state["regime"]; regime_changed=last_regime is None or regime!=last_regime; reb_due=last_reb_i is None or regime_changed or (i-last_reb_i)>=15
'''
insert2='''        r7_state=r7_states[di]; r7_cands=r7_cands_map[di]; r05_state=r05_states[di]; r05_cands=r05_cands_map[di]

        # HISTORICAL REGRESSION ONLY: lock the R0.5 signal date/code/order
        # sequence to the formal workbook, but still let the Portfolio Layer
        # recalculate target cash, quantity and affordability. This isolates
        # scanner/data revisions from portfolio-execution bugs.
        fxday=r05_order_by_date.get(di)
        if fxday is None:
            r05_cands=pd.DataFrame()
        else:
            day=by_date.get(di)
            locked=[]
            for j,fr in fxday.iterrows():
                code=str(fr.code).zfill(4)
                rr=day[day.code.astype(str).eq(code)] if day is not None else pd.DataFrame()
                if rr.empty:
                    raise RuntimeError(f"R05_GOLDEN_SIGNAL_MISSING_MARKET_ROW date={di} code={code}")
                z=rr.iloc[-1].copy()
                z["r05_rank"]=j+1
                z["r05_score"]=1_000_000.0-(j+1)
                locked.append(z)
            r05_cands=pd.DataFrame(locked)
            r05_state=dict(r05_state)
            r05_state["risk_on"]=True

        regime=r7_state["regime"]; regime_changed=last_regime is None or regime!=last_regime; reb_due=last_reb_i is None or regime_changed or (i-last_reb_i)>=15
'''
if anchor2 not in s: raise SystemExit('R05 signal override anchor missing')
s=s.replace(anchor2,insert2,1)

# Strictly compare every R0.5 order that the portfolio actually creates against
# the workbook row for that T date. This catches capital-path and sizing drift.
anchor3='''            if created: pending_buys.setdefault(exdate,[]).extend(created)

        nav_rows.append'''
insert3='''            actual_r05=[o for o in created if o.strategy=="R05"]
            exp=r05_order_by_date.get(di)
            expected_rows=[] if exp is None else list(exp.itertuples(index=False))
            if len(actual_r05)!=len(expected_rows):
                raise RuntimeError(f"R05_GOLDEN_ORDER_COUNT_DIVERGENCE date={di} expected={[(str(x.code).zfill(4),int(x.shares)) for x in expected_rows]} actual={[(o.code,o.shares) for o in actual_r05]} nav={nav:.2f} cash={cash:.2f} sell_keys={sorted(sell_keys)}")
            for o,x in zip(actual_r05,expected_rows):
                ec=str(x.code).zfill(4); el=round(float(x.limit),2); es=int(x.shares); et=float(x.target_cash)
                if o.code!=ec or round(float(o.limit),2)!=el or int(o.shares)!=es or abs(float(o.target_cash)-et)>0.05:
                    raise RuntimeError(f"R05_GOLDEN_ORDER_VALUE_DIVERGENCE date={di} expected={(ec,el,es,et)} actual={(o.code,round(float(o.limit),2),int(o.shares),float(o.target_cash))} nav={nav:.2f} cash={cash:.2f} sell_keys={sorted(sell_keys)}")
            if created: pending_buys.setdefault(exdate,[]).extend(created)

        nav_rows.append'''
if anchor3 not in s: raise SystemExit('R05 strict compare anchor missing')
s=s.replace(anchor3,insert3,1)

old='''"historical_regression_uses_locked_underlying_exit_dates":True,"r05_locked_exit_fixture_rows":int(len(fx)),"live_forward_uses_rule_engine":True,'''
new='''"historical_regression_uses_locked_underlying_exit_dates":True,"r05_locked_exit_fixture_rows":int(len(fx)),"historical_regression_uses_locked_r05_signal_schedule":True,"r05_locked_order_fixture_rows":int(len(r05fx)),"live_forward_uses_rule_engine":True,'''
if old not in s: raise SystemExit('R05 result metadata anchor missing')
s=s.replace(old,new,1)
s=s.replace('AlphaPilot-R10-FastValidation-v6-GOLDEN-EXIT-SCHEDULE','AlphaPilot-R10-FastValidation-v10-GOLDEN-R05-SIGNAL-SCHEDULE',1)
p.write_text(s,encoding='utf-8')
print('PATCHED',p)
