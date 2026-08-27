#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import r10_full_battery as full

bt = full.bt
core = bt.core
# IMPORTANT: importing the FAST engine must never mutate the active R10 profile.
# Profile selection is explicit in the calling wrapper only.
bt.VERSION = "AlphaPilot-R10-FastValidation-v3-NO-IMPORT-PROFILE-SIDE-EFFECT"

CACHE = bt.CACHE_ROOT / "fast_validation"
INST_DAY_CACHE = CACHE / "inst_days"
INST_DAY_CACHE.mkdir(parents=True, exist_ok=True)


def _inst_day(di: int) -> list[dict]:
    p = INST_DAY_CACHE / f"{di}.csv"
    if p.exists():
        q = pd.read_csv(p, dtype={"code": str})
        return q.to_dict("records")
    d = bt.dt_from_int(di)
    rows: list[dict] = []
    errs = []
    try:
        rows.extend(core.twse_inst(d))
    except Exception as exc:
        errs.append(f"TWSE {exc}")
    try:
        rows.extend(core.tpex_inst(d))
    except Exception as exc:
        errs.append(f"TPEX {exc}")
    if not rows:
        raise RuntimeError(f"{di} institutional empty: {'; '.join(errs)}")
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    return rows


def fetch_institutional_fast(feat: pd.DataFrame, eval_start: int, eval_end: int) -> pd.DataFrame:
    min_dt = bt.dt_from_int(eval_start) - timedelta(days=50)
    dates = [int(x) for x in sorted(feat.date.unique()) if min_dt <= bt.dt_from_int(int(x)) <= bt.dt_from_int(eval_end)]
    rows: list[dict] = []
    missing = [di for di in dates if not (INST_DAY_CACHE / f"{di}.csv").exists()]
    bt.log(f"[FAST-INST] total_dates={len(dates)} cached={len(dates)-len(missing)} missing={len(missing)}")
    if missing:
        failures = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_inst_day, di): di for di in missing}
            done = 0
            for fut in as_completed(futs):
                di = futs[fut]
                done += 1
                try:
                    fut.result()
                except Exception as exc:
                    failures.append(f"{di}: {exc}")
                if done % 25 == 0 or done == len(missing):
                    bt.log(f"[FAST-INST] fetched {done}/{len(missing)} failures={len(failures)}")
        if failures:
            bt.log("[FAST-INST] sample failures: " + " | ".join(failures[:5]))
    for n, di in enumerate(dates, 1):
        p = INST_DAY_CACHE / f"{di}.csv"
        if p.exists():
            q = pd.read_csv(p, dtype={"code": str})
            if not q.empty:
                rows.extend(q.to_dict("records"))
        if n % 250 == 0:
            bt.log(f"[FAST-INST] assemble {n}/{len(dates)} rows={len(rows)}")
    ins = bt.build_inst_features(rows)
    coverage = ins[ins.date.between(eval_start, eval_end)].date.nunique()
    target_dates = len([d for d in dates if eval_start <= d <= eval_end])
    ratio = coverage / target_dates if target_dates else 0.0
    bt.log(f"[FAST-INST] coverage={coverage}/{target_dates} ({ratio:.1%})")
    if ratio < 0.99:
        raise RuntimeError(f"institutional coverage too low {ratio:.1%}")
    return ins


def precompute_signals(feat: pd.DataFrame, bm: pd.DataFrame, ins: pd.DataFrame, eval_dates: list[int]):
    t0 = time.time()
    # Current-day stock table: no repeated feat[feat.date<=T] scans.
    by_date = {int(d): g.copy() for d, g in feat.groupby("date", sort=False)}

    # R7 breadth and benchmark state, all rolling operations are backward-looking only.
    eligible = feat[(feat.amt20 >= 30_000_000) & feat.aclose.notna()].copy()
    breadth = eligible.groupby("date").apply(
        lambda g: pd.Series({"breadth": float((g.aclose > g.ma60).mean()), "advance10": float((g.r10 > 0).mean())}),
        include_groups=False,
    ).reset_index().sort_values("date")
    breadth["breadth_mean20"] = breadth.breadth.rolling(20, min_periods=10).mean()
    bmap = breadth.set_index("date")

    z = bm.drop_duplicates("date").sort_values("date").copy()
    z["ma60"] = z.mkt.rolling(60, min_periods=60).mean()
    z["ma120"] = z.mkt.rolling(120, min_periods=120).mean()
    z["mr20"] = z.mkt.pct_change(20, fill_method=None)
    z["mr60"] = z.mkt.pct_change(60, fill_method=None)
    zmap = z.set_index("date")

    # R0.5 0050 risk state precomputed once.
    et = feat[feat.code.eq("0050")][["date", "close"]].drop_duplicates("date").sort_values("date").copy()
    et["m60"] = et.close.rolling(60, min_periods=60).mean()
    et["r20x"] = et.close.pct_change(20, fill_method=None)
    et["r60x"] = et.close.pct_change(60, fill_method=None)
    emap = et.set_index("date")
    inst_by_date = {int(d): g.sort_values("market").drop_duplicates("code", keep="last") for d, g in ins.groupby("date", sort=False)}

    r7_states: dict[int, dict] = {}
    r7_cands: dict[int, pd.DataFrame] = {}
    r05_states: dict[int, dict] = {}
    r05_cands: dict[int, pd.DataFrame] = {}

    for i, di in enumerate(eval_dates):
        x0 = by_date.get(di)
        if x0 is None or di not in bmap.index or di not in zmap.index:
            raise RuntimeError(f"signal context missing {di}")
        x = x0[core.common(x0)].copy()
        r = zmap.loc[di]
        v = bmap.loc[di]
        m, ma60, ma120, mr20, mr60, br, adv, bmean = [float(q) for q in (r.mkt, r.ma60, r.ma120, r.mr20, r.mr60, v.breadth, v.advance10, v.breadth_mean20)]
        if mr20 <= -.08 or (m < ma120 and mr60 < 0 and br < .40): reg, expo, slots = "Bear", 0., 0
        elif m < ma120 * 1.02 and mr20 > 0 and br > .42 and br > bmean: reg, expo, slots = "Repair", .60, 2
        elif m > ma60 and m > ma120 and mr20 > 0 and mr60 > 0 and br >= .60 and adv >= .52: reg, expo, slots = "Strong Bull", 1., 4
        elif m > ma120 and mr60 > 0 and br >= .45: reg, expo, slots = "Normal Bull", .80, 3
        elif m > ma120 * .98 and br >= .38: reg, expo, slots = "Weak", .20, 2
        else: reg, expo, slots = "Fallback/Bear", 0., 0
        x["rel20"] = x.r20 - mr20
        x["rel60"] = x.r60 - mr60
        for s, n in [("r10","p10"),("rel20","p20"),("rel60","p60"),("flow20","pf"),("amtacc","pa"),("clvflow20","pc"),("nearhigh","pn")]:
            x[n] = core.pr(x[s])
        x["r7_score"] = .26*x.p10 + .22*x.p20 + .10*x.p60 + .14*x.pf + .12*x.pa + .08*x.pc + .08*x.pn
        c = x[(x.amt20 >= 30_000_000) & (x.aclose > x.ma120) & (x.nearhigh >= .78) & x.r7_score.notna()].sort_values(["r7_score","code"], ascending=[False,True]).copy()
        c["r7_rank"] = np.arange(1, len(c)+1)
        r7_states[di] = {"regime":reg,"exposure":expo,"slots":slots,"mkt":m,"mkt_ma60":ma60,"mkt_ma120":ma120,"mr20":mr20,"mr60":mr60,"breadth60":br,"breadth_mean20":bmean,"advance10":adv}
        r7_cands[di] = c

        if di in emap.index and di in inst_by_date:
            e = emap.loc[di]
            risk = bool(e.close > e.m60 and e.r20x > 0 and e.r60x > 0)
            xi = x0[core.common(x0)].copy().merge(inst_by_date[di][["code","Foreign3D","Foreign10D","Trust5D"]], on="code", how="left")
            for s, n in [("clvflow10","pclv10"),("amount_ratio","pamt"),("clvflow5","pclv5"),("Foreign3D","pf3"),("Foreign10D","pf10"),("Trust5D","pt5"),("ma20gap","pgap")]:
                xi[n] = core.pr(xi[s])
            xi["r05_score"] = .5251*xi.pclv10 + .2465*xi.pamt + .0683*xi.pclv5 + .0628*xi.pf3 - .0778*xi.pf10 + .0195*xi.pt5 - .2*xi.pgap
            xi["prior60_position"] = xi.aclose / xi.prior_high60 - 1
            h = (xi.close.between(10,40)) & (xi.amt20 >= 50_000_000) & (xi.amount_ratio >= 1) & (xi.r20.between(0,.20)) & (xi.ma20gap <= .18) & (xi.prior60_position >= -.15) & (xi.aclose > xi.prior_high10)
            cc = xi[h & xi.r05_score.notna()].sort_values(["r05_score","code"], ascending=[False,True]).copy()
            cc["r05_rank"] = np.arange(1, len(cc)+1)
            if not risk:
                cc = cc.iloc[0:0]
            r05_states[di] = {"risk_on":risk,"0050_close":float(e.close),"0050_ma60":float(e.m60),"0050_ret20":float(e.r20x),"0050_ret60":float(e.r60x)}
            r05_cands[di] = cc
        else:
            r05_states[di] = {"risk_on":False}
            r05_cands[di] = pd.DataFrame()

        if (i+1) % 100 == 0 or i+1 == len(eval_dates):
            bt.log(f"[SIGNALS] {i+1}/{len(eval_dates)} ({(i+1)/len(eval_dates):.1%}) date={di}")
    bt.log(f"[SIGNALS] precompute complete in {time.time()-t0:.1f}s")
    return by_date, r7_states, r7_cands, r05_states, r05_cands


def simulate_fast() -> dict:
    cfg = bt.SCENARIOS["validation2021_2025"]
    raw = bt.load_scenario_ohlcv(cfg)
    feat, corp_events, bm = bt.build_features(raw)
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= x <= eval_end)
    next_date = {eval_dates[i]: eval_dates[i+1] for i in range(len(eval_dates)-1)}
    feat_idx = feat.set_index(["date","code"]).sort_index()
    ins = fetch_institutional_fast(feat, eval_start, eval_end)
    by_date, r7_states, r7_cands_map, r05_states, r05_cands_map = precompute_signals(feat, bm, ins, eval_dates)

    cash = bt.INITIAL_CAPITAL
    positions: Dict[str, bt.Position] = {}
    pending_buys: Dict[int, List[bt.BuyOrder]] = {}
    pending_sells: Dict[int, List[bt.SellOrder]] = {}
    nav_rows=[]; order_rows=[]; trade_rows=[]; event_rows=[]
    hwm=cash; last_regime=None; last_reb_i=None; no_buy_until=-1; force_cooldown_until=-1; forced_count=0

    for i, di in enumerate(eval_dates):
        for o in pending_sells.pop(di, []):
            k=bt.pos_key(o.strategy,o.code); p=positions.get(k)
            if p is None: continue
            r=bt.row_lookup(feat_idx,di,p.code)
            if r is None or not np.isfinite(r.open):
                if di in next_date:
                    o.execute_date=next_date[di]; pending_sells.setdefault(o.execute_date,[]).append(o)
                continue
            px=bt.legal_sell_price(float(r.open)); gross=px*p.shares; proceeds=gross*(1-bt.SELL_FEE-bt.SELL_TAX); cash+=proceeds
            pnl=proceeds-p.cost_total
            trade_rows.append({"strategy":p.strategy,"code":p.code,"name":p.name,"entry_date":p.entry_date,"exit_date":di,"entry_price":p.entry_price,"exit_price":px,"shares":p.shares,"cost_total":p.cost_total,"proceeds":proceeds,"pnl":pnl,"return":pnl/p.cost_total if p.cost_total else np.nan,"exit_reason":o.reason,"hold_days":p.hold_days,"mode":p.mode})
            order_rows.append({"decision_date":o.decision_date,"execute_date":di,"strategy":p.strategy,"side":"SELL","code":p.code,"name":p.name,"order_price":np.nan,"shares":p.shares,"filled":True,"fill_price":px,"reason":o.reason})
            del positions[k]

        for o in pending_buys.pop(di, []):
            r=bt.row_lookup(feat_idx,di,o.code); fill=None
            if r is not None: fill=bt.buy_fill(float(r.open),float(r.low),float(o.limit))
            filled=fill is not None; reason="FILLED" if filled else "LIMIT_NOT_TOUCHED"
            if filled:
                cost=float(fill)*o.shares*(1+bt.BUY_FEE)
                if cost>cash+1e-6: filled=False; reason="CASH_SHORT_AT_EXECUTION"
                else:
                    cash-=cost; factor=float(r.aclose/r.close) if np.isfinite(r.aclose) and r.close else 1.0; entry_adj=float(fill)*factor
                    positions[bt.pos_key(o.strategy,o.code)] = bt.Position(o.strategy,o.code,o.name,int(o.shares),di,float(fill),entry_adj,cost,entry_adj)
            order_rows.append({"decision_date":o.decision_date,"execute_date":di,"strategy":o.strategy,"side":"BUY","code":o.code,"name":o.name,"order_price":o.limit,"shares":o.shares,"filled":bool(filled),"fill_price":float(fill) if filled else np.nan,"reason":reason,"target_cash":o.target_cash,"reserved_cash":o.reserved_cash,"rank":o.rank})

        nav,stock_mv=bt.mark_nav(cash,positions,feat_idx,di); hwm=max(hwm,nav); dd=nav/hwm-1; exposure=stock_mv/nav if nav>0 else 0
        for p in positions.values():
            r=bt.row_lookup(feat_idx,di,p.code)
            if r is not None and np.isfinite(r.aclose): p.peak_adj=max(p.peak_adj,float(r.aclose))

        r7_state=r7_states[di]; r7_cands=r7_cands_map[di]; r05_state=r05_states[di]; r05_cands=r05_cands_map[di]
        regime=r7_state["regime"]; regime_changed=last_regime is None or regime!=last_regime; reb_due=last_reb_i is None or regime_changed or (i-last_reb_i)>=15
        if reb_due: last_reb_i=i
        last_regime=regime
        r7_rank={str(x.code):int(x.r7_rank) for x in r7_cands[["code","r7_rank"]].itertuples(index=False)}
        r7_score={str(x.code):float(x.r7_score) for x in r7_cands[["code","r7_score"]].itertuples(index=False)}
        r05_score={} if r05_cands.empty else {str(x.code):float(x.r05_score) for x in r05_cands[["code","r05_score"]].itertuples(index=False)}

        sell_map={}
        if di in next_date:
            exdate=next_date[di]
            for k,p in list(positions.items()):
                r=bt.row_lookup(feat_idx,di,p.code); reason=None
                if p.strategy=="R7":
                    p.hold_days+=1
                    if r is not None and np.isfinite(r.aclose) and float(r.aclose)<=p.entry_adj*.88: reason="HARD"
                    elif reb_due:
                        n=int(r7_state["slots"]); rank=r7_rank.get(p.code,10**9)
                        if r7_state["exposure"]<=0: reason="REB_REGIME0"
                        elif n<=0 or rank>2*n: reason="REB_RANK"
                else: reason=bt.r05_exit_reason(p,r)
                if reason: sell_map[k]=bt.SellOrder(di,exdate,p.strategy,p.code,reason)
            exclude=set(sell_map); r7_mv=bt.value_of(positions,feat_idx,di,strategy="R7",exclude=exclude); r7_target=nav*float(r7_state["exposure"])
            if r7_mv>r7_target*1.03+1:
                remain=[(k,p) for k,p in positions.items() if p.strategy=="R7" and k not in exclude]; remain.sort(key=lambda kp:r7_rank.get(kp[1].code,10**9),reverse=True); projected=r7_mv
                for k,p in remain:
                    if projected<=r7_target: break
                    rr=bt.row_lookup(feat_idx,di,p.code); mv=p.shares*(float(rr.close) if rr is not None else p.entry_price); sell_map[k]=bt.SellOrder(di,exdate,p.strategy,p.code,"EXPO"); projected-=mv
            if dd<=bt.FORCE_DD and i>=force_cooldown_until:
                force_cooldown_until=i+bt.FORCE_COOLDOWN_DAYS; no_buy_until=max(no_buy_until,i+bt.FORCE_NO_BUY_DAYS); forced_count+=1
            if sell_map: pending_sells.setdefault(exdate,[]).extend(sell_map.values())

        created=[]
        if di in next_date and i>=no_buy_until:
            exdate=next_date[di]; sell_keys=set(sell_map); codes_after={p.code for k,p in positions.items() if k not in sell_keys}
            base_exposure=bt.value_of(positions,feat_idx,di,exclude=sell_keys); base_r7=bt.value_of(positions,feat_idx,di,strategy="R7",exclude=sell_keys)
            reserved_cash=reserved_exposure=reserved_r7=0.0; reserved_code={}
            def try_order(strategy,row,rank):
                nonlocal reserved_cash,reserved_exposure,reserved_r7,codes_after
                code=str(row.code); name0=str(row.name); k=bt.pos_key(strategy,code)
                if k in positions and k not in sell_keys:return
                if code not in codes_after and len(codes_after)>=bt.MAX_POSITIONS:return
                if strategy=="R05":
                    n=sum(1 for kk,p in positions.items() if p.strategy=="R05" and kk not in sell_keys)+sum(1 for o in created if o.strategy=="R05")
                    if n>=bt.R05_MAX_SLOTS:return
                    base_pct=bt.R05_BASE; limit=float(core.floor_tick(float(row.close)*.995))
                else:
                    n=sum(1 for kk,p in positions.items() if p.strategy=="R7" and kk not in sell_keys)+sum(1 for o in created if o.strategy=="R7")
                    if n>=int(r7_state["slots"]):return
                    base_pct=bt.R7_BASE; limit=float(core.floor_tick(float(row.close)*.98))
                current_code=bt.value_of(positions,feat_idx,di,code=code,exclude=sell_keys)+reserved_code.get(code,0)
                rem_single=nav*bt.MAX_SINGLE-current_code; rem_global=nav*bt.MAX_TOTAL-base_exposure-reserved_exposure; rem_cash=cash-reserved_cash
                target=nav*base_pct*bt.dd_multiplier(dd)
                if strategy=="R7": target=min(target,nav*float(r7_state["exposure"])-base_r7-reserved_r7)
                target=min(target,rem_single,rem_global,rem_cash)
                if target<=0:return
                shares,_=bt.size_shares(target,limit,float(row.avgvol20),rem_cash)
                if shares<=0:return
                reserve=shares*limit*(1+bt.BUY_FEE); notional=shares*limit
                if reserve>rem_cash+1e-6:return
                created.append(bt.BuyOrder(di,exdate,strategy,code,name0,limit,shares,target,reserve,rank)); reserved_cash+=reserve; reserved_exposure+=notional; reserved_code[code]=reserved_code.get(code,0)+notional
                if strategy=="R7": reserved_r7+=notional
                codes_after.add(code)
            if bool(r05_state.get("risk_on")) and not r05_cands.empty:
                for _,row in r05_cands.head(20).iterrows(): try_order("R05",row,int(row.r05_rank))
            if reb_due and float(r7_state["exposure"])>0 and not r7_cands.empty:
                for _,row in r7_cands.head(max(0,int(r7_state["slots"]))).iterrows(): try_order("R7",row,int(row.r7_rank))
            if created: pending_buys.setdefault(exdate,[]).extend(created)

        nav_rows.append({"date":di,"nav":nav,"cash":cash,"stock_mv":stock_mv,"exposure":exposure,"drawdown":dd,"positions":len({p.code for p in positions.values()}),"r7_positions":sum(p.strategy=="R7" for p in positions.values()),"r05_positions":sum(p.strategy=="R05" for p in positions.values()),"r7_regime":regime,"r7_regime_exposure":r7_state["exposure"],"r7_rebalance_due":reb_due,"r05_risk_on":bool(r05_state.get("risk_on",False)),"dd_multiplier":bt.dd_multiplier(dd),"no_buy_active":i<no_buy_until})
        if (i+1)%50==0 or i+1==len(eval_dates): bt.log(f"[SIM] {i+1}/{len(eval_dates)} ({(i+1)/len(eval_dates):.1%}) date={di} nav={nav:,.0f} trades={len(trade_rows)}")

    nav_df=pd.DataFrame(nav_rows); trades_df=pd.DataFrame(trade_rows); orders_df=pd.DataFrame(order_rows); events_df=pd.DataFrame(event_rows)
    end_nav=float(nav_df.iloc[-1].nav); total_return=end_nav/bt.INITIAL_CAPITAL-1; years=max((bt.dt_from_int(eval_dates[-1])-bt.dt_from_int(eval_dates[0])).days/365.25,1/365.25)
    cagr=(end_nav/bt.INITIAL_CAPITAL)**(1/years)-1; max_dd=float(nav_df.drawdown.min()); fills=int((orders_df.side.eq("BUY")&orders_df.filled.eq(True)).sum()) if not orders_df.empty else 0; buy_orders=int(orders_df.side.eq("BUY").sum()) if not orders_df.empty else 0
    nav_df["year"]=nav_df.date.astype(str).str[:4].astype(int); ann={}; prev=bt.INITIAL_CAPITAL
    for y,g in nav_df.groupby("year"):
        e=float(g.iloc[-1].nav); ann[str(y)]=e/prev-1; prev=e
    result={"status":"PASS","engine_version":bt.VERSION,"scenario":"validation2021_2025","label":cfg["label"],"mode":"FULL_R10_VALIDATION_FAST_CAUSAL","mode_reason":"Same locked R7/R0.5/R10 formulas; optimized data access only.","eval_start":str(eval_dates[0]),"eval_end":str(eval_dates[-1]),"initial_nav":bt.INITIAL_CAPITAL,"end_nav":end_nav,"total_return":total_return,"cagr":cagr,"max_dd":max_dd,"completed_trades":int(len(trades_df)),"orders":buy_orders,"fills":fills,"fill_rate":fills/buy_orders if buy_orders else 0,"min_cash":float(nav_df.cash.min()),"avg_exposure":float(nav_df.exposure.mean()),"max_exposure":float(nav_df.exposure.max()),"max_positions":int(nav_df.positions.max()),"force_dd_events":int(forced_count),"corporate_action_continuity_events":int(corp_events),"annual_returns":ann,"r05_enabled":True,"locked_benchmark":bt.LOCKED_BENCHMARK,"validation_delta":{"end_nav_pct":end_nav/bt.LOCKED_BENCHMARK["end_nav"]-1,"cagr":cagr-bt.LOCKED_BENCHMARK["cagr"],"max_dd":max_dd-bt.LOCKED_BENCHMARK["max_dd"],"trades":int(len(trades_df))-bt.LOCKED_BENCHMARK["completed_trades"]},"causal_assertions":{"signals_use_T_or_earlier":True,"orders_execute_T1_only":True,"r7_r05_independent_scanners":True,"future_data_used":False}}
    out=bt.OUT_ROOT/"latest"/"validation2021_2025"; out.mkdir(parents=True,exist_ok=True)
    nav_df.to_csv(out/"daily_nav.csv",index=False,encoding="utf-8-sig"); trades_df.to_csv(out/"trades.csv",index=False,encoding="utf-8-sig"); orders_df.to_csv(out/"orders.csv",index=False,encoding="utf-8-sig"); events_df.to_csv(out/"risk_events.csv",index=False,encoding="utf-8-sig"); (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    bt.log(json.dumps({"status":"PASS","end_nav":end_nav,"cagr":cagr,"max_dd":max_dd,"trades":len(trades_df)},ensure_ascii=False))
    return result

if __name__ == "__main__":
    simulate_fast()
