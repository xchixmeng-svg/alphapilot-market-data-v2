"""Causal validation for the three-layer AlphaPilot specification.

The implementation deliberately reuses the independently audited loader from
``validate_six_factor_strategies``.  Raw prices are used for execution and
split-adjusted prices only for signals; held share counts are multiplied on a
detected split.  All decisions are made at T close and execute on T+1.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import validate_six_factor_strategies as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "three_layer_strategy_v3"
OUT.mkdir(parents=True, exist_ok=True)
INITIAL = 1_000_000.0
FEE, TAX, SELL_ADVERSE, LOT, MAX_W = .001425, .003, .02, 100, .20


def add_features(raw: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    x = core.features(raw, revenue).sort_values(["code", "date"]).copy()
    inst = pd.read_parquet(core.DATA / "institutional_2020_2025.parquet")
    inst.columns = [str(c).lower() for c in inst.columns]
    inst["code"] = inst.code.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    inst["date"] = pd.to_datetime(inst.date.astype(str), format="%Y%m%d", errors="coerce")
    for c in ("foreign_net", "trust_net", "dealer_net"):
        inst[c] = pd.to_numeric(inst[c], errors="coerce").fillna(0)
    inst["inst_net"] = inst[["foreign_net", "trust_net", "dealer_net"]].sum(axis=1)
    x = x.merge(inst[["date", "code", "foreign_net", "trust_net", "dealer_net", "inst_net"]],
                on=["date", "code"], how="left")
    for c in ("foreign_net", "trust_net", "dealer_net", "inst_net"):
        x[c] = x[c].fillna(0)
    g = x.groupby("code", sort=False)
    x["prev_hi20"] = g.adj_high.transform(lambda s: s.rolling(20, min_periods=20).max().shift())
    x["prev_hi100"] = g.adj_high.transform(lambda s: s.rolling(100, min_periods=100).max().shift())
    x["box_hi60"] = g.adj_high.transform(lambda s: s.rolling(60, min_periods=60).max().shift())
    x["box_lo60"] = g.adj_low.transform(lambda s: s.rolling(60, min_periods=60).min().shift())
    x["ma3"] = g.adj_close.transform(lambda s: s.rolling(3, min_periods=3).mean())
    x["ma5"] = g.adj_close.transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["ma60"] = g.adj_close.transform(lambda s: s.rolling(60, min_periods=60).mean())
    x["vol20"] = g.volume.transform(lambda s: s.rolling(20, min_periods=20).mean())
    x["vol5"] = g.volume.transform(lambda s: s.rolling(5, min_periods=5).mean())
    x["value20"] = (x.close*x.volume).groupby(x.code).transform(lambda s:s.rolling(20,min_periods=20).mean())
    x["inst5"] = g.inst_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
    x["foreign_pos5"] = g.foreign_net.transform(lambda s:s.gt(0).rolling(5,min_periods=5).sum())
    x["trust_pos5"] = g.trust_net.transform(lambda s:s.gt(0).rolling(5,min_periods=5).sum())
    x["liquid_days5"] = g.volume.transform(lambda s:s.ge(5_000_000).rolling(5,min_periods=5).sum())
    x["flow_ratio5"] = x.inst5 / g.volume.transform(lambda s:s.rolling(5,min_periods=5).sum()).replace(0,np.nan)
    x["ma20_slope"] = g.ma20.diff(5)
    x["ret1"] = g.adj_close.pct_change()
    x["ret_std20"] = g.adj_close.transform(lambda s:s.pct_change().rolling(20,min_periods=20).std())
    x["prev_low"] = g.adj_low.shift()
    x["gap"] = x.adj_open / g.adj_close.shift() - 1
    x["body"] = x.adj_close / x.adj_open - 1
    x["quiet"] = x.vol5 / x.vol20
    x["quiet_prev"] = x.groupby("code",sort=False).quiet.shift()
    x["volume_burst"] = x.volume / x.vol20
    x["support_low20"] = g.adj_close.transform(lambda s:s.rolling(20,min_periods=20).min().shift())
    x["prev_adj_high"] = g.adj_high.shift()
    x["inst3"] = g.inst_net.transform(lambda s:s.rolling(3,min_periods=3).sum().shift())
    real_body=(x.adj_close-x.adj_open).abs()
    upper_shadow=x.adj_high-x[["adj_open","adj_close"]].max(axis=1)
    x["short_breakout"]=(x.base & (x.adj_close>=x.box_hi60*1.02) &
        (x.volume_burst>=2.5) & (real_body>0) & (upper_shadow<=real_body*.30))
    x["accel_prev90"] = g.accel.transform(lambda s:s.rolling(90,min_periods=30).mean().shift())
    x["inst_sell5"] = g.inst_net.transform(lambda s:s.lt(0).rolling(5,min_periods=5).sum())
    x["inst_sell5_recent10"] = x.groupby("code",sort=False).inst_sell5.transform(
        lambda s:s.rolling(10,min_periods=1).max())
    # Sector history is absent from the immutable archive.  A market-wide
    # breadth pulse is the conservative, reproducible proxy; it never uses a
    # future classification or future return.
    breadth = x[x.base].groupby("date").apply(
        lambda z: pd.Series({"market_money_pulse": (z.volume_burst.ge(1.5)&z.ret1.gt(0)).mean()}),
        include_groups=False)
    x = x.merge(breadth, left_on="date", right_index=True, how="left")

    # V3 signals are causal multi-day patterns.  The stored reference price is
    # carried into the order and position so exits use the actual setup level.
    x["short"]=False; x["short_ref"]=np.nan
    green_confirm=(x.adj_close>x.adj_open)&(x.volume>=x.vol20*1.2)
    # Evaluate each permissible breakout age with grouped shifts/rolling
    # windows.  This is equivalent to a per-stock state machine but scales to
    # the full three-million-row archive.
    for age in range(1,6):
        prior_breakout=x.short_breakout.groupby(x.code,sort=False).shift(age).fillna(False)
        box=g.box_hi60.shift(age).where(prior_breakout)
        bvol=g.volume.shift(age).where(prior_breakout)
        min_close=g.adj_close.transform(lambda s:s.rolling(age,min_periods=age).min())
        min_low=g.adj_low.transform(lambda s:s.rolling(age,min_periods=age).min())
        max_vol=g.volume.transform(lambda s:s.rolling(age,min_periods=age).max())
        ok=(box.notna()&(min_close>=box*.99)&(min_low<=box*1.01)&
            (max_vol<bvol*.60)&green_confirm&~x.short)
        x.loc[ok,"short"]=True; x.loc[ok,"short_ref"]=box[ok]
    near_support=x.support_low20.notna()&x.adj_close.between(x.support_low20,x.support_low20*1.02)
    x["swing"]=False; x["swing_ref"]=np.nan
    rebound=(x.adj_close>x.prev_adj_high)&(x.volume>=x.vol20*1.2)&(x.inst3>0)
    for age in range(1,4):
        touched=near_support.groupby(x.code,sort=False).shift(age).fillna(False)
        support=x.support_low20.groupby(x.code,sort=False).shift(age)
        ok=touched&rebound&~x.swing
        x.loc[ok,"swing"]=True; x.loc[ok,"swing_ref"]=support[ok]
    # Layer 3: market gate is added per decision day in simulation. Revenue is
    # point-in-time (available on the 10th of the following month).
    x["large"] = (x.base & (x.yoy > 0) & (x.accel.notna()) &
        ((x.foreign_pos5 >= 3) | (x.trust_pos5 >= 3)) & (x.quiet_prev < .75) &
        (x.volume_burst >= 2) & ((x.adj_close > x.prev_hi20) | (x.adj_close > x.prev_hi100)))
    return x.sort_values(["date", "code"]).reset_index(drop=True)


def layer_score(z: pd.DataFrame, layer: str) -> pd.Series:
    if layer == "short":
        raw = z.volume_burst.rank(pct=True)+z.ret1.rank(pct=True)+z.flow_ratio5.rank(pct=True)
    elif layer == "swing":
        raw = z.flow_ratio5.rank(pct=True)+z.volume_burst.rank(pct=True)+(-abs(z.adj_close/z.swing_ref-1)).rank(pct=True)
    else:
        raw = z.yoy.rank(pct=True)+z.accel.rank(pct=True)+z.volume_burst.rank(pct=True)+z.inst5.rank(pct=True)
    return raw.rank(pct=True)  # required layer-internal percentile


def simulate(x: pd.DataFrame, enabled: tuple[str,...], label: str):
    idx=x[x.code.eq("0050")].set_index("date").sort_index()
    days=list(idx.loc["2021-01-04":"2026-08-28"].index.unique())
    by={d:z.set_index("code") for d,z in x[x.date.isin(days)].groupby("date")}
    cash=INITIAL; pos={}; pending_buys=[]; pending_sells={}; ledger=[]; rejected=[]; vals=[]; split_events=[]
    last={}; peak=INITIAL; max_slots=8; dd_events=[]
    for day in days:
        bars=by[day]
        last.update({str(c):float(v) for c,v in bars.close.items() if pd.notna(v)})
        for c in list(pos):
            if c in bars.index and int(bars.at[c,"split_mult"])>1:
                m=int(bars.at[c,"split_mult"]); old=pos[c]["shares"]; old_entry=pos[c]["entry"]
                pos[c]["shares"]*=m; pos[c]["entry"]/=m; pos[c]["high"]/=m
                assert pos[c]["shares"]==old*m
                split_events.append(dict(date=day,code=c,multiplier=m,shares_before=old,
                    shares_after=pos[c]["shares"],entry_before=old_entry,entry_after=pos[c]["entry"]))
        for c,reason in list(pending_sells.items()):
            if c in pos and c in bars.index:
                raw=float(bars.at[c,"open"]); fill=raw*(1-SELL_ADVERSE); sh=pos[c]["shares"]
                fee=fill*sh*FEE; tax=fill*sh*TAX; cash += fill*sh-fee-tax
                ledger.append(dict(execute_date=day,decision_date=pos[c]["sell_decision"],side="SELL",code=c,
                    layer=pos[c]["layer"],cash_pool="shared_1m_account",shares=sh,raw_price=raw,fill_price=fill,commission=fee,tax=tax,reason=reason))
                del pos[c]; del pending_sells[c]
        for order in pending_buys:
            c=order["code"]
            if c in pos or c not in bars.index: continue
            op,lo=float(bars.at[c,"open"]),float(bars.at[c,"low"]); limit=order["limit"]
            fill=min(limit,op*1.005) if op<=limit else (limit if lo<=limit else None)
            if fill is None: continue
            nav=cash+sum(p["shares"]*last[k] for k,p in pos.items())
            budget=min(order["budget"],nav*MAX_W,cash/(1+FEE)); sh=int(budget/fill//LOT*LOT)
            while sh>0 and sh*fill*(1+FEE)>cash: sh-=LOT
            if sh<=0: continue
            fee=sh*fill*FEE; cash-=sh*fill+fee
            pos[c]={"shares":sh,"entry":fill,"high":float(bars.at[c,"close"]),"days":0,
                    "layer":order["layer"],"box_hi":order["box_hi"],"support":order["support"],
                    "below_box_days":0,"below_ma5_days":0}
            ledger.append(dict(execute_date=day,decision_date=order["decision_date"],side="BUY",code=c,
                layer=order["layer"],cash_pool="shared_1m_account",shares=sh,raw_price=op,fill_price=fill,
                commission=fee,tax=0,reason="signal",expected_rr=order["expected_rr"]))
            if cash < -1e-6: raise RuntimeError(f"shared cash pool went negative: {cash}")
        pending_buys=[]
        nav=cash+sum(p["shares"]*last[c] for c,p in pos.items()); peak=max(peak,nav); dd=nav/peak-1
        vals.append((day,nav,cash,len(pos),dd))
        for c,p in pos.items():
            if c not in bars.index: continue
            r=bars.loc[c]; cl=float(r.close); p["days"]+=1; p["high"]=max(p["high"],cl); reason=None
            if p["layer"]=="short":
                p["below_box_days"]=p["below_box_days"]+1 if cl<p["box_hi"] else 0
                if p["below_box_days"]>=2: reason="box_failure_two_closes_inside"
                elif cl<p["box_hi"]*.96 and r.volume>=r.vol20: reason="box_failure_volume_below_minus_4pct"
                elif p["days"]>1 and cl<float(r.ma3): reason="ma3_break"
                elif p["days"]>1 and cl<float(r.prev_low): reason="prior_low_break"
                elif r.inst_net<0 and r.volume<r.vol20: reason="flow_cools"
            elif p["layer"]=="swing":
                p["below_ma5_days"]=p["below_ma5_days"]+1 if cl<float(r.ma5) else 0
                if cl<p["support"]*.97: reason="prior_swing_low_minus_3pct"
                elif cl/p["high"]-1<=-.08: reason="trail_8pct"
                elif p["below_ma5_days"]>=2: reason="two_closes_below_ma5"
            else:
                gain=p["high"]/p["entry"]-1
                if cl/p["entry"]-1<=-.15: reason="large_hard_stop_minus_15pct"
                elif gain>=.15 and cl/p["high"]-1<=-.13: reason="trail_13pct_after_gain_15pct"
                elif p["days"]>90 and not (pd.notna(r.accel_prev90) and r.accel>r.accel_prev90 and r.inst_sell5_recent10<5):
                    reason="over_90d_continuation_failed"
            if reason and p["days"]>=1:
                p["sell_decision"]=day; pending_sells[c]=reason
        # V3 portfolio guard: never indiscriminately liquidate the large layer.
        if dd<=-.06:
            for layer in ("short","swing"):
                members=[c for c,p in pos.items() if p["layer"]==layer and c not in pending_sells]
                exposure=sum(pos[c]["shares"]*last[c] for c in members)
                for c in sorted(members,key=lambda q:last[q]/pos[q]["entry"]-1):
                    if exposure<=nav*.50: break
                    pos[c]["sell_decision"]=day; pending_sells[c]="portfolio_dd6_reduce_layer_to_50pct"
                    exposure-=pos[c]["shares"]*last[c]
        if dd<=-.09:
            for c,p in pos.items():
                if p["layer"]=="short" and c not in pending_sells:
                    p["sell_decision"]=day; pending_sells[c]="portfolio_dd9_exit_short"
            swing=sorted((c for c,p in pos.items() if p["layer"]=="swing" and c not in pending_sells),
                         key=lambda c:last[c]/pos[c]["entry"]-1,reverse=True)
            for c in swing[1:]:
                pos[c]["sell_decision"]=day; pending_sells[c]="portfolio_dd9_keep_best_swing"
        if dd<=-.14:
            for c,p in pos.items():
                if p["layer"]=="large" and c not in pending_sells and c in bars.index:
                    cl=float(bars.at[c,"close"]); ma60=float(bars.at[c,"ma60"])
                    if cl<ma60 or cl/p["entry"]-1<=-.15:
                        p["sell_decision"]=day; pending_sells[c]="portfolio_dd14_large_structure_broken"
        dd_events.append((day,dd,len(pending_sells)))
        if dd<=-.12 or day not in idx.index: continue
        mr=idx.loc[day]
        exposure=.85 if mr.adj_close>mr.ma60 and mr.adj_close>mr.ma200 else (.60 if mr.adj_close>mr.ma200 else (.35 if mr.adj_close>mr.ma60 else .15))
        slots=max_slots-len(pos)-len(pending_sells)
        if slots<=0: continue
        candidates=[]
        for layer in enabled:
            z=bars[bars[layer] & ~bars.index.isin(pos)].copy()
            if layer=="large" and not ((mr.adj_close>mr.ma20) or (mr.adj_close>mr.ma60)): continue
            if not z.empty:
                z["pct_score"]=layer_score(z,layer)
                candidates.extend((c,layer,float(z.at[c,"pct_score"]),float(z.at[c,"short_ref"])
                    if layer=="short" else np.nan,float(z.at[c,"swing_ref"])
                    if layer=="swing" else np.nan) for c in z.index)
        candidates=sorted(candidates,key=lambda q:q[2],reverse=True)
        chosen=[]; seen=set()
        for q in candidates:
            if q[0] not in seen: chosen.append(q); seen.add(q[0])
            if len(chosen)>=slots: break
        target=max(0,exposure*nav-sum(pos[c]["shares"]*last[c] for c in pos)); each=min(nav*MAX_W,target/max(1,len(chosen)))
        pending_buys=[]
        for c,l,_,box,support in chosen:
            row=bars.loc[c]
            discount=min(.015,float(row.ret_std20)) if pd.notna(row.ret_std20) else .015
            limit=float(row.close)*(1-discount) if l=="large" else float(row.close)*.98
            stop=box*.96 if l=="short" else (support*.97 if l=="swing" else limit*.85)
            reward=limit*(1.08 if l=="short" else (1.15 if l=="swing" else 1.25))
            risk=limit-stop
            rr=(reward-limit)/risk if risk>0 else -np.inf
            if rr<1.5:
                rejected.append(dict(decision_date=day,code=c,layer=l,reason="expected_rr_below_1p5",
                                     limit=limit,stop=stop,reward=reward,expected_rr=rr))
                continue
            pending_buys.append(dict(code=c,layer=l,limit=limit,budget=each,
                               decision_date=day,box_hi=box,support=support,expected_rr=rr))
    curve=pd.DataFrame(vals,columns=["date","nav","cash","holdings","drawdown"]).set_index("date")
    years=(curve.index[-1]-curve.index[0]).days/365.2425
    result=dict(strategy=label,final_nav=float(curve.nav.iloc[-1]),total_return=float(curve.nav.iloc[-1]/INITIAL-1),
        cagr=float((curve.nav.iloc[-1]/INITIAL)**(1/years)-1),max_drawdown=float(curve.drawdown.min()),
        buys=sum(t["side"]=="BUY" for t in ledger),sells=sum(t["side"]=="SELL" for t in ledger))
    return result,curve,pd.DataFrame(ledger),pd.DataFrame(split_events),pd.DataFrame(rejected)


def audit(x, results, ledgers, curves, split_audits, rejected_orders):
    market_days=sorted(x.loc[x.code.eq("0050"),"date"].drop_duplicates())
    next_market_day={market_days[i]:market_days[i+1] for i in range(len(market_days)-1)}
    exact_t1=[]; fee_tax=[]
    for t in ledgers:
        if t.empty: continue
        exact_t1.extend(next_market_day.get(pd.Timestamp(d))==pd.Timestamp(e)
                        for d,e in zip(t.decision_date,t.execute_date))
        buy=t.side.eq("BUY"); sell=t.side.eq("SELL")
        fee_tax.append(np.allclose(t.commission,t.shares*t.fill_price*FEE,rtol=0,atol=.01))
        fee_tax.append((t.loc[buy,"tax"].abs()<=.01).all())
        fee_tax.append(np.allclose(t.loc[sell,"tax"],t.loc[sell,"shares"]*t.loc[sell,"fill_price"]*TAX,rtol=0,atol=.01))
    ca=pd.read_csv(core.OUT/"corporate_action_audit.csv")
    held_split_ok=True
    for s in split_audits:
        if s.empty: continue
        held_split_ok &= bool(((s.shares_after==s.shares_before*s.multiplier) &
                          np.isclose(s.entry_after*s.multiplier,s.entry_before,rtol=0,atol=1e-9)).all())
    combined=ledgers[3]
    checks={
        "coverage_2021_2026": set(range(2021,2027)).issubset(set(x.date.dt.year.unique())),
        "0050_market_gate_and_benchmark_present": bool(x.code.eq("0050").any()) and any(r["strategy"]=="0050_BH" for r in results),
        "all_shares_integer_100_step": all(t.empty or ((t.shares.astype(int)==t.shares)&(t.shares%100==0)).all() for t in ledgers),
        "all_execution_exact_next_market_day": bool(exact_t1) and all(exact_t1),
        "shared_cash_pool_single_nonnegative": (not combined.empty and combined.cash_pool.nunique()==1 and
            all((curve.cash>=-1e-6).all() and (curve.holdings<=8).all() for curve in curves)),
        "fees_and_sell_tax_use_fill_price": bool(fee_tax) and all(fee_tax),
        "split_price_notional_invariant": (not ca.empty and ca.invariant_pass.astype(str).str.lower().eq("true").all()),
        "held_split_share_and_entry_invariant": bool(held_split_ok),
        "expected_reward_risk_at_least_1p5": (all(
            t.empty or (t.loc[t.side.eq("BUY"),"expected_rr"]>=1.5).all() for t in ledgers) and all(
            j.empty or (j.expected_rr<1.5).all() for j in rejected_orders)),
        "rejected_orders_have_reasons": all(j.empty or j.reason.notna().all() for j in rejected_orders),
        "performance_finite": all(np.isfinite(r["final_nav"]) and np.isfinite(r["max_drawdown"]) for r in results),
        "v3_short_signals_have_box_reference": bool(x.loc[x.short,"short_ref"].notna().all()),
        "v3_swing_signals_have_support_reference": bool(x.loc[x.swing,"swing_ref"].notna().all()),
        "v3_swing_has_no_ma20_exit": all(t.empty or not t.reason.astype(str).str.contains("ma20_break").any() for t in ledgers),
        "v3_large_has_no_indiscriminate_dd_exit": all(t.empty or not t.reason.astype(str).eq("portfolio_dd_guard").any() for t in ledgers),
    }
    checks={k:bool(v) for k,v in checks.items()}
    (OUT/"contract_audit.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(checks.values()): raise RuntimeError(f"contract audit failed: {checks}")


def completed_trade_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Pair each buy with its later sell and attribute net P&L to exit reason."""
    if ledger.empty: return pd.DataFrame()
    opened={}; rows=[]
    for r in ledger.itertuples(index=False):
        if r.side=="BUY": opened[str(r.code)]=r
        else:
            b=opened.pop(str(r.code),None)
            if b is None: continue
            buy_cost=b.shares*b.fill_price+b.commission+b.tax
            sell_net=r.shares*r.fill_price-r.commission-r.tax
            pnl=sell_net-buy_cost
            rows.append(dict(code=str(r.code),layer=r.layer,exit_reason=r.reason,
                buy_date=b.execute_date,sell_date=r.execute_date,pnl=pnl,
                return_pct=pnl/buy_cost,win=pnl>0))
    return pd.DataFrame(rows)


def main():
    raw=core.load_ohlcv(); revenue=core.fetch_revenue(); x=add_features(raw,revenue)
    configs=[(("short",),"short_layer"),(("swing",),"swing_layer"),(("large",),"large_layer"),
             (("short","swing","large"),"combined_three_layer")]
    results=[]; curves=[]; ledgers=[]; split_audits=[]; rejected_orders=[]
    for layers,label in configs:
        r,c,t,s,j=simulate(x,layers,label); results.append(r); ledgers.append(t); curves.append(c)
        split_audits.append(s); rejected_orders.append(j); t.to_csv(OUT/f"trades_{label}.csv",index=False)
        j.to_csv(OUT/f"rejected_orders_{label}.csv",index=False)
        s.to_csv(OUT/f"held_split_audit_{label}.csv",index=False); print(label,r,flush=True)
    bm,bc=core.benchmark(x); results.append(bm)
    pd.DataFrame(results).to_csv(OUT/"performance_summary.csv",index=False)
    pd.concat([curve.nav.rename(configs[i][1]) for i,curve in enumerate(curves)]+[bc.rename("0050_BH")],axis=1).to_csv(OUT/"equity_curves.csv")
    annual=[]
    named_curves={configs[i][1]:curve.nav for i,curve in enumerate(curves)}
    named_curves["0050_BH"]=bc
    for strategy,series in named_curves.items():
        for year,z in series.dropna().groupby(series.dropna().index.year):
            if 2021<=year<=2026 and len(z)>1:
                annual.append(dict(strategy=strategy,year=int(year),start_nav=float(z.iloc[0]),
                    end_nav=float(z.iloc[-1]),year_return=float(z.iloc[-1]/z.iloc[0]-1),
                    year_max_drawdown=float((z/z.cummax()-1).min())))
    pd.DataFrame(annual).to_csv(OUT/"yearly_performance.csv",index=False)
    exits=[]
    for i,t in enumerate(ledgers):
        label=configs[i][1]
        paired=completed_trade_rows(t)
        paired.to_csv(OUT/f"completed_trades_{label}.csv",index=False)
        if paired.empty: continue
        for (layer,reason),z in paired.groupby(["layer","exit_reason"]):
            exits.append(dict(strategy=label,layer=layer,reason=reason,count=len(z),
                wins=int(z.win.sum()),win_rate=float(z.win.mean()),net_pnl=float(z.pnl.sum()),
                average_pnl=float(z.pnl.mean()),average_return=float(z.return_pct.mean())))
    pd.DataFrame(exits).to_csv(OUT/"exit_reason_distribution.csv",index=False)
    audit(x,results,ledgers,curves,split_audits,rejected_orders)
    report={"period":{"warmup":"2020","train":"2021-2023","test":"2024-2025","blind":"2026-01-01..2026-08-28"},
      "version":"V3 fixed numeric specification",
      "execution":"T close decision, T+1 precommitted limit (large layer volatility discount; other layers 98% close), sells next open less 2%, full fee/tax, 100-share step",
      "validation_gates":{"short_complete_trades_at_least_80":int(results[0]["sells"])>=80,
        "short_complete_trades":int(results[0]["sells"])},
      "data_limitations":{"sector_flow":"market breadth proxy (archive has no point-in-time sector map)",
        "broker_concentration":"institutional net/volume proxy; buyer/seller broker counts absent and never fabricated",
        "validation_warning":"V2 was designed after observing V1; historical results are reference only and require at least three months forward validation"},
      "results":results}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
