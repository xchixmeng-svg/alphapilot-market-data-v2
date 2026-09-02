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
OUT = ROOT / "artifacts" / "three_layer_strategy"
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
    x["prev_low"] = g.adj_low.shift()
    x["gap"] = x.adj_open / g.adj_close.shift() - 1
    x["body"] = x.adj_close / x.adj_open - 1
    x["quiet"] = x.vol5 / x.vol20
    x["volume_burst"] = x.volume / x.vol20
    # Sector history is absent from the immutable archive.  A market-wide
    # breadth pulse is the conservative, reproducible proxy; it never uses a
    # future classification or future return.
    breadth = x[x.base].groupby("date").apply(
        lambda z: pd.Series({"market_money_pulse": (z.volume_burst.ge(1.5)&z.ret1.gt(0)).mean()}),
        include_groups=False)
    x = x.merge(breadth, left_on="date", right_index=True, how="left")

    # Layer 1: long box, gap + bullish breakout, volume/value and institutions.
    x["short"] = (x.base & (x.box_hi60/x.box_lo60-1 <= .35) &
        (x.adj_close > x.box_hi60) & (x.gap >= .01) & (x.body >= .02) &
        (x.volume_burst >= 1.8) & (x.value20 >= 30e6) & (x.inst_net > 0) &
        (x.market_money_pulse >= .08))
    # Layer 2: five-day confirmation. Buyer/seller-count data is unavailable
    # in the source archive, so negative concentration cannot be fabricated;
    # positive institutional accumulation / volume is a declared proxy.
    x["swing"] = (x.base & (x.liquid_days5 >= 3) & (x.flow_ratio5 >= .02) &
        (x.inst5 > 0) & x.adj_close.between(x.ma20*.98, x.ma20*1.02) &
        (x.ma20_slope > 0) & (x.adj_close >= x.box_lo60))
    # Layer 3: market gate is added per decision day in simulation. Revenue is
    # point-in-time (available on the 10th of the following month).
    x["large"] = (x.base & (x.yoy > 0) & (x.accel.notna()) &
        ((x.foreign_pos5 >= 3) | (x.trust_pos5 >= 3)) & (x.quiet.shift(1) < .75) &
        (x.volume_burst >= 2) & ((x.adj_close > x.prev_hi20) | (x.adj_close > x.prev_hi100)))
    return x.sort_values(["date", "code"]).reset_index(drop=True)


def layer_score(z: pd.DataFrame, layer: str) -> pd.Series:
    if layer == "short":
        raw = z.volume_burst.rank(pct=True)+z.ret1.rank(pct=True)+z.flow_ratio5.rank(pct=True)
    elif layer == "swing":
        raw = z.flow_ratio5.rank(pct=True)+z.ma20_slope.rank(pct=True)+(-abs(z.adj_close/z.ma20-1)).rank(pct=True)
    else:
        raw = z.yoy.rank(pct=True)+z.accel.rank(pct=True)+z.volume_burst.rank(pct=True)+z.inst5.rank(pct=True)
    return raw.rank(pct=True)  # required layer-internal percentile


def simulate(x: pd.DataFrame, enabled: tuple[str,...], label: str):
    idx=x[x.code.eq("0050")].set_index("date").sort_index()
    days=list(idx.loc["2021-01-04":"2026-08-28"].index.unique())
    by={d:z.set_index("code") for d,z in x[x.date.isin(days)].groupby("date")}
    cash=INITIAL; pos={}; pending_buys=[]; pending_sells={}; ledger=[]; vals=[]
    last={}; peak=INITIAL; lock=0; cool=0; max_slots=8
    for day in days:
        bars=by[day]
        last.update({str(c):float(v) for c,v in bars.close.items() if pd.notna(v)})
        for c in list(pos):
            if c in bars.index and int(bars.at[c,"split_mult"])>1:
                m=int(bars.at[c,"split_mult"]); old=pos[c]["shares"]
                pos[c]["shares"]*=m; pos[c]["entry"]/=m; pos[c]["high"]/=m
                assert pos[c]["shares"]==old*m
        for c,reason in list(pending_sells.items()):
            if c in pos and c in bars.index:
                raw=float(bars.at[c,"open"]); fill=raw*(1-SELL_ADVERSE); sh=pos[c]["shares"]
                fee=raw*sh*FEE; tax=raw*sh*TAX; cash += fill*sh-fee-tax
                ledger.append(dict(execute_date=day,decision_date=pos[c]["sell_decision"],side="SELL",code=c,
                    layer=pos[c]["layer"],shares=sh,raw_price=raw,fill_price=fill,commission=fee,tax=tax,reason=reason))
                del pos[c]; del pending_sells[c]
        for order in pending_buys:
            c=order["code"]
            if c in pos or c not in bars.index: continue
            op,lo=float(bars.at[c,"open"]),float(bars.at[c,"low"]); limit=order["limit"]
            fill=min(limit,op*1.005) if op<=limit else (limit if lo<=limit else None)
            if fill is None: continue
            nav=cash+sum(p["shares"]*last[k] for k,p in pos.items())
            budget=min(order["budget"],nav*MAX_W,cash/(1+FEE)); sh=int(budget/fill//LOT*LOT)
            while sh>0 and sh*fill+sh*op*FEE>cash: sh-=LOT
            if sh<=0: continue
            fee=sh*op*FEE; cash-=sh*fill+fee
            pos[c]={"shares":sh,"entry":fill,"high":float(bars.at[c,"close"]),"days":0,
                    "layer":order["layer"],"box_hi":order["box_hi"]}
            ledger.append(dict(execute_date=day,decision_date=order["decision_date"],side="BUY",code=c,
                layer=order["layer"],shares=sh,raw_price=op,fill_price=fill,commission=fee,tax=0,reason="signal"))
        pending_buys=[]
        nav=cash+sum(p["shares"]*last[c] for c,p in pos.items()); peak=max(peak,nav); dd=nav/peak-1
        vals.append((day,nav,cash,len(pos),dd))
        for c,p in pos.items():
            if c not in bars.index: continue
            r=bars.loc[c]; cl=float(r.close); p["days"]+=1; p["high"]=max(p["high"],cl); reason=None
            if p["layer"]=="short":
                if cl < p["box_hi"]*.99: reason="box_failure"
                elif p["days"]>1 and cl<float(r.ma3): reason="ma3_break"
                elif p["days"]>1 and cl<float(r.prev_low): reason="prior_low_break"
                elif r.inst_net<0 and r.volume<r.vol20: reason="flow_cools"
            elif p["layer"]=="swing":
                if cl<float(r.ma20) or r.ma20_slope<=0: reason="ma20_break"
                elif cl/p["high"]-1<=-.08: reason="trail_8pct"
            else:
                if cl/p["high"]-1<=-.10: reason="trail_10pct"
                elif p["days"]>=60: reason="max_60_days"
            if reason and p["days"]>=1:
                p["sell_decision"]=day; pending_sells[c]=reason
        if dd<=-.14 and cool==0:
            lock=10; cool=15
            exposure=sum(pos[c]["shares"]*last[c] for c in pos)/nav
            for c in sorted(pos,key=lambda k:pos[k]["days"],reverse=True):
                if exposure<=.50: break
                pos[c]["sell_decision"]=day; pending_sells[c]="portfolio_dd_guard"
                exposure-=pos[c]["shares"]*last[c]/nav
        lock=max(0,lock-1); cool=max(0,cool-1)
        if lock or day not in idx.index: continue
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
                candidates.extend((c,layer,float(z.at[c,"pct_score"]),float(z.at[c,"box_hi60"])) for c in z.index)
        candidates=sorted(candidates,key=lambda q:q[2],reverse=True)
        chosen=[]; seen=set()
        for q in candidates:
            if q[0] not in seen: chosen.append(q); seen.add(q[0])
            if len(chosen)>=slots: break
        target=max(0,exposure*nav-sum(pos[c]["shares"]*last[c] for c in pos)); each=min(nav*MAX_W,target/max(1,len(chosen)))
        pending_buys=[dict(code=c,layer=l,limit=float(bars.at[c,"close"])*.98,budget=each,
                           decision_date=day,box_hi=box) for c,l,_,box in chosen]
    curve=pd.DataFrame(vals,columns=["date","nav","cash","holdings","drawdown"]).set_index("date")
    years=(curve.index[-1]-curve.index[0]).days/365.2425
    result=dict(strategy=label,final_nav=float(curve.nav.iloc[-1]),total_return=float(curve.nav.iloc[-1]/INITIAL-1),
        cagr=float((curve.nav.iloc[-1]/INITIAL)**(1/years)-1),max_drawdown=float(curve.drawdown.min()),
        buys=sum(t["side"]=="BUY" for t in ledger),sells=sum(t["side"]=="SELL" for t in ledger))
    return result,curve,pd.DataFrame(ledger)


def audit(x, results, ledgers):
    checks={
        "coverage_2021_2026": set(range(2021,2027)).issubset(set(x.date.dt.year.unique())),
        "0050_market_gate_present": bool(x.code.eq("0050").any()),
        "all_shares_integer_100_step": all(t.empty or ((t.shares.astype(int)==t.shares)&(t.shares%100==0)).all() for t in ledgers),
        "all_execution_after_decision": all(t.empty or (pd.to_datetime(t.execute_date)>pd.to_datetime(t.decision_date)).all() for t in ledgers),
        "split_notional_audit_exists": (core.OUT/"corporate_action_audit.csv").exists(),
        "performance_finite": all(np.isfinite(r["final_nav"]) and np.isfinite(r["max_drawdown"]) for r in results),
    }
    (OUT/"contract_audit.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(checks.values()): raise RuntimeError(f"contract audit failed: {checks}")


def main():
    raw=core.load_ohlcv(); revenue=core.fetch_revenue(); x=add_features(raw,revenue)
    configs=[(("short",),"short_layer"),(("swing",),"swing_layer"),(("large",),"large_layer"),
             (("short","swing","large"),"combined_three_layer")]
    results=[]; curves=[]; ledgers=[]
    for layers,label in configs:
        r,c,t=simulate(x,layers,label); results.append(r); ledgers.append(t); curves.append(c.nav.rename(label))
        t.to_csv(OUT/f"trades_{label}.csv",index=False); print(label,r,flush=True)
    bm,bc=core.benchmark(x); results.append(bm); curves.append(bc.rename("0050_BH"))
    pd.DataFrame(results).to_csv(OUT/"performance_summary.csv",index=False)
    pd.concat(curves,axis=1).to_csv(OUT/"equity_curves.csv")
    audit(x,results,ledgers)
    report={"period":{"warmup":"2020","train":"2021-2023","test":"2024-2025","blind":"2026-01-01..2026-08-28"},
      "execution":"T close decision, T+1 precommitted 98% close limit, sells next open less 2%, full fee/tax, 100-share step",
      "data_limitations":{"sector_flow":"market breadth proxy (archive has no point-in-time sector map)",
        "broker_concentration":"institutional net/volume proxy; buyer/seller broker counts absent and never fabricated"},
      "results":results}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
