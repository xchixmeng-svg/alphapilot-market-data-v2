"""Stage-2 causal filter research for AlphaPilot six-factor study."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import validate_six_factor_strategies as core

OUT=core.ROOT/"artifacts"/"filtered_factor_research"
OUT.mkdir(parents=True,exist_ok=True)

def period_metrics(curve,lo,hi):
    q=curve.loc[lo:hi].dropna()
    if len(q)<2:return {"return":np.nan,"cagr":np.nan,"max_drawdown":np.nan}
    base=float(q.iloc[0]); years=(q.index[-1]-q.index[0]).days/365.2425
    dd=q/q.cummax()-1
    return {"return":float(q.iloc[-1]/base-1),"cagr":float((q.iloc[-1]/base)**(1/years)-1),
            "max_drawdown":float(dd.min())}

def candidate_specs():
    seeds=["A","B","D","E","A&D","A&E","B&D","B&E","D&E","A&D&E"]
    trends=["none","rising60","above60","ordered"]
    groups=["none","near90","mom_0_30","quiet_volume","revenue_yoy10"]
    liquid=[30,50,100]
    return [{"seed":s,"trend":t,"quality":q,"adv_m":a}
            for s in seeds for t in trends for q in groups for a in liquid]

def mask_for(x,p):
    z=pd.Series(True,index=x.index)
    for name in p["seed"].split("&"): z &= x[name]
    if p["trend"]=="rising60": z &= x.ma60.gt(x.ma60_lag20)
    elif p["trend"]=="above60": z &= x.adj_close.gt(x.ma60)
    elif p["trend"]=="ordered": z &= x.D
    if p["quality"]=="near90": z &= x.near_high.ge(.90)
    elif p["quality"]=="mom_0_30": z &= x.ret20.between(0,.30)
    elif p["quality"]=="quiet_volume": z &= x.vol_accel.lt(.50)
    elif p["quality"]=="revenue_yoy10": z &= x.yoy.ge(.10)
    z &= x.adv20.ge(p["adv_m"]*1e6)
    return z & x.base

def composite_score(z,_):
    cols=[]
    for c,ascending in (("near_high",True),("ret20",True),("accel",True),("breakout",True),("vol_accel",False)):
        if c in z:
            cols.append(z[c].rank(pct=True,ascending=ascending).fillna(.5))
    return sum(cols)/len(cols) if cols else pd.Series(0,index=z.index)

def signal_screen(x,specs):
    x=x.copy()
    x["fwd20"]=x.groupby("code").adj_close.shift(-20)/x.adj_close-1
    x["fwd40"]=x.groupby("code").adj_close.shift(-40)/x.adj_close-1
    rows=[]
    for i,p in enumerate(specs):
        m=mask_for(x,p); row={"candidate":f"F{i:04d}",**p,"signals":int(m.sum())}; valid=True
        for label,lo,hi in (("train","2021-01-01","2023-12-31"),("test","2024-01-01","2025-12-31")):
            for h in (20,40):
                q=x[m & x.date.between(lo,hi)].dropna(subset=[f"fwd{h}"])
                daily=q.groupby("date")[f"fwd{h}"].mean()
                row[f"{label}_{h}_days"]=len(daily)
                row[f"{label}_{h}_mean"]=float(daily.mean()) if len(daily) else np.nan
                row[f"{label}_{h}_t"]=float(core.hac_t(daily,h-1)) if len(daily) else np.nan
                if len(daily)<100: valid=False
        row["screen_valid"]=valid
        vals=[row["train_20_t"],row["train_40_t"],row["test_20_t"],row["test_40_t"]]
        means=[row["train_20_mean"],row["train_40_mean"],row["test_20_mean"],row["test_40_mean"]]
        row["robust_signal_score"]=min(vals)+20*min(means) if valid and all(np.isfinite(vals+means)) else -999
        rows.append(row)
    return pd.DataFrame(rows).sort_values("robust_signal_score",ascending=False)

def main():
    d=core.load_ohlcv(); rev=core.fetch_revenue(); x=core.features(d,rev)
    x["ma60_lag20"]=x.groupby("code").ma60.shift(20)
    specs=candidate_specs(); screen=signal_screen(x,specs)
    screen.to_csv(OUT/"all_signal_candidates.csv",index=False)
    shortlist=screen[(screen.screen_valid)&(screen.train_20_mean>0)&(screen.test_20_mean>0)&
                     (screen.train_40_mean>0)&(screen.test_40_mean>0)].head(30).copy()
    if len(shortlist)<5: raise RuntimeError(f"too few train/test-confirmed candidates: {len(shortlist)}")
    core.score=composite_score
    perf=[]; curves={}; ledgers={}
    for _,r in shortlist.iterrows():
        cid=r.candidate; p={k:r[k] for k in ("seed","trend","quality","adv_m")}
        x["RESEARCH"]=mask_for(x,p)
        m,c,t=core.simulate(x,"RESEARCH")
        tr=period_metrics(c,"2021-01-01","2023-12-31")
        te=period_metrics(c,"2024-01-01","2025-12-31")
        bl=period_metrics(c,"2026-01-01","2026-12-31")
        row={"candidate":cid,**p,"full_final_nav":m["final_nav"],"full_cagr":m["cagr"],
             "full_mdd":m["max_drawdown"],"buys":m["buys"],
             **{f"train_{k}":v for k,v in tr.items()},
             **{f"test_{k}":v for k,v in te.items()},
             **{f"blind2026_{k}":v for k,v in bl.items()}}
        row["selection_pass"]=bool(tr["cagr"]>0 and te["cagr"]>0 and tr["max_drawdown"]>-.35 and te["max_drawdown"]>-.35)
        row["selection_score"]=min(tr["cagr"],te["cagr"])+.25*min(tr["max_drawdown"],te["max_drawdown"])
        perf.append(row); curves[cid]=c; ledgers[cid]=t
        print("[PORTFOLIO]",row,flush=True)
    perf=pd.DataFrame(perf).sort_values(["selection_pass","selection_score"],ascending=False)
    perf.to_csv(OUT/"portfolio_train_test_selection.csv",index=False)
    frozen=list(perf[perf.selection_pass].head(5).candidate)
    if not frozen: frozen=list(perf.head(3).candidate)
    final=perf[perf.candidate.isin(frozen)].copy()
    final.to_csv(OUT/"frozen_finalists_with_2026_blind.csv",index=False)
    pd.concat({k:curves[k] for k in frozen},axis=1).to_csv(OUT/"finalist_equity_curves.csv")
    for k in frozen: ledgers[k].to_csv(OUT/f"trades_{k}.csv",index=False)
    bm,bc=core.benchmark(x)
    report={"status":"RESEARCH_PASS" if final.selection_pass.any() else "NO_ROBUST_FINALIST",
            "selection_used":"2021-2023 train + 2024-2025 test only",
            "blind_policy":"2026 excluded from all ranking",
            "screened_candidates":len(screen),"portfolio_shortlist":len(shortlist),"frozen_finalists":frozen,
            "benchmark_0050":{"full":bm,"train":period_metrics(bc,"2021-01-01","2023-12-31"),
              "test":period_metrics(bc,"2024-01-01","2025-12-31"),
              "blind2026":period_metrics(bc,"2026-01-01","2026-12-31")},
            "finalists":final.to_dict("records")}
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__":main()
