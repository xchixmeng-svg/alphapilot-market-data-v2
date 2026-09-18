#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import backtest_ai_market_reasoning_v6_1_corp_safe as v61

OUT=ROOT/"v6_1_actionability"
OUT.mkdir(exist_ok=True)
TOPKS=(1,3,5)
THRESHOLDS=(0.05,0.10,0.20)

def path_metrics(ds,picks,h):
    rows=[]
    groups={(str(c),int(s)):g.sort_values("date") for (c,s),g in ds.groupby(["code","price_segment_id"],sort=False)}
    for r in picks.itertuples(index=False):
        g=groups.get((str(r.code),int(r.price_segment_id)))
        if g is None: continue
        dates=g["date"].to_numpy(dtype=np.int64); closes=g["close"].to_numpy(dtype=float)
        pos=int(np.searchsorted(dates,int(r.date)))
        if pos>=len(dates) or dates[pos]!=int(r.date): continue
        fut=closes[pos+1:min(pos+1+h,len(closes))]
        if len(fut)==0: continue
        rets=fut/float(r.close)-1
        o={"date":int(r.date),"code":str(r.code).zfill(4),"price_segment_id":int(r.price_segment_id),
           "rank_q50":int(r.rank_q50),"horizon":h,"entry_close":float(r.close),
           "pred_q10":float(r.pred_q10),"pred_q25":float(r.pred_q25),"pred_q50":float(r.pred_q50),
           "pred_q75":float(r.pred_q75),"pred_q90":float(r.pred_q90),
           "actual_end_return":float(r.actual_end_return),
           "max_return_within_h":float(np.nanmax(rets)),"min_return_within_h":float(np.nanmin(rets)),
           "q50_abs_error":float(abs(float(r.actual_end_return)-float(r.pred_q50)))}
        for t in THRESHOLDS:
            tag=int(round(t*100)); idx=np.flatnonzero(rets>=t)
            o[f"hit_{tag}pct"]=bool(len(idx)); o[f"sessions_to_{tag}pct"]=int(idx[0]+1) if len(idx) else None
        rows.append(o)
    return pd.DataFrame(rows)

def run(year):
    panel,base_feats,_=v61.load_frozen_panel(require_lock=True)
    ds,feats=v61.add_safe_observable_transforms(panel,base_feats)
    train,test,_=v61.year_context(ds,year)
    models,_=v61.fit_models(train,feats,v61.RNG_SEED+year)
    chunks=[]
    for h in v61.AUDIT_HORIZONS:
        actual=v61.fwd_return(test,h); valid=np.isfinite(actual.to_numpy(dtype=float))
        e=test.loc[valid,["date","code","price_segment_id","close"]].copy()
        p=v61.predict_quantiles(models,test.loc[valid],feats,h)
        for j,q in enumerate((10,25,50,75,90)): e[f"pred_q{q}"]=p[:,j]
        e["actual_end_return"]=actual.to_numpy(dtype=float)[valid]
        e["rank_q50"]=e.groupby("date")["pred_q50"].rank(method="first",ascending=False).astype(int)
        chunks.append(path_metrics(ds,e[e["rank_q50"]<=5],h))
    d=pd.concat(chunks,ignore_index=True)
    d.to_csv(OUT/f"V6_1_ACTIONABILITY_{year}_TOP5_DETAIL.csv",index=False,encoding="utf-8-sig")
    rows=[]
    for h in v61.AUDIT_HORIZONS:
        for k in TOPKS:
            x=d[(d["horizon"]==h)&(d["rank_q50"]<=k)]
            row={"year":year,"horizon":h,"top_k":k,"dates":int(x["date"].nunique()),"picks":int(len(x)),
                 "positive_end_rate":float((x["actual_end_return"]>0).mean()),
                 "mean_end_return":float(x["actual_end_return"].mean()),"median_end_return":float(x["actual_end_return"].median()),
                 "mean_max_return":float(x["max_return_within_h"].mean()),"median_max_return":float(x["max_return_within_h"].median()),
                 "mean_min_return":float(x["min_return_within_h"].mean()),"median_min_return":float(x["min_return_within_h"].median()),
                 "median_q50_abs_error":float(x["q50_abs_error"].median())}
            for t in THRESHOLDS:
                tag=int(round(t*100)); row[f"hit_{tag}pct_rate"]=float(x[f"hit_{tag}pct"].mean())
                hh=x.loc[x[f"hit_{tag}pct"],f"sessions_to_{tag}pct"].dropna()
                row[f"median_sessions_to_{tag}pct_when_hit"]=float(hh.median()) if len(hh) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/f"V6_1_ACTIONABILITY_{year}_SUMMARY.csv",index=False,encoding="utf-8-sig")
    (OUT/f"V6_1_ACTIONABILITY_{year}_META.json").write_text(json.dumps({"year":year,"status":"DIAGNOSTIC_ONLY","2025_opened":False},indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,required=True); ns=ap.parse_args()
    if ns.year not in (2022,2023,2024): raise SystemExit("only repair-validation years")
    run(ns.year)
