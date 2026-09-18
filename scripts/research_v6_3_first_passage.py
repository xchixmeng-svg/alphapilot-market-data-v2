#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import backtest_ai_market_reasoning_v6_1_corp_safe as safe

OUT=ROOT/"v6_3_first_passage_results"
OUT.mkdir(exist_ok=True)

HORIZONS=(5,10,20,40,60,120)
TOPKS=(1,3,5)
UP_BARRIER=0.10
DOWN_BARRIER=-0.05
SEED=926616
MAX_TRAIN_ROWS=480_000
MODEL_PARAMS=dict(
    learning_rate=0.05,
    max_iter=160,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=4.0,
)

def first_passage_targets(df:pd.DataFrame,h:int)->pd.DataFrame:
    n=len(df)
    success=np.full(n,np.nan,dtype=np.float32)
    up_time=np.full(n,np.nan,dtype=np.float32)
    down_time=np.full(n,np.nan,dtype=np.float32)
    mfe=np.full(n,np.nan,dtype=np.float32)
    mae=np.full(n,np.nan,dtype=np.float32)
    end_ret=np.full(n,np.nan,dtype=np.float32)

    tmp=df.reset_index(drop=True)
    for _,g in tmp.groupby(["code","price_segment_id"],sort=False):
        pos=g.index.to_numpy(dtype=np.int64)
        c=g["close"].to_numpy(dtype=float)
        if len(c)<=h:
            continue
        w=sliding_window_view(c,h+1)
        base=w[:,0:1]
        r=w[:,1:]/base-1.0
        up=r>=UP_BARRIER
        dn=r<=DOWN_BARRIER
        up_any=up.any(axis=1)
        dn_any=dn.any(axis=1)
        up_first=np.where(up_any,up.argmax(axis=1)+1,h+1)
        dn_first=np.where(dn_any,dn.argmax(axis=1)+1,h+1)
        ok=up_any & (up_first<dn_first)
        m=len(w)
        tgt=pos[:m]
        success[tgt]=ok.astype(np.float32)
        up_time[tgt]=np.where(up_any,up_first,np.nan)
        down_time[tgt]=np.where(dn_any,dn_first,np.nan)
        mfe[tgt]=np.max(r,axis=1).astype(np.float32)
        mae[tgt]=np.min(r,axis=1).astype(np.float32)
        end_ret[tgt]=r[:,-1].astype(np.float32)

    return pd.DataFrame({
        "success":success,"up_time":up_time,"down_time":down_time,
        "mfe":mfe,"mae":mae,"end_return":end_ret
    },index=df.index)

def make_training(train,feats,seed):
    rng=np.random.default_rng(seed)
    per_h=MAX_TRAIN_ROWS//len(HORIZONS)
    xs=[]; ys=[]; base_rates={}
    for h in HORIZONS:
        t=first_passage_targets(train,h)
        y=t["success"].to_numpy(dtype=float)
        valid=train["universe_ok"].to_numpy() & np.isfinite(y)
        idx=np.flatnonzero(valid)
        base_rates[h]=float(y[idx].mean()) if len(idx) else np.nan
        if len(idx)>per_h:
            idx=rng.choice(idx,size=per_h,replace=False)
        part=train.iloc[idx]
        xs.append(safe.model_frame(part,feats,np.full(len(part),h,dtype=np.float32)))
        ys.append(y[idx].astype(np.int8))
    X=np.concatenate(xs); y=np.concatenate(ys)
    return X,y,base_rates

def fit_model(train,feats,seed):
    X,y,base_rates=make_training(train,feats,seed)
    m=HistGradientBoostingClassifier(loss="log_loss",random_state=seed,**MODEL_PARAMS)
    m.fit(X,y)
    print(json.dumps({"event":"fit","rows":len(y),"positive_rate":float(y.mean()),"base_rates":base_rates}),flush=True)
    return m,len(y),base_rates

def score_year(year:int):
    if year not in (2022,2023,2024):
        raise RuntimeError("V6.3 development authorizes only 2022-2024; 2025 remains sealed")
    panel,base_feats,_=safe.load_frozen_panel(require_lock=False)
    ds,feats=safe.add_safe_observable_transforms(panel,base_feats)
    train,test,cutoff=safe.year_context(ds,year)
    model,nfit,base_rates=fit_model(train,feats,SEED+year)

    longs=[]
    for h in HORIZONS:
        t=first_passage_targets(test,h)
        y=t["success"].to_numpy(dtype=float)
        valid=np.isfinite(y)
        e=test.loc[valid,["date","code","price_segment_id","close"]].copy()
        p=model.predict_proba(safe.model_frame(test.loc[valid],feats,float(h)))[:,1]
        e["horizon"]=h
        e["p_success"]=p
        e["base_rate"]=base_rates[h]
        e["uplift"]=e["p_success"]-e["base_rate"]
        e["actual_success"]=y[valid]
        e["up_time"]=t.loc[valid,"up_time"].to_numpy(dtype=float)
        e["actual_mfe"]=t.loc[valid,"mfe"].to_numpy(dtype=float)
        e["actual_mae"]=t.loc[valid,"mae"].to_numpy(dtype=float)
        e["actual_end_return"]=t.loc[valid,"end_return"].to_numpy(dtype=float)
        longs.append(e)
    long=pd.concat(longs,ignore_index=True)

    best=(long.sort_values(["date","code","uplift"],ascending=[True,True,False])
          .drop_duplicates(["date","code"],keep="first"))
    best["rank"]=best.groupby("date")["uplift"].rank(method="first",ascending=False).astype(int)
    best["active"]=best["uplift"]>0
    top=best[(best["rank"]<=5)&best["active"]].copy()

    rows=[]
    for k in TOPKS:
        x=top[top["rank"]<=k].copy()
        row={"year":year,"top_k":k,"days_total":int(best["date"].nunique()),"active_days":int(x["date"].nunique()),
             "picks":int(len(x)),"clean_success_rate":float(x["actual_success"].mean()) if len(x) else None,
             "mean_predicted_success":float(x["p_success"].mean()) if len(x) else None,
             "mean_matched_base_rate":float(x["base_rate"].mean()) if len(x) else None,
             "realized_uplift_vs_base":float(x["actual_success"].mean()-x["base_rate"].mean()) if len(x) else None,
             "brier":float(brier_score_loss(x["actual_success"],x["p_success"])) if len(x) else None,
             "positive_end_rate":float((x["actual_end_return"]>0).mean()) if len(x) else None,
             "mean_end_return":float(x["actual_end_return"].mean()) if len(x) else None,
             "median_end_return":float(x["actual_end_return"].median()) if len(x) else None,
             "mean_mfe":float(x["actual_mfe"].mean()) if len(x) else None,
             "median_mfe":float(x["actual_mfe"].median()) if len(x) else None,
             "mean_mae":float(x["actual_mae"].mean()) if len(x) else None,
             "median_mae":float(x["actual_mae"].median()) if len(x) else None,
             "hit_20pct_rate":float((x["actual_mfe"]>=0.20).mean()) if len(x) else None,
             "median_sessions_to_10pct_when_clean_success":float(x.loc[x["actual_success"]==1,"up_time"].median()) if (x["actual_success"]==1).any() else None,
             "median_predicted_best_horizon":float(x["horizon"].median()) if len(x) else None}
        for h in HORIZONS:
            row[f"horizon_{h}_share"]=float((x["horizon"]==h).mean()) if len(x) else None
        for pcut in (0.6,0.7,0.8):
            z=x[x["p_success"]>=pcut]
            row[f"p{int(pcut*100)}_picks"]=int(len(z))
            row[f"p{int(pcut*100)}_actual_success_rate"]=float(z["actual_success"].mean()) if len(z) else None
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT/f"V6_3_{year}_SUMMARY.csv",index=False,encoding="utf-8-sig")
    best.to_csv(OUT/f"V6_3_{year}_ALL_BEST_HORIZON.csv",index=False,encoding="utf-8-sig")
    top.to_csv(OUT/f"V6_3_{year}_TOP5_DETAIL.csv",index=False,encoding="utf-8-sig")
    meta={"year":year,"status":"DEVELOPMENT_ONLY","target":"close reaches +10% before close reaches -5% within horizon",
          "ranking":"model probability uplift versus horizon-matched unconditional base rate",
          "train_cutoff":cutoff,"fit_rows":nfit,"base_rates":base_rates,
          "reset_safe":True,"2025_opened":False}
    (OUT/f"V6_3_{year}_META.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.3 SUMMARY]",json.dumps(rows,ensure_ascii=False),flush=True)

def aggregate(path:Path):
    x=pd.concat([pd.read_csv(path/f"V6_3_{y}_SUMMARY.csv") for y in (2022,2023,2024)],ignore_index=True)
    x.to_csv(OUT/"V6_3_DEVELOPMENT_SUMMARY.csv",index=False,encoding="utf-8-sig")
    agg=[]
    for k in TOPKS:
        q=x[x["top_k"]==k].copy(); w=q["picks"].to_numpy(dtype=float)
        wm=lambda c:float(np.average(q[c].to_numpy(dtype=float),weights=w))
        row={"top_k":k,"years":[2022,2023,2024],"picks":int(w.sum()),
             "clean_success_rate":wm("clean_success_rate"),"mean_predicted_success":wm("mean_predicted_success"),
             "mean_matched_base_rate":wm("mean_matched_base_rate"),"realized_uplift_vs_base":wm("realized_uplift_vs_base"),
             "positive_end_rate":wm("positive_end_rate"),"mean_end_return":wm("mean_end_return"),
             "median_end_return_weighted":wm("median_end_return"),"mean_mfe":wm("mean_mfe"),
             "median_mfe_weighted":wm("median_mfe"),"mean_mae":wm("mean_mae"),
             "median_mae_weighted":wm("median_mae"),"hit_20pct_rate":wm("hit_20pct_rate")}
        agg.append(row)
    (OUT/"V6_3_DEVELOPMENT_AGG.json").write_text(json.dumps(agg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.3 AGG]",json.dumps(agg,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int); ap.add_argument("--aggregate-dir")
    ns=ap.parse_args()
    if bool(ns.year)==bool(ns.aggregate_dir): raise SystemExit("specify exactly one")
    score_year(ns.year) if ns.year else aggregate(Path(ns.aggregate_dir))
