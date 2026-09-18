#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import backtest_ai_market_reasoning_v6_1_corp_safe as safe
import research_v6_3_first_passage as fp

OUT=ROOT/"v6_5_cross_sectional_pareto_results"
OUT.mkdir(exist_ok=True)

HORIZONS=(5,10,20,40,60,120)
TOPKS=(1,3,5)
SEED=926616
MAX_TRAIN_ROWS=480_000
MODEL_PARAMS=dict(
    learning_rate=0.05,
    max_iter=160,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=4.0,
)

def path_targets(df:pd.DataFrame,h:int)->pd.DataFrame:
    out=pd.DataFrame(index=df.index)
    grp=df.groupby(["code","price_segment_id"],group_keys=False)
    idx=FixedForwardWindowIndexer(window_size=h)
    fmax=grp["close"].transform(lambda s:s.shift(-1).rolling(window=idx,min_periods=h).max())
    fmin=grp["close"].transform(lambda s:s.shift(-1).rolling(window=idx,min_periods=h).min())
    fend=grp["close"].shift(-h)
    out["mfe"]=fmax/df["close"]-1.0
    out["mae"]=fmin/df["close"]-1.0
    out["end_return"]=fend/df["close"]-1.0
    # Cross-sectional future ranks are the learning targets:
    # high upside rank and high safety rank (MAE closer to zero is better).
    out["upside_rank"]=out["mfe"].groupby(df["date"]).rank(method="average",pct=True)
    out["safety_rank"]=out["mae"].groupby(df["date"]).rank(method="average",pct=True)
    return out

def make_training(train,feats,seed):
    rng=np.random.default_rng(seed)
    per_h=MAX_TRAIN_ROWS//len(HORIZONS)
    xs=[]; yu=[]; ys=[]
    for h in HORIZONS:
        t=path_targets(train,h)
        valid=(train["universe_ok"].to_numpy()
               & np.isfinite(t["upside_rank"].to_numpy(dtype=float))
               & np.isfinite(t["safety_rank"].to_numpy(dtype=float)))
        ix=np.flatnonzero(valid)
        if len(ix)>per_h:
            ix=rng.choice(ix,size=per_h,replace=False)
        part=train.iloc[ix]
        xs.append(safe.model_frame(part,feats,np.full(len(part),h,dtype=np.float32)))
        yu.append(t["upside_rank"].iloc[ix].to_numpy(dtype=np.float32))
        ys.append(t["safety_rank"].iloc[ix].to_numpy(dtype=np.float32))
    return np.concatenate(xs),np.concatenate(yu),np.concatenate(ys)

def fit_heads(train,feats,seed):
    X,yu,ys=make_training(train,feats,seed)
    up=HistGradientBoostingRegressor(loss="squared_error",random_state=seed,**MODEL_PARAMS)
    safety=HistGradientBoostingRegressor(loss="squared_error",random_state=seed+1,**MODEL_PARAMS)
    up.fit(X,yu); safety.fit(X,ys)
    print(json.dumps({
        "event":"fit","rows":len(yu),"features":len(feats),
        "up_rank_mean":float(np.mean(yu)),"safety_rank_mean":float(np.mean(ys))
    }),flush=True)
    return up,safety,len(yu)

def score_year(year:int):
    if year not in (2022,2023,2024):
        raise RuntimeError("V6.5 development authorizes only 2022-2024; 2025 remains sealed")

    panel,base_feats,_=safe.load_frozen_panel(require_lock=False)
    ds,feats=safe.add_safe_observable_transforms(panel,base_feats)
    train,test,cutoff=safe.year_context(ds,year)
    up_model,safety_model,nfit=fit_heads(train,feats,SEED+year)

    longs=[]
    for h in HORIZONS:
        t=path_targets(test,h)
        clean=fp.first_passage_targets(test,h)
        valid=(np.isfinite(t["mfe"].to_numpy(dtype=float))
               & np.isfinite(clean["success"].to_numpy(dtype=float)))
        e=test.loc[valid,["date","code","price_segment_id","close"]].copy()
        X=safe.model_frame(test.loc[valid],feats,float(h))
        e["horizon"]=h
        e["pred_upside_rank"]=np.clip(up_model.predict(X),0.0,1.0)
        e["pred_safety_rank"]=np.clip(safety_model.predict(X),0.0,1.0)
        # Maximin Pareto score: no objective weight to tune.
        e["balanced_score"]=np.minimum(e["pred_upside_rank"],e["pred_safety_rank"])
        e["mean_objective_score"]=(e["pred_upside_rank"]+e["pred_safety_rank"])/2.0
        e["actual_mfe"]=t.loc[valid,"mfe"].to_numpy(dtype=float)
        e["actual_mae"]=t.loc[valid,"mae"].to_numpy(dtype=float)
        e["actual_end_return"]=t.loc[valid,"end_return"].to_numpy(dtype=float)
        e["actual_upside_rank"]=t.loc[valid,"upside_rank"].to_numpy(dtype=float)
        e["actual_safety_rank"]=t.loc[valid,"safety_rank"].to_numpy(dtype=float)
        e["clean_success"]=clean.loc[valid,"success"].to_numpy(dtype=float)
        e["sessions_to_10pct"]=clean.loc[valid,"up_time"].to_numpy(dtype=float)
        longs.append(e)

    long=pd.concat(longs,ignore_index=True)
    best=(long.sort_values(
        ["date","code","balanced_score","mean_objective_score"],
        ascending=[True,True,False,False]
    ).drop_duplicates(["date","code"],keep="first"))
    best["rank"]=best.groupby("date")["balanced_score"].rank(method="first",ascending=False).astype(int)
    top=best[best["rank"]<=5].copy()

    rows=[]
    for k in TOPKS:
        x=top[top["rank"]<=k].copy()
        row={
            "year":year,"top_k":k,"days":int(x["date"].nunique()),"picks":int(len(x)),
            "clean_success_rate":float(x["clean_success"].mean()),
            "positive_end_rate":float((x["actual_end_return"]>0).mean()),
            "mean_end_return":float(x["actual_end_return"].mean()),
            "median_end_return":float(x["actual_end_return"].median()),
            "mean_mfe":float(x["actual_mfe"].mean()),
            "median_mfe":float(x["actual_mfe"].median()),
            "mean_mae":float(x["actual_mae"].mean()),
            "median_mae":float(x["actual_mae"].median()),
            "hit_10pct_rate":float((x["actual_mfe"]>=0.10).mean()),
            "hit_20pct_rate":float((x["actual_mfe"]>=0.20).mean()),
            "median_sessions_to_10pct_when_clean_success":float(
                x.loc[x["clean_success"]==1,"sessions_to_10pct"].median()
            ) if (x["clean_success"]==1).any() else None,
            "mean_actual_upside_rank":float(x["actual_upside_rank"].mean()),
            "mean_actual_safety_rank":float(x["actual_safety_rank"].mean()),
            "median_balanced_score":float(x["balanced_score"].median()),
            "median_horizon":float(x["horizon"].median()),
        }
        for h in HORIZONS:
            row[f"horizon_{h}_share"]=float((x["horizon"]==h).mean())
        rows.append(row)

    bands=[]
    for cut in (0.50,0.60,0.70,0.80,0.90):
        z=best[best["balanced_score"]>=cut]
        bands.append({
            "year":year,"score_cut":cut,"picks":int(len(z)),
            "clean_success_rate":float(z["clean_success"].mean()) if len(z) else None,
            "positive_end_rate":float((z["actual_end_return"]>0).mean()) if len(z) else None,
            "median_mae":float(z["actual_mae"].median()) if len(z) else None,
            "median_mfe":float(z["actual_mfe"].median()) if len(z) else None,
            "mean_actual_upside_rank":float(z["actual_upside_rank"].mean()) if len(z) else None,
            "mean_actual_safety_rank":float(z["actual_safety_rank"].mean()) if len(z) else None,
        })

    pd.DataFrame(rows).to_csv(OUT/f"V6_5_{year}_SUMMARY.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(bands).to_csv(OUT/f"V6_5_{year}_SCORE_BANDS.csv",index=False,encoding="utf-8-sig")
    best.to_csv(OUT/f"V6_5_{year}_ALL_BEST_HORIZON.csv",index=False,encoding="utf-8-sig")
    top.to_csv(OUT/f"V6_5_{year}_TOP5_DETAIL.csv",index=False,encoding="utf-8-sig")
    meta={
        "year":year,"status":"DEVELOPMENT_ONLY",
        "architecture":"dual-head cross-sectional future-rank learning",
        "targets":["same-date future MFE percentile","same-date future MAE safety percentile"],
        "selection_score":"min(predicted_upside_rank, predicted_safety_rank)",
        "market_regime_rule":None,
        "no_tunable_objective_weights":True,
        "train_cutoff":cutoff,"fit_rows":nfit,"reset_safe":True,"2025_opened":False
    }
    (OUT/f"V6_5_{year}_META.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.5 SUMMARY]",json.dumps(rows,ensure_ascii=False),flush=True)
    print("[V6.5 BANDS]",json.dumps(bands,ensure_ascii=False),flush=True)

def aggregate(path:Path):
    x=pd.concat([pd.read_csv(path/f"V6_5_{y}_SUMMARY.csv") for y in (2022,2023,2024)],ignore_index=True)
    x.to_csv(OUT/"V6_5_DEVELOPMENT_SUMMARY.csv",index=False,encoding="utf-8-sig")
    agg=[]
    for k in TOPKS:
        q=x[x["top_k"]==k].copy(); w=q["picks"].to_numpy(dtype=float)
        wm=lambda c:float(np.average(q[c].to_numpy(dtype=float),weights=w))
        agg.append({
            "top_k":k,"years":[2022,2023,2024],"picks":int(w.sum()),
            "clean_success_rate":wm("clean_success_rate"),
            "positive_end_rate":wm("positive_end_rate"),
            "mean_end_return":wm("mean_end_return"),
            "median_end_return_weighted":wm("median_end_return"),
            "mean_mfe":wm("mean_mfe"),"median_mfe_weighted":wm("median_mfe"),
            "mean_mae":wm("mean_mae"),"median_mae_weighted":wm("median_mae"),
            "hit_10pct_rate":wm("hit_10pct_rate"),"hit_20pct_rate":wm("hit_20pct_rate"),
            "mean_actual_upside_rank":wm("mean_actual_upside_rank"),
            "mean_actual_safety_rank":wm("mean_actual_safety_rank"),
        })
    (OUT/"V6_5_DEVELOPMENT_AGG.json").write_text(json.dumps(agg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.5 AGG]",json.dumps(agg,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int)
    ap.add_argument("--aggregate-dir")
    ns=ap.parse_args()
    if bool(ns.year)==bool(ns.aggregate_dir):
        raise SystemExit("specify exactly one")
    score_year(ns.year) if ns.year else aggregate(Path(ns.aggregate_dir))
