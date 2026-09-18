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

OUT=ROOT/"v6_2_opportunity_ranking_results"
OUT.mkdir(exist_ok=True)

HORIZONS=(5,10,20,40,60,120)
TOPKS=(1,3,5)
THRESHOLDS=(0.05,0.10,0.20)
MAX_TRAIN_ROWS=480_000
SEED=926616
MODEL_PARAMS=dict(
    learning_rate=0.05,
    max_iter=160,
    max_leaf_nodes=31,
    min_samples_leaf=120,
    l2_regularization=4.0,
)

def future_path_targets(df:pd.DataFrame,h:int)->pd.DataFrame:
    out=pd.DataFrame(index=df.index)
    grp=df.groupby(["code","price_segment_id"],group_keys=False)
    idx=FixedForwardWindowIndexer(window_size=h)
    fmax=grp["close"].transform(lambda s:s.shift(-1).rolling(window=idx,min_periods=h).max())
    fmin=grp["close"].transform(lambda s:s.shift(-1).rolling(window=idx,min_periods=h).min())
    fend=grp["close"].shift(-h)
    out["mfe"]=fmax/df["close"]-1.0
    out["mae"]=fmin/df["close"]-1.0
    out["end_return"]=fend/df["close"]-1.0
    out["opportunity"]=out["mfe"]+out["mae"]
    out["time_normalized_opportunity"]=out["opportunity"]/np.sqrt(float(h))
    return out

def make_training(train,feats,seed):
    rng=np.random.default_rng(seed)
    per_h=MAX_TRAIN_ROWS//len(HORIZONS)
    xs=[]; ys=[]
    for h in HORIZONS:
        t=future_path_targets(train,h)
        valid=train["universe_ok"].to_numpy() & np.isfinite(t["time_normalized_opportunity"].to_numpy(dtype=float))
        idx=np.flatnonzero(valid)
        if len(idx)>per_h:
            idx=rng.choice(idx,size=per_h,replace=False)
        part=train.iloc[idx]
        xs.append(safe.model_frame(part,feats,np.full(len(part),h,dtype=np.float32)))
        ys.append(t["time_normalized_opportunity"].iloc[idx].to_numpy(dtype=np.float32))
    return np.concatenate(xs),np.concatenate(ys)

def fit_model(train,feats,seed):
    X,y=make_training(train,feats,seed)
    m=HistGradientBoostingRegressor(loss="squared_error",random_state=seed,**MODEL_PARAMS)
    m.fit(X,y)
    print(json.dumps({"event":"fit","rows":len(y),"features":len(feats)}),flush=True)
    return m,len(y)

def score_year(year:int):
    if year not in (2022,2023,2024):
        raise RuntimeError("V6.2 development authorizes only 2022-2024; 2025 remains sealed")
    panel,base_feats,_=safe.load_frozen_panel(require_lock=False)
    ds,feats=safe.add_safe_observable_transforms(panel,base_feats)
    train,test,cutoff=safe.year_context(ds,year)
    model,nfit=fit_model(train,feats,SEED+year)

    longs=[]
    for h in HORIZONS:
        t=future_path_targets(test,h)
        valid=np.isfinite(t["opportunity"].to_numpy(dtype=float))
        e=test.loc[valid,["date","code","price_segment_id","close"]].copy()
        X=safe.model_frame(test.loc[valid],feats,float(h))
        e["horizon"]=h
        e["pred_opportunity"]=model.predict(X)
        e["actual_mfe"]=t.loc[valid,"mfe"].to_numpy(dtype=float)
        e["actual_mae"]=t.loc[valid,"mae"].to_numpy(dtype=float)
        e["actual_end_return"]=t.loc[valid,"end_return"].to_numpy(dtype=float)
        e["actual_opportunity"]=t.loc[valid,"opportunity"].to_numpy(dtype=float)
        longs.append(e)
    long=pd.concat(longs,ignore_index=True)

    # AI chooses the most attractive horizon independently for each ticker/date.
    best=long.sort_values(["date","code","pred_opportunity"],ascending=[True,True,False]).drop_duplicates(["date","code"],keep="first")
    best["rank"]=best.groupby("date")["pred_opportunity"].rank(method="first",ascending=False).astype(int)
    best["active"]=best["pred_opportunity"]>0

    # Human-readable time-to-hit only for top5 active picks.
    top=best[(best["rank"]<=5)&best["active"]].copy()
    group_arrays={(str(c),int(s)):(g.sort_values("date")["date"].to_numpy(np.int64),g.sort_values("date")["close"].to_numpy(float))
                  for (c,s),g in ds.groupby(["code","price_segment_id"],sort=False)}
    for t in THRESHOLDS:
        tag=int(round(t*100)); vals=[]
        for r in top.itertuples(index=False):
            dates,closes=group_arrays[(str(r.code),int(r.price_segment_id))]
            pos=int(np.searchsorted(dates,int(r.date)))
            fut=closes[pos+1:min(pos+1+int(r.horizon),len(closes))]
            rr=fut/float(r.close)-1.0
            hit=np.flatnonzero(rr>=t)
            vals.append(int(hit[0]+1) if len(hit) else np.nan)
        top[f"sessions_to_{tag}pct"]=vals

    rows=[]
    for k in TOPKS:
        x=top[top["rank"]<=k].copy()
        row={"year":year,"top_k":k,"days_total":int(best["date"].nunique()),
             "active_days":int(x["date"].nunique()),"picks":int(len(x)),
             "positive_end_rate":float((x["actual_end_return"]>0).mean()) if len(x) else None,
             "mean_end_return":float(x["actual_end_return"].mean()) if len(x) else None,
             "median_end_return":float(x["actual_end_return"].median()) if len(x) else None,
             "mean_mfe":float(x["actual_mfe"].mean()) if len(x) else None,
             "median_mfe":float(x["actual_mfe"].median()) if len(x) else None,
             "mean_mae":float(x["actual_mae"].mean()) if len(x) else None,
             "median_mae":float(x["actual_mae"].median()) if len(x) else None,
             "mean_actual_opportunity":float(x["actual_opportunity"].mean()) if len(x) else None,
             "median_actual_opportunity":float(x["actual_opportunity"].median()) if len(x) else None,
             "median_predicted_best_horizon":float(x["horizon"].median()) if len(x) else None,
             "horizon_5_share":float((x["horizon"]==5).mean()) if len(x) else None,
             "horizon_10_share":float((x["horizon"]==10).mean()) if len(x) else None,
             "horizon_20_share":float((x["horizon"]==20).mean()) if len(x) else None,
             "horizon_40_share":float((x["horizon"]==40).mean()) if len(x) else None,
             "horizon_60_share":float((x["horizon"]==60).mean()) if len(x) else None,
             "horizon_120_share":float((x["horizon"]==120).mean()) if len(x) else None}
        for t in THRESHOLDS:
            tag=int(round(t*100))
            row[f"hit_{tag}pct_rate"]=float((x["actual_mfe"]>=t).mean()) if len(x) else None
            hit=x.loc[x["actual_mfe"]>=t,f"sessions_to_{tag}pct"].dropna()
            row[f"median_sessions_to_{tag}pct_when_hit"]=float(hit.median()) if len(hit) else None
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT/f"V6_2_{year}_SUMMARY.csv",index=False,encoding="utf-8-sig")
    best.to_csv(OUT/f"V6_2_{year}_ALL_BEST_HORIZON.csv",index=False,encoding="utf-8-sig")
    top.to_csv(OUT/f"V6_2_{year}_TOP5_DETAIL.csv",index=False,encoding="utf-8-sig")
    meta={"year":year,"status":"DEVELOPMENT_ONLY","train_cutoff":cutoff,"fit_rows":nfit,
          "target":"(future_mfe + future_mae) / sqrt(horizon) for cross-horizon comparability",
          "horizons":list(HORIZONS),"reset_safe":True,"2025_opened":False}
    (OUT/f"V6_2_{year}_META.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.2 SUMMARY]",json.dumps(rows,ensure_ascii=False),flush=True)

def aggregate(path:Path):
    frames=[pd.read_csv(path/f"V6_2_{y}_SUMMARY.csv") for y in (2022,2023,2024)]
    x=pd.concat(frames,ignore_index=True)
    x.to_csv(OUT/"V6_2_DEVELOPMENT_SUMMARY.csv",index=False,encoding="utf-8-sig")
    agg=[]
    for k in TOPKS:
        q=x[x["top_k"]==k].copy()
        w=q["picks"].to_numpy(dtype=float)
        wm=lambda c:float(np.average(q[c].to_numpy(dtype=float),weights=w))
        row={"top_k":k,"years":[2022,2023,2024],"picks":int(w.sum()),
             "positive_end_rate":wm("positive_end_rate"),"mean_end_return":wm("mean_end_return"),
             "median_end_return_weighted":wm("median_end_return"),"mean_mfe":wm("mean_mfe"),
             "median_mfe_weighted":wm("median_mfe"),"mean_mae":wm("mean_mae"),
             "median_mae_weighted":wm("median_mae"),"mean_actual_opportunity":wm("mean_actual_opportunity")}
        for t in THRESHOLDS:
            tag=int(round(t*100)); row[f"hit_{tag}pct_rate"]=wm(f"hit_{tag}pct_rate")
        agg.append(row)
    (OUT/"V6_2_DEVELOPMENT_AGG.json").write_text(json.dumps(agg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[V6.2 AGG]",json.dumps(agg,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int); ap.add_argument("--aggregate-dir")
    ns=ap.parse_args()
    if bool(ns.year)==bool(ns.aggregate_dir): raise SystemExit("specify exactly one")
    score_year(ns.year) if ns.year else aggregate(Path(ns.aggregate_dir))
