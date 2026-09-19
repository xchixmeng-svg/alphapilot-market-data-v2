#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from run_v6_2_core_numerical_full_matrix import load,safe_features,sample_idx,SEED,MAX_TRAIN,MAX_CAL
from train_v6_2_launch_distribution_head import launch_bin

ROOT=Path("v6_2_launch_full_scores"); ROOT.mkdir(exist_ok=True)
AUDIT=Path("inputs/launch/V6_2_LAUNCH_LABEL_AUDIT_ROWS.parquet")
PURGE=120
CUTS=[1,3,5,10,20,30]

def calibrate_bins(raw_cal,y_cal,raw_score,nclass=7):
    cols=[]
    for k in range(nclass):
        ir=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip")
        ir.fit(raw_cal[:,k],(y_cal==k).astype(float))
        cols.append(ir.predict(raw_score[:,k]))
    p=np.column_stack(cols); p=np.clip(p,1e-8,None); return p/p.sum(axis=1,keepdims=True)

def main():
    x,base_feats=load(); z,feats=safe_features(x,base_feats)
    a=pd.read_parquet(AUDIT)
    a["date"]=pd.to_numeric(a["date"],errors="raise").astype(np.int64)
    a["code"]=a["code"].astype(str).str.zfill(4)
    a=a[(a["success_h120"]==1)&a["launch_session"].notna()].copy()
    a["launch_bin"]=launch_bin(a["launch_session"].to_numpy())

    lab=z[["date","code"]].merge(a[["date","code","launch_session","launch_bin"]],
        on=["date","code"],how="left",validate="one_to_one")
    launch_session=lab["launch_session"].to_numpy(float)
    ybin=lab["launch_bin"].fillna(-1).to_numpy(np.int8)

    dates=np.array(sorted(z["date"].unique()),dtype=np.int64)
    cal_start=int(dates[np.searchsorted(dates,20230101)])
    hold_start=int(dates[np.searchsorted(dates,20240101)])
    pcal=int(np.searchsorted(dates,cal_start)); phold=int(np.searchsorted(dates,hold_start))
    train_end=int(dates[pcal-PURGE-1]); cal_end=int(dates[phold-PURGE-1])

    winner_valid=z["universe_ok"].to_numpy() & (ybin>=0)
    tr=np.flatnonzero(winner_valid & (z["date"].to_numpy()<=train_end))
    ca=np.flatnonzero(winner_valid & (z["date"].to_numpy()>=cal_start) & (z["date"].to_numpy()<=cal_end))
    tr=sample_idx(tr,MAX_TRAIN,SEED+41001); ca=sample_idx(ca,MAX_CAL,SEED+41002)
    score=np.flatnonzero(z["universe_ok"] & (z["date"]>=hold_start) & (z["date"]<=20241231))

    X=z[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    model=HistGradientBoostingClassifier(
      learning_rate=0.05,max_iter=140,max_leaf_nodes=31,min_samples_leaf=120,
      l2_regularization=4.0,random_state=SEED+41003
    )
    model.fit(X[tr],ybin[tr])
    raw_ca=model.predict_proba(X[ca]); raw_score=model.predict_proba(X[score])
    pbin=calibrate_bins(raw_ca,ybin[ca],raw_score)
    cum=np.cumsum(pbin[:,:6],axis=1)

    out=z.iloc[score][["date","code"]].copy()
    for j,h in enumerate(CUTS):
        out[f"p_launch_by_{h}_cond_win120"]=cum[:,j]
    out.to_parquet(ROOT/"V6_2_LAUNCH_CONDITIONAL_FULL_2024_SCORES.parquet",index=False)
    summary={
      "status":"PASS","scored_2024_rows":int(len(out)),
      "interpretation":"P(successful launch by h | eventual +10% winner within 120 sessions, X)",
      "monotonicity_violation_rows":int(np.sum(np.any(np.diff(cum,axis=1)<-1e-12,axis=1))),
      "2025_opened":False,"candidate_gate":"NONE"
    }
    (ROOT/"V6_2_LAUNCH_CONDITIONAL_FULL_SCORE_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)
if __name__=="__main__": main()
