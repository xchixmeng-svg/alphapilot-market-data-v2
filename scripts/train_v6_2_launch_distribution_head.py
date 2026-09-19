#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from run_v6_2_core_numerical_full_matrix import load, safe_features, sample_idx, SEED, MAX_TRAIN, MAX_CAL

ROOT=Path("v6_2_launch_distribution_head"); ROOT.mkdir(exist_ok=True)
AUDIT=Path("inputs/launch/V6_2_LAUNCH_LABEL_AUDIT_ROWS.parquet")
PURGE=120
CUTS=[1,3,5,10,20,30]
BIN_NAMES=["D1","D2_3","D4_5","D6_10","D11_20","D21_30","GT30"]

def launch_bin(s):
    s=np.asarray(s,float)
    y=np.full(len(s),6,dtype=np.int8)
    y[s<=30]=5
    y[s<=20]=4
    y[s<=10]=3
    y[s<=5]=2
    y[s<=3]=1
    y[s<=1]=0
    return y

def brier_binary(y,p):
    return float(np.mean((p-y)**2))

def ece_binary(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); e=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any():
            e += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(e)

def calibrate_bins(raw_cal,y_cal,raw_ev,nclass):
    cols=[]
    for k in range(nclass):
        ir=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip")
        ir.fit(raw_cal[:,k],(y_cal==k).astype(float))
        cols.append(ir.predict(raw_ev[:,k]))
    p=np.column_stack(cols)
    p=np.clip(p,1e-8,None)
    p=p/p.sum(axis=1,keepdims=True)
    return p

def main():
    x,base_feats=load()
    z,feats=safe_features(x,base_feats)
    a=pd.read_parquet(AUDIT)
    a["date"]=pd.to_numeric(a["date"],errors="raise").astype(np.int64)
    a["code"]=a["code"].astype(str).str.zfill(4)
    a=a[(a["success_h120"]==1)&a["launch_session"].notna()].copy()
    a["launch_bin"]=launch_bin(a["launch_session"].to_numpy())
    z=z.merge(a[["date","code","launch_session","prelaunch_mae","launch_bin"]],
              on=["date","code"],how="inner",validate="one_to_one")
    if len(z)<100000: raise RuntimeError("launch merge too small")

    dates=np.array(sorted(z["date"].unique()),dtype=np.int64)
    cal_start=int(dates[np.searchsorted(dates,20230101)])
    hold_start=int(dates[np.searchsorted(dates,20240101)])
    pcal=int(np.searchsorted(dates,cal_start)); phold=int(np.searchsorted(dates,hold_start))
    train_end=int(dates[pcal-PURGE-1]); cal_end=int(dates[phold-PURGE-1])

    valid=z["universe_ok"] & z["launch_bin"].between(0,6)
    tr=np.flatnonzero(valid & (z["date"]<=train_end))
    ca=np.flatnonzero(valid & (z["date"]>=cal_start) & (z["date"]<=cal_end))
    ev=np.flatnonzero(valid & (z["date"]>=hold_start) & (z["date"]<=20241231))
    tr=sample_idx(tr,MAX_TRAIN,SEED+41001); ca=sample_idx(ca,MAX_CAL,SEED+41002)
    if min(len(tr),len(ca),len(ev))<5000: raise RuntimeError("split too small")

    X=z[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    y=z["launch_bin"].to_numpy(np.int8)
    model=HistGradientBoostingClassifier(
        learning_rate=0.05,max_iter=140,max_leaf_nodes=31,min_samples_leaf=120,
        l2_regularization=4.0,random_state=SEED+41003
    )
    model.fit(X[tr],y[tr])
    if not np.array_equal(model.classes_,np.arange(7)):
        raise RuntimeError(f"unexpected classes {model.classes_}")

    raw_ca=model.predict_proba(X[ca]); raw_ev=model.predict_proba(X[ev])
    pbin=calibrate_bins(raw_ca,y[ca],raw_ev,7)
    yev=y[ev]
    train_freq=np.bincount(y[tr],minlength=7)/len(tr)
    base_bin=np.tile(train_freq,(len(ev),1))

    pred=pd.DataFrame({
      "date":z.iloc[ev]["date"].to_numpy(),
      "code":z.iloc[ev]["code"].to_numpy(),
      "realized_launch_session":z.iloc[ev]["launch_session"].to_numpy(),
      "prelaunch_mae":z.iloc[ev]["prelaunch_mae"].to_numpy()
    })
    for k,n in enumerate(BIN_NAMES):
        pred[f"p_launch_bin_{n}"]=pbin[:,k]

    # cumulative probabilities are guaranteed monotone by construction.
    cum=np.cumsum(pbin[:,:6],axis=1)
    base_cum=np.cumsum(base_bin[:,:6],axis=1)
    cumulative_metrics=[]
    for j,h in enumerate(CUTS):
        actual=(z.iloc[ev]["launch_session"].to_numpy()<=h).astype(int)
        p=cum[:,j]; pb=base_cum[:,j]
        cumulative_metrics.append({
          "horizon":h,
          "holdout_rate":float(actual.mean()),
          "brier_model":brier_binary(actual,p),
          "brier_base":brier_binary(actual,pb),
          "ece":ece_binary(actual,p)
        })
        pred[f"p_launch_by_{h}_cond_win120"]=p

    mono=int(np.sum(np.any(np.diff(cum,axis=1)<-1e-12,axis=1)))
    out={
      "status":"PROTOTYPE_COMPLETE",
      "scientific_lock":False,
      "architecture":"SINGLE_7_BIN_LAUNCH_TIME_DISTRIBUTION_HEAD",
      "conditional_on":"eventual +10% winner within 120 sessions",
      "not_candidate_gate":True,
      "split":{"train_max":train_end,"calibration_min":cal_start,"calibration_max":cal_end,
               "holdout_min":hold_start,"holdout_max":20241231,"purge_sessions":PURGE},
      "rows":{"train":int(len(tr)),"calibration":int(len(ca)),"holdout":int(len(ev))},
      "feature_count":len(feats),
      "multiclass_logloss_model":float(log_loss(yev,pbin,labels=list(range(7)))),
      "multiclass_logloss_base":float(log_loss(yev,base_bin,labels=list(range(7)))),
      "monotonicity_violation_rows":mono,
      "cumulative_metrics":cumulative_metrics,
      "2025_opened":False
    }
    pred.to_parquet(ROOT/"V6_2_LAUNCH_DISTRIBUTION_HOLDOUT.parquet",index=False)
    pd.DataFrame(cumulative_metrics).to_csv(ROOT/"V6_2_LAUNCH_DISTRIBUTION_METRICS.csv",index=False)
    (ROOT/"V6_2_LAUNCH_DISTRIBUTION_SUMMARY.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
