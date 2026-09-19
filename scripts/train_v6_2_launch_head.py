#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

from run_v6_2_core_numerical_full_matrix import load, safe_features, sample_idx, SEED, MAX_TRAIN, MAX_CAL

ROOT=Path("v6_2_launch_head"); ROOT.mkdir(exist_ok=True)
AUDIT=Path("inputs/launch/V6_2_LAUNCH_LABEL_AUDIT_ROWS.parquet")
HORIZONS=[1,3,5,10,20,30]
PURGE=120

def ece_binary(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); e=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any():
            e += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(e)

def calibrate(raw_cal,y_cal,raw_ev):
    ir=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip")
    ir.fit(raw_cal,y_cal.astype(float))
    return np.clip(ir.predict(raw_ev),1e-6,1-1e-6)

def main():
    x,base_feats=load()
    z,feats=safe_features(x,base_feats)
    a=pd.read_parquet(AUDIT)
    a["date"]=pd.to_numeric(a["date"],errors="raise").astype(np.int64)
    a["code"]=a["code"].astype(str).str.zfill(4)

    # Conditional launch timing is trained only on eventual +10% winners with valid launch labels.
    a=a[(a["success_h120"]==1) & a["launch_session"].notna()].copy()
    cols=["date","code","launch_session","prelaunch_mae"]
    z=z.merge(a[cols],on=["date","code"],how="inner",validate="one_to_one")
    if len(z)<100000:
        raise RuntimeError(f"launch merge too small {len(z)}")

    dates=np.array(sorted(z["date"].unique()),dtype=np.int64)
    cal_start=int(dates[np.searchsorted(dates,20230101)])
    hold_start=int(dates[np.searchsorted(dates,20240101)])
    pcal=int(np.searchsorted(dates,cal_start)); phold=int(np.searchsorted(dates,hold_start))
    if pcal<PURGE or phold<PURGE:
        raise RuntimeError("insufficient purge")
    train_end=int(dates[pcal-PURGE-1])
    cal_end=int(dates[phold-PURGE-1])

    valid=z["universe_ok"] & z["launch_session"].notna()
    tr_all=np.flatnonzero(valid & (z["date"]<=train_end))
    ca_all=np.flatnonzero(valid & (z["date"]>=cal_start) & (z["date"]<=cal_end))
    ev=np.flatnonzero(valid & (z["date"]>=hold_start) & (z["date"]<=20241231))
    if min(len(tr_all),len(ca_all),len(ev))<5000:
        raise RuntimeError("split too small")

    X=z[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    launch=z["launch_session"].to_numpy(float)

    metrics=[]
    pred=pd.DataFrame({"date":z.iloc[ev]["date"].to_numpy(),"code":z.iloc[ev]["code"].to_numpy(),
                       "realized_launch_session":launch[ev],
                       "prelaunch_mae":z.iloc[ev]["prelaunch_mae"].to_numpy()})
    for h in HORIZONS:
        y=(launch<=h).astype(np.int8)
        tr=sample_idx(tr_all,MAX_TRAIN,SEED+30000+h)
        ca=sample_idx(ca_all,MAX_CAL,SEED+31000+h)
        model=HistGradientBoostingClassifier(
            learning_rate=0.05,max_iter=110,max_leaf_nodes=31,min_samples_leaf=120,
            l2_regularization=4.0,random_state=SEED+32000+h
        )
        model.fit(X[tr],y[tr])
        raw_ca=model.predict_proba(X[ca])[:,1]
        raw_ev=model.predict_proba(X[ev])[:,1]
        p=calibrate(raw_ca,y[ca],raw_ev)
        yev=y[ev]
        base=float(y[tr].mean())
        pb=np.full(len(ev),base)
        metrics.append({
          "horizon":h,
          "definition":f"P(realized successful-launch session <= {h} | eventual +10% winner within 120 sessions)",
          "rows_train":int(len(tr)),"rows_cal":int(len(ca)),"rows_holdout":int(len(ev)),
          "holdout_rate":float(yev.mean()),
          "logloss_model":float(log_loss(yev,p,labels=[0,1])),
          "logloss_base":float(log_loss(yev,pb,labels=[0,1])),
          "brier_model":float(brier_score_loss(yev,p)),
          "brier_base":float(brier_score_loss(yev,pb)),
          "ece":ece_binary(yev,p)
        })
        pred[f"p_launch_by_{h}_cond_win120"]=p

    # Monotonicity of independently calibrated heads is diagnostic; do not silently force it.
    arr=pred[[f"p_launch_by_{h}_cond_win120" for h in HORIZONS]].to_numpy()
    mono_viol=int(np.sum(np.any(np.diff(arr,axis=1)<-1e-9,axis=1)))
    summary={
      "status":"PROTOTYPE_COMPLETE",
      "scientific_lock":False,
      "model_role":"AUXILIARY_LAUNCH_HEAD_WITHIN_NUMERICAL_OPPORTUNITY_ENGINE",
      "conditional_on":"eventual +10% winner within 120 sessions",
      "not_candidate_gate":True,
      "split":{"train_max":train_end,"calibration_min":cal_start,"calibration_max":cal_end,
               "holdout_min":hold_start,"holdout_max":20241231,"purge_sessions":PURGE},
      "feature_count":len(feats),
      "holdout_rows":int(len(ev)),
      "monotonicity_violation_rows":mono_viol,
      "metrics":metrics,
      "2025_opened":False
    }
    pred.to_parquet(ROOT/"V6_2_LAUNCH_HEAD_HOLDOUT.parquet",index=False)
    pd.DataFrame(metrics).to_csv(ROOT/"V6_2_LAUNCH_HEAD_METRICS.csv",index=False)
    (ROOT/"V6_2_LAUNCH_HEAD_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
