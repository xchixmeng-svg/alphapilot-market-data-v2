#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

from run_v6_2_core_numerical_full_matrix import load, safe_features, sample_idx, SEED, MAX_TRAIN, MAX_CAL

ROOT=Path("v6_2_unconditional_profit_matrix"); ROOT.mkdir(exist_ok=True)

TARGETS={}
for pct in (10,20,30,50):
    for h in (20,30,40,60,120):
        TARGETS[f"u{pct}_h{h}"]=(pct/100.0,h)

def label_one_segment(g,target,horizon):
    n=len(g)
    close=g["close"].to_numpy(float)
    high=g["high"].to_numpy(float)
    low=g["low"].to_numpy(float)
    y=np.full(n,-1,dtype=np.int8)  # -1 censored / insufficient full horizon
    t_hit=np.full(n,np.nan,dtype=np.float32)
    mfe=np.full(n,np.nan,dtype=np.float32)
    mae=np.full(n,np.nan,dtype=np.float32)
    # Strict full-horizon labels only; no asymmetric positive-only late-year labels.
    full=np.arange(n)+horizon < n
    for i in np.flatnonzero(full):
        hh=high[i+1:i+horizon+1]/close[i]-1.0
        ll=low[i+1:i+horizon+1]/close[i]-1.0
        mfe[i]=float(np.max(hh)); mae[i]=float(np.min(ll))
        hit=np.flatnonzero(hh>=target)
        y[i]=1 if len(hit) else 0
        if len(hit): t_hit[i]=int(hit[0]+1)
    return y,t_hit,mfe,mae

def add_labels(z,target,horizon):
    y=np.full(len(z),-1,dtype=np.int8)
    th=np.full(len(z),np.nan,np.float32)
    mfe=np.full(len(z),np.nan,np.float32)
    mae=np.full(len(z),np.nan,np.float32)
    for _,idx in z.groupby(["code","price_segment_id"],sort=False).groups.items():
        ii=np.asarray(idx,dtype=np.int64)
        a,b,c,d=label_one_segment(z.loc[ii],target,horizon)
        y[ii]=a; th[ii]=b; mfe[ii]=c; mae[ii]=d
    q=z.copy()
    q["y_hit"]=y; q["time_to_hit"]=th; q["mfe_h"]=mfe; q["mae_h"]=mae
    return q

def calibrate_binary(raw_cal,y_cal,raw_eval):
    ir=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip")
    ir.fit(raw_cal,y_cal.astype(float))
    p=ir.predict(raw_eval)
    return np.clip(p,1e-6,1-1e-6)

def ece_binary(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); e=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any(): e += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(e)

def run(task):
    target,horizon=TARGETS[task]
    x,base_feats=load()
    z,feats=safe_features(x,base_feats)
    z=add_labels(z,target,horizon)

    dates=np.array(sorted(z["date"].unique()),dtype=np.int64)
    cal_start=int(dates[np.searchsorted(dates,20230101)])
    hold_start=int(dates[np.searchsorted(dates,20240101)])
    pcal=int(np.searchsorted(dates,cal_start)); phold=int(np.searchsorted(dates,hold_start))
    if pcal<horizon or phold<horizon: raise RuntimeError("insufficient purge")
    train_end=int(dates[pcal-horizon-1]); cal_end=int(dates[phold-horizon-1])

    valid=z["universe_ok"] & z["y_hit"].isin([0,1])
    tr=np.flatnonzero(valid & (z["date"]<=train_end))
    ca=np.flatnonzero(valid & (z["date"]>=cal_start) & (z["date"]<=cal_end))
    ev=np.flatnonzero(valid & (z["date"]>=hold_start) & (z["date"]<=20241231))
    tr=sample_idx(tr,MAX_TRAIN,SEED+10000+horizon+int(target*100))
    ca=sample_idx(ca,MAX_CAL,SEED+20000+horizon+int(target*100))
    if min(len(tr),len(ca),len(ev))<5000: raise RuntimeError(f"split too small {len(tr)} {len(ca)} {len(ev)}")

    X=z[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    y=z["y_hit"].to_numpy(np.int8)
    model=HistGradientBoostingClassifier(
        learning_rate=0.05,max_iter=110,max_leaf_nodes=31,min_samples_leaf=120,
        l2_regularization=4.0,random_state=SEED+horizon+int(target*100)
    )
    model.fit(X[tr],y[tr])
    if not np.array_equal(model.classes_,np.array([0,1])):
        raise RuntimeError(f"unexpected classes {model.classes_}")
    raw_cal=model.predict_proba(X[ca])[:,1]
    raw_ev=model.predict_proba(X[ev])[:,1]
    p=calibrate_binary(raw_cal,y[ca],raw_ev)
    yev=y[ev].astype(int)
    base_rate=float(y[tr].mean())
    base=np.full(len(ev),base_rate,dtype=float)

    q90=float(np.quantile(p,.90)); q95=float(np.quantile(p,.95))
    top10=p>=q90; top5=p>=q95
    hold_rate=float(yev.mean())

    result={
      "status":"PROTOTYPE_DIAGNOSTIC_ONLY",
      "label_family":"UNCONDITIONAL_UPSIDE",
      "task":task,"target":target,"horizon":horizon,
      "definition":f"Hit +{int(target*100)}% within {horizon} sessions regardless of interim drawdown; only full-horizon labels are used.",
      "split":{"train_max":train_end,"calibration_min":cal_start,"calibration_max":cal_end,
               "holdout_min":hold_start,"holdout_max":20241231,"purge_sessions":horizon},
      "rows":{"train":int(len(tr)),"calibration":int(len(ca)),"holdout":int(len(ev))},
      "metrics":{
        "holdout_hit_rate":hold_rate,
        "logloss_model":float(log_loss(yev,p,labels=[0,1])),
        "logloss_base":float(log_loss(yev,base,labels=[0,1])),
        "brier_model":float(brier_score_loss(yev,p)),
        "brier_base":float(brier_score_loss(yev,base)),
        "ece":ece_binary(yev,p),
        "top10_prob_threshold_diagnostic_only":q90,
        "top10_actual_hit_rate":float(yev[top10].mean()),
        "top10_lift_vs_holdout_base":float(yev[top10].mean()/hold_rate) if hold_rate else None,
        "top5_prob_threshold_diagnostic_only":q95,
        "top5_actual_hit_rate":float(yev[top5].mean()),
        "top5_lift_vs_holdout_base":float(yev[top5].mean()/hold_rate) if hold_rate else None,
      },
      "feature_count":len(feats),
      "censored_rows_total":int((z["y_hit"]==-1).sum()),
      "sealed_2025_rows_read":0,
      "admission_rule":"NONE; probabilities are evidence for AI reasoning only"
    }

    ed=z.iloc[ev][["date","code","time_to_hit","mfe_h","mae_h"]].copy()
    ed["actual_hit"]=yev; ed["p_hit"]=p
    try: ed["p_decile"]=pd.qcut(ed["p_hit"],10,labels=False,duplicates="drop")
    except Exception: ed["p_decile"]=0
    dec=(ed.groupby("p_decile",dropna=False)
      .agg(n=("code","size"),mean_p=("p_hit","mean"),actual_hit_rate=("actual_hit","mean"),
           mean_mfe=("mfe_h","mean"),mean_mae=("mae_h","mean"),
           median_time_to_hit=("time_to_hit","median"))
      .reset_index())
    dec.to_csv(ROOT/f"{task}_PROBABILITY_DECILES.csv",index=False)
    (ROOT/f"{task}_SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    ed.sort_values(["date","p_hit","code"],ascending=[True,False,True]).groupby("date").head(25).to_csv(
        ROOT/f"{task}_AUDIT_SAMPLE.csv",index=False
    )
    print(json.dumps(result,ensure_ascii=False),flush=True)

def aggregate(inp):
    rows=[]
    for p in Path(inp).rglob("*_SUMMARY.json"):
        rows.append(json.loads(p.read_text()))
    if len(rows)!=len(TARGETS): raise RuntimeError(f"expected {len(TARGETS)} summaries, got {len(rows)}")
    rows=sorted(rows,key=lambda r:(r["target"],r["horizon"]))
    out={
      "layer":"V6.2 Unconditional Primary Profit Numerical Matrix",
      "status":"PROTOTYPE_COMPLETE",
      "scientific_lock":False,
      "2025_opened":False,
      "task_count":len(rows),
      "tasks":rows,
      "interpretation_rule":"Primary labels are unconditional. No probability threshold, TopK, or deterministic candidate admission rule is chosen from these results. Legacy barrier matrix is secondary path-risk diagnostics only."
    }
    (ROOT/"V6_2_UNCONDITIONAL_PROFIT_MATRIX_SUMMARY.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    pd.DataFrame([{"task":r["task"],"target":r["target"],"horizon":r["horizon"],**r["metrics"]} for r in rows]).to_csv(
        ROOT/"V6_2_UNCONDITIONAL_PROFIT_MATRIX_METRICS.csv",index=False
    )
    print(json.dumps({"status":out["status"],"task_count":len(rows),"2025_opened":False},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--task",choices=list(TARGETS)); ap.add_argument("--aggregate")
    ns=ap.parse_args()
    if bool(ns.task)==bool(ns.aggregate): raise SystemExit("choose exactly one")
    run(ns.task) if ns.task else aggregate(ns.aggregate)
