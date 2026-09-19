#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from run_v6_2_core_numerical_full_matrix import load,safe_features,sample_idx,SEED,MAX_TRAIN,MAX_CAL
from run_v6_2_unconditional_profit_matrix import TARGETS,add_labels,calibrate_binary

ROOT=Path("v6_2_unconditional_full_scores"); ROOT.mkdir(exist_ok=True)

def main(task):
    target,horizon=TARGETS[task]
    x,base_feats=load(); z,feats=safe_features(x,base_feats); z=add_labels(z,target,horizon)

    dates=np.array(sorted(z["date"].unique()),dtype=np.int64)
    cal_start=int(dates[np.searchsorted(dates,20230101)])
    hold_start=int(dates[np.searchsorted(dates,20240101)])
    pcal=int(np.searchsorted(dates,cal_start)); phold=int(np.searchsorted(dates,hold_start))
    train_end=int(dates[pcal-horizon-1]); cal_end=int(dates[phold-horizon-1])

    labeled=z["universe_ok"] & z["y_hit"].isin([0,1])
    tr=np.flatnonzero(labeled & (z["date"]<=train_end))
    ca=np.flatnonzero(labeled & (z["date"]>=cal_start) & (z["date"]<=cal_end))
    tr=sample_idx(tr,MAX_TRAIN,SEED+10000+horizon+int(target*100))
    ca=sample_idx(ca,MAX_CAL,SEED+20000+horizon+int(target*100))
    score=np.flatnonzero(z["universe_ok"] & (z["date"]>=hold_start) & (z["date"]<=20241231))
    if min(len(tr),len(ca),len(score))<5000: raise RuntimeError("split too small")

    X=z[feats].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32,copy=False)
    y=z["y_hit"].to_numpy(np.int8)
    model=HistGradientBoostingClassifier(
      learning_rate=0.05,max_iter=110,max_leaf_nodes=31,min_samples_leaf=120,
      l2_regularization=4.0,random_state=SEED+horizon+int(target*100)
    )
    model.fit(X[tr],y[tr])
    raw_ca=model.predict_proba(X[ca])[:,1]
    raw_score=model.predict_proba(X[score])[:,1]
    p=calibrate_binary(raw_ca,y[ca],raw_score)

    out=z.iloc[score][["date","code","y_hit","time_to_hit","mfe_h","mae_h"]].copy()
    col=f"p_hit{int(target*100)}_h{horizon}"
    out[col]=p
    out.to_parquet(ROOT/f"{task}_FULL_2024_SCORES.parquet",index=False)
    summary={
      "status":"PASS","task":task,"probability_column":col,
      "scored_2024_rows":int(len(out)),
      "labeled_eval_rows":int(out["y_hit"].isin([0,1]).sum()),
      "2025_opened":False,"candidate_gate":"NONE"
    }
    (ROOT/f"{task}_FULL_SCORE_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--task",required=True,choices=list(TARGETS))
    main(ap.parse_args().task)
