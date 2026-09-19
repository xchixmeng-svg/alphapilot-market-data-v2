#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("v6_2_probability_surface_reconciliation"); ROOT.mkdir(exist_ok=True)
TARGETS=[10,20,30,50]
HORIZONS=[20,30,40,60,120]
TOL=1e-12
MAX_SWEEPS=40

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any():
            out += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    return float(out)

def brier(y,p): return float(np.mean((p-y)**2))

def logloss(y,p):
    p=np.clip(p,1e-8,1-1e-8)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def violations(a):
    # a: rows x target x horizon
    hv=np.zeros(len(a),dtype=bool)
    tv=np.zeros(len(a),dtype=bool)
    for t in range(len(TARGETS)):
        hv |= np.any(np.diff(a[:,t,:],axis=1)<-TOL,axis=1)
    for h in range(len(HORIZONS)):
        tv |= np.any(np.diff(a[:,:,h],axis=1)>TOL,axis=1)
    return hv,tv

def reconcile(a):
    z=a.astype(np.float64,copy=True)
    sweeps=0
    for sweep in range(1,MAX_SWEEPS+1):
        changed=False
        # Projection onto adjacent horizon halfspaces: p(h_i) <= p(h_{i+1})
        for t in range(len(TARGETS)):
            for h in range(len(HORIZONS)-1):
                left=z[:,t,h]; right=z[:,t,h+1]
                m=left>right+TOL
                if m.any():
                    mid=(left[m]+right[m])/2.0
                    left[m]=mid; right[m]=mid; changed=True
        # Projection onto adjacent target halfspaces: p(+low) >= p(+high)
        for t in range(len(TARGETS)-1):
            for h in range(len(HORIZONS)):
                low=z[:,t,h]; high=z[:,t+1,h]
                m=low<high-TOL
                if m.any():
                    mid=(low[m]+high[m])/2.0
                    low[m]=mid; high[m]=mid; changed=True
        sweeps=sweep
        hv,tv=violations(z)
        if not hv.any() and not tv.any():
            break
        if not changed:
            break
    return np.clip(z,0,1),sweeps

# Build source panel with actual labels retained.
files=sorted(Path("source_inputs").rglob("*_FULL_2024_SCORES.parquet"))
profit=[]
for f in files:
    x=pd.read_parquet(f)
    pcols=[c for c in x.columns if re.fullmatch(r"p_hit(10|20|30|50)_h(20|30|40|60|120)",c)]
    if len(pcols)!=1:
        continue
    pcol=pcols[0]
    m=re.fullmatch(r"p_hit(10|20|30|50)_h(20|30|40|60|120)",pcol)
    t,h=m.groups()
    q=x[["date","code","y_hit",pcol]].copy()
    q=q.rename(columns={"y_hit":f"y_hit{t}_h{h}"})
    profit.append((pcol,q))
if len(profit)!=20:
    raise RuntimeError(f"expected 20 profit score files, got {len(profit)}")

panel=None
for pcol,q in sorted(profit):
    panel=q if panel is None else panel.merge(q,on=["date","code"],how="inner",validate="one_to_one")

launch_files=list(Path("source_inputs").rglob("V6_2_LAUNCH_CONDITIONAL_FULL_2024_SCORES.parquet"))
if len(launch_files)!=1:
    raise RuntimeError(f"expected one launch score file, got {len(launch_files)}")
launch=pd.read_parquet(launch_files[0])
panel=panel.merge(launch,on=["date","code"],how="inner",validate="one_to_one")

raw=np.empty((len(panel),len(TARGETS),len(HORIZONS)),dtype=np.float64)
for ti,t in enumerate(TARGETS):
    for hi,h in enumerate(HORIZONS):
        raw[:,ti,hi]=panel[f"p_hit{t}_h{h}"].to_numpy(float)

pre_h,pre_t=violations(raw)
rec,sweeps=reconcile(raw)
post_h,post_t=violations(rec)

adj=np.abs(rec-raw)
metrics=[]
for ti,t in enumerate(TARGETS):
    for hi,h in enumerate(HORIZONS):
        y=panel[f"y_hit{t}_h{h}"].to_numpy()
        valid=np.isin(y,[0,1])
        yy=y[valid].astype(float)
        pr=raw[valid,ti,hi]
        pc=rec[valid,ti,hi]
        metrics.append({
          "target_pct":t,"horizon":h,"n_labeled":int(valid.sum()),
          "brier_raw":brier(yy,pr),"brier_reconciled":brier(yy,pc),
          "brier_delta":brier(yy,pc)-brier(yy,pr),
          "logloss_raw":logloss(yy,pr),"logloss_reconciled":logloss(yy,pc),
          "logloss_delta":logloss(yy,pc)-logloss(yy,pr),
          "ece_raw":ece(yy,pr),"ece_reconciled":ece(yy,pc),
          "ece_delta":ece(yy,pc)-ece(yy,pr),
          "mean_abs_adjustment":float(np.mean(np.abs(pc-pr))),
          "p95_abs_adjustment":float(np.quantile(np.abs(pc-pr),.95))
        })
        panel[f"p_hit{t}_h{h}_raw"]=raw[:,ti,hi]
        panel[f"p_hit{t}_h{h}"]=rec[:,ti,hi]

# Recompute unconditional successful-launch probabilities using reconciled P(+10 within 120).
for h in [1,3,5,10,20,30]:
    panel[f"p_successful_launch_by_{h}"]=panel["p_hit10_h120"]*panel[f"p_launch_by_{h}_cond_win120"]

launch_arr=panel[[f"p_successful_launch_by_{h}" for h in [1,3,5,10,20,30]]].to_numpy()
launch_viol=int(np.sum(np.any(np.diff(launch_arr,axis=1)<-TOL,axis=1)))

mdf=pd.DataFrame(metrics)
summary={
  "status":"COHERENT" if not post_h.any() and not post_t.any() and launch_viol==0 else "FAIL",
  "scientific_lock":False,
  "method":"cyclic symmetric Euclidean projection onto adjacent order halfspaces; validation compares pre/post probability quality",
  "rows":int(len(panel)),
  "sweeps":int(sweeps),
  "pre_horizon_violation_rows":int(pre_h.sum()),
  "pre_target_violation_rows":int(pre_t.sum()),
  "post_horizon_violation_rows":int(post_h.sum()),
  "post_target_violation_rows":int(post_t.sum()),
  "successful_launch_monotonic_violation_rows":launch_viol,
  "mean_abs_probability_adjustment":float(adj.mean()),
  "p95_abs_probability_adjustment":float(np.quantile(adj,.95)),
  "max_abs_probability_adjustment":float(adj.max()),
  "tasks_brier_improved":int((mdf["brier_delta"]<0).sum()),
  "tasks_brier_worsened":int((mdf["brier_delta"]>0).sum()),
  "mean_brier_delta":float(mdf["brier_delta"].mean()),
  "max_brier_worsening":float(mdf["brier_delta"].max()),
  "tasks_logloss_improved":int((mdf["logloss_delta"]<0).sum()),
  "tasks_logloss_worsened":int((mdf["logloss_delta"]>0).sum()),
  "mean_logloss_delta":float(mdf["logloss_delta"].mean()),
  "2025_opened":False,
  "candidate_gate":"NONE"
}
panel.to_parquet(ROOT/"V6_2_NUMERICAL_DECISION_PANEL_2024_RECONCILED.parquet",index=False)
mdf.to_csv(ROOT/"V6_2_RECONCILIATION_TASK_METRICS.csv",index=False)
(ROOT/"V6_2_RECONCILIATION_AUDIT.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary),flush=True)
if summary["status"]!="COHERENT":
    raise SystemExit(2)
