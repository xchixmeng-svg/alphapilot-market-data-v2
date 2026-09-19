#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("v6_2_numerical_decision_panel"); ROOT.mkdir(exist_ok=True)
files=sorted(Path("matrix_inputs").rglob("*_FULL_2024_SCORES.parquet"))
if len(files)!=20: raise RuntimeError(f"expected 20 profit score files, got {len(files)}")
panel=None
for f in files:
    x=pd.read_parquet(f)
    pcols=[c for c in x.columns if c.startswith("p_hit")]
    if len(pcols)!=1: raise RuntimeError(f"bad probability cols {f} {pcols}")
    q=x[["date","code",pcols[0]]]
    panel=q if panel is None else panel.merge(q,on=["date","code"],how="inner",validate="one_to_one")
launch=pd.read_parquet(next(Path("launch_inputs").rglob("V6_2_LAUNCH_CONDITIONAL_FULL_2024_SCORES.parquet")))
panel=panel.merge(launch,on=["date","code"],how="inner",validate="one_to_one")

targets=[10,20,30,50]; horizons=[20,30,40,60,120]
hviol=np.zeros(len(panel),dtype=bool); tviol=np.zeros(len(panel),dtype=bool)
hmax=np.zeros(len(panel)); tmax=np.zeros(len(panel))
for t in targets:
    a=panel[[f"p_hit{t}_h{h}" for h in horizons]].to_numpy()
    d=np.diff(a,axis=1)
    hviol |= np.any(d < -1e-12,axis=1)
    hmax=np.maximum(hmax,np.maximum(0,-d.min(axis=1)))
for h in horizons:
    a=panel[[f"p_hit{t}_h{h}" for t in targets]].to_numpy()
    d=np.diff(a,axis=1)
    tviol |= np.any(d > 1e-12,axis=1)
    tmax=np.maximum(tmax,np.maximum(0,d.max(axis=1)))

# Combine successful-launch timing with unconditional +10% winner probability.
for h in [1,3,5,10,20,30]:
    panel[f"p_successful_launch_by_{h}"]=panel["p_hit10_h120"]*panel[f"p_launch_by_{h}_cond_win120"]

launch_cols=[f"p_successful_launch_by_{h}" for h in [1,3,5,10,20,30]]
launch_arr=panel[launch_cols].to_numpy()
lviol=np.any(np.diff(launch_arr,axis=1)<-1e-12,axis=1)

panel["profit_horizon_monotonic_violation"]=hviol
panel["profit_target_monotonic_violation"]=tviol
panel.to_parquet(ROOT/"V6_2_NUMERICAL_DECISION_PANEL_2024.parquet",index=False)
summary={
 "status":"NEEDS_RECONCILIATION" if (hviol.any() or tviol.any()) else "PASS",
 "rows":int(len(panel)),
 "profit_horizon_monotonic_violation_rows":int(hviol.sum()),
 "profit_target_monotonic_violation_rows":int(tviol.sum()),
 "max_horizon_violation":float(hmax.max()),
 "max_target_violation":float(tmax.max()),
 "successful_launch_monotonic_violation_rows":int(lviol.sum()),
 "launch_combination_rule":"P(successful launch by h)=P(+10% within 120)*P(launch by h | +10% winner within 120)",
 "candidate_gate":"NONE",
 "2025_opened":False
}
(ROOT/"V6_2_NUMERICAL_DECISION_PANEL_AUDIT.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary),flush=True)
