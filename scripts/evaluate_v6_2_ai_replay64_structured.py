#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path("final"); OUT.mkdir(exist_ok=True)
rows=[]
for f in sorted(Path("reason_inputs").glob("shard_*_responses.jsonl")):
    rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
if len(rows)!=64: raise RuntimeError(f"expected 64 responses, got {len(rows)}")
r=pd.DataFrame(rows)
o=pd.read_csv("prepared/HIDDEN_OUTCOMES.csv",dtype={"code":str})
m=r.merge(o,on=["case_id","date","code"],how="left",validate="one_to_one",suffixes=("","_out"))
if m["y_hit10_h120"].isna().any(): raise RuntimeError("outcome merge miss")
m["admitted"]=m["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])
m["resid10_h120"]=m["y_hit10_h120"]-m["p_hit10_h120_out"]
n=int(m["admitted"].sum())
def stat(mask,prefix):
    q=m[mask]
    z={f"{prefix}_n":int(len(q))}
    for t in [10,20,30,50]:
        z[f"{prefix}_hit{t}_h120"]=float(q[f"y_hit{t}_h120"].mean()) if len(q) else None
    z[f"{prefix}_mean_mfe_h120"]=float(q["mfe_h120"].mean()) if len(q) else None
    z[f"{prefix}_median_mae_h120"]=float(q["mae_h120"].median()) if len(q) else None
    z[f"{prefix}_mean_resid10_h120"]=float(q["resid10_h120"].mean()) if len(q) else None
    return z
s={"status":"DIAGNOSTIC_ONLY","n":64,"2025_opened":False,"decisions":m["decision"].value_counts().to_dict()}
s.update(stat(m["admitted"],"ai_admitted")); s.update(stat(~m["admitted"],"ai_not_admitted"))
if n:
    s.update(stat(m.index.isin(m.nlargest(n,"p_hit10_h120_out").index),"num_p10_equal_count"))
    s.update(stat(m.index.isin(m.nlargest(n,"p_hit30_h120").index),"num_p30_equal_count"))
vals=np.sort(m["p_hit10_h120_out"].unique())
best={"agreement":0.0,"cut":None}
for cut in vals:
    pred=m["p_hit10_h120_out"]>=cut
    acc=float((pred==m["admitted"]).mean())
    if acc>best["agreement"]: best={"agreement":acc,"cut":float(cut)}
s["best_single_p10_threshold_agreement_with_ai"]=best["agreement"]
s["best_single_p10_threshold_cut_diagnostic_only"]=best["cut"]
s["threshold_equivalence_warning"]=bool(best["agreement"]>=0.90)
s["warning"]="64 cases are an architecture diagnostic, not a scientific accuracy claim."
m.to_json(OUT/"AI_REPLAY64_CASES.json",orient="records",force_ascii=False,indent=2)
(OUT/"AI_REPLAY64_SUMMARY.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(s,ensure_ascii=False),flush=True)
