#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path("final"); OUT.mkdir(exist_ok=True)
rows=[]
for f in sorted(Path("reason_inputs").rglob("responses_shard_*.json")):
    rows.extend(json.loads(f.read_text()))
if len(rows)!=64: raise RuntimeError(f"expected 64 AI responses, got {len(rows)}")
r=pd.DataFrame(rows)
o=pd.read_csv("prepared/HIDDEN_OUTCOMES.csv",dtype={"code":str})
m=r.merge(o,on=["case_id","date","code"],how="left",validate="one_to_one",suffixes=("","_out"))
if m["y_hit10_h120"].isna().any(): raise RuntimeError("outcome merge miss")
m["admitted"]=m["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])
m["resid10_h120"]=m["y_hit10_h120"]-m["p_hit10_h120_out"]

n_adm=int(m["admitted"].sum())
def stat(mask,prefix):
    q=m[mask]
    d={f"{prefix}_n":int(len(q))}
    for t in [10,20,30,50]:
        d[f"{prefix}_hit{t}_h120"]=float(q[f"y_hit{t}_h120"].mean()) if len(q) else None
    d[f"{prefix}_mean_mfe_h120"]=float(q["mfe_h120"].mean()) if len(q) else None
    d[f"{prefix}_median_mae_h120"]=float(q["mae_h120"].median()) if len(q) else None
    d[f"{prefix}_mean_resid10_h120"]=float(q["resid10_h120"].mean()) if len(q) else None
    return d
summary={"status":"DIAGNOSTIC_ONLY","n":64,"2025_opened":False,"decisions":m["decision"].value_counts().to_dict()}
summary.update(stat(m["admitted"],"ai_admitted"))
summary.update(stat(~m["admitted"],"ai_not_admitted"))

# Equal-count raw numerical baselines are evaluation comparators only, never production admission rules.
if n_adm>0:
    base10=set(m.nlargest(n_adm,"p_hit10_h120_out")["case_id"])
    base30=set(m.nlargest(n_adm,"p_hit30_h120")["case_id"])
    summary.update(stat(m["case_id"].isin(base10),"num_p10_equal_count"))
    summary.update(stat(m["case_id"].isin(base30),"num_p30_equal_count"))

# Best single p_hit10 threshold agreement with AI admission: architecture diagnostic only.
s=m.sort_values("p_hit10_h120_out").reset_index(drop=True)
y=s["admitted"].astype(int).to_numpy()
best=0.0; best_cut=None
vals=s["p_hit10_h120_out"].to_numpy(float)
for i in range(len(s)+1):
    pred=np.zeros(len(s),int); pred[i:]=1
    acc=float((pred==y).mean())
    if acc>best:
        best=acc
        best_cut=float(vals[i]) if i<len(vals) else None
summary["best_single_p10_threshold_agreement_with_ai"]=best
summary["best_single_p10_threshold_cut_diagnostic_only"]=best_cut
summary["threshold_equivalence_warning"]=bool(best>=0.90)

# Failure-exit sanity: actionable outputs must all have valid pre-entry exit.
action=m[m["admitted"]]
summary["actionable_failure_exit_complete_rate"]=float(action["failure_exit"].apply(lambda z:isinstance(z,dict) and z.get("exit_price") is not None).mean()) if len(action) else None

# Entry/failure-exit path diagnostics (not a full execution backtest).
paths=pd.read_parquet("prepared/HIDDEN_FUTURE_PATHS.parquet")
triggers=0; later10=0
for _,rr in action.iterrows():
    fx=float(rr["failure_exit"]["exit_price"])
    q=paths[paths["case_id"]==rr["case_id"]].sort_values("session")
    hit=q.index[q["low"]<=fx]
    if len(hit):
        triggers+=1
        first_idx=hit[0]
        ses=int(q.loc[first_idx,"session"])
        later=q[q["session"]>ses]
        entry_ref=(float(rr["entry"]["ideal_low"])+float(rr["entry"]["ideal_high"]))/2
        if len(later) and float(later["high"].max())>=entry_ref*1.10: later10+=1
summary["failure_exit_triggered_cases_diagnostic"]=triggers
summary["failure_exit_trigger_then_later_plus10_cases_diagnostic"]=later10
summary["warning"]="64 cases are an integration/architecture diagnostic, not a scientific accuracy claim. Formal large-sample replay is still required."

m.to_json(OUT/"AI_REPLAY_64_CASES.json",orient="records",force_ascii=False,indent=2)
pd.DataFrame([summary]).to_json(OUT/"AI_REPLAY_64_SUMMARY.json",orient="records",force_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False),flush=True)
