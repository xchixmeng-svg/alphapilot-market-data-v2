#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REASON=Path("inputs/reason/REAL_EVIDENCE_REASONING_RESULTS.csv")
AUDIT=Path("inputs/num/u10_d5_h20_AUDIT_SAMPLE.csv")
OUT=Path("out"); OUT.mkdir(exist_ok=True)

r=pd.read_csv(REASON,dtype={"code":str})
a=pd.read_csv(AUDIT,dtype={"code":str})
for x in (r,a):
    x["code"]=x["code"].str.zfill(4)
    x["date"]=pd.to_numeric(x["date"],errors="raise").astype(int)

keep=["date","code","actual_state","time_to_first","mfe_h","mae_h","p_up","p_down","p_neither"]
m=r.merge(a[keep],on=["date","code"],how="left",validate="one_to_one",suffixes=("","_audit"))
if m["actual_state"].isna().any():
    raise RuntimeError("historical outcome merge miss")

m["actual_up_first"]=(m["actual_state"]=="UP_FIRST").astype(int)
m["actual_down_first"]=(m["actual_state"]=="DOWN_FIRST").astype(int)
m["admitted"]=m["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])
m.to_csv(OUT/"EIGHT_CASE_HISTORICAL_OUTCOMES.csv",index=False)

by=(m.groupby("decision")
      .agg(n=("code","size"),
           up_first_rate=("actual_up_first","mean"),
           down_first_rate=("actual_down_first","mean"),
           mean_p_up=("p_up","mean"),
           mean_mfe=("mfe_h","mean"),
           mean_mae=("mae_h","mean"),
           median_time_to_first=("time_to_first","median"))
      .reset_index())
by.to_csv(OUT/"EIGHT_CASE_BY_DECISION.csv",index=False)

summary={
  "status":"DIAGNOSTIC_ONLY",
  "n":int(len(m)),
  "task":"u10_d5_h20",
  "candidate_n":int(m["admitted"].sum()),
  "candidate_up_first_rate":float(m.loc[m["admitted"],"actual_up_first"].mean()) if m["admitted"].any() else None,
  "candidate_down_first_rate":float(m.loc[m["admitted"],"actual_down_first"].mean()) if m["admitted"].any() else None,
  "watch_reject_n":int((~m["admitted"]).sum()),
  "watch_reject_up_first_rate":float(m.loc[~m["admitted"],"actual_up_first"].mean()) if (~m["admitted"]).any() else None,
  "all8_up_first_rate":float(m["actual_up_first"].mean()),
  "decisions":m["decision"].value_counts().to_dict(),
  "warning":"Eight cases are far too small for accuracy claims. This is only a smoke test of whether final AI decisions show directionally sensible separation.",
  "2025_opened":False
}
(OUT/"EIGHT_CASE_HISTORICAL_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False),flush=True)
