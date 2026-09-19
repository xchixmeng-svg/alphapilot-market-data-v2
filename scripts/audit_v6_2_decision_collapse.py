#!/usr/bin/env python3
from __future__ import annotations
import json,re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path("out"); OUT.mkdir(exist_ok=True)
cases_path=next(Path("inputs/final").rglob("AI_REPLAY_64_CASES.json"))
outcome_path=next(Path("inputs/prepared").rglob("HIDDEN_OUTCOMES.csv"))
manifest_paths=list(Path("inputs/prepared").rglob("cases_shard_*.jsonl"))

# Preserve code exactly as a JSON string; pd.read_json may infer numeric-looking
# stock codes and silently destroy leading zeroes (e.g. 0050 / 009829).
ai=pd.DataFrame(json.loads(cases_path.read_text()))
o=pd.read_csv(outcome_path,dtype={"code":"string"})

def normalize_code(s: pd.Series) -> pd.Series:
    x=s.astype("string").str.strip().str.replace(r"\.0$","",regex=True)
    # Taiwan security codes are at least four characters; zfill repairs accidental
    # short numeric coercions such as 50 -> 0050, while leaving 6-char codes intact.
    return x.str.zfill(4)

for name,df in (("ai",ai),("outcomes",o)):
    if "code" not in df.columns:
        raise RuntimeError(f"{name} missing code column")
    df["code"]=normalize_code(df["code"])
    if df["code"].isna().any():
        raise RuntimeError(f"{name} has null code")
    df["date"]=pd.to_numeric(df["date"],errors="raise").astype(np.int64)
    df["case_id"]=pd.to_numeric(df["case_id"],errors="raise").astype(np.int64)

if str(ai["code"].dtype) != str(o["code"].dtype):
    raise RuntimeError(f"code dtype mismatch after normalization: {ai['code'].dtype} vs {o['code'].dtype}")
packets=[]
for p in manifest_paths:
    for line in p.read_text().splitlines():
        if line.strip(): packets.append(json.loads(line))
pk=pd.DataFrame([{
    "case_id":x["case_id"],
    "avail_n":len(x["evidence"]["available_evidence_ids"]),
    "missing_n":len(x["evidence"]["missing_evidence_ids"]),
    "has_mops":int("mops_material_information" in x["evidence"]["available_evidence_ids"]),
    "has_inst":int("institutional_flow" in x["evidence"]["available_evidence_ids"]),
    "has_rev":int("revenue" in x["evidence"]["available_evidence_ids"]),
    "has_val":int("valuation" in x["evidence"]["available_evidence_ids"]),
    "p10":x["numerical_reference_read_only"].get("p_hit10_h120"),
    "p20":x["numerical_reference_read_only"].get("p_hit20_h120"),
    "p30":x["numerical_reference_read_only"].get("p_hit30_h120"),
    "launch30":x["numerical_reference_read_only"].get("p_successful_launch_by_30"),
} for x in packets])

m=ai.merge(o,on=["case_id","date","code"],how="left",validate="one_to_one",suffixes=("","_out"),indicator="_outcome_merge")
if not (m["_outcome_merge"]=="both").all():
    bad=m.loc[m["_outcome_merge"]!="both",["case_id","date","code","_outcome_merge"]]
    raise RuntimeError("outcome key mismatch after canonical code normalization: "+bad.head(10).to_json(orient="records"))
m=m.drop(columns="_outcome_merge").merge(pk,on="case_id",how="left",validate="one_to_one")
m["actual10"]=m["y_hit10_h120"].astype(int)
m["actual20"]=m["y_hit20_h120"].astype(int)
m["actual30"]=m["y_hit30_h120"].astype(int)

# Primary diagnostic: is AI collapsing regardless of calibrated probability?
bins=pd.qcut(m["p10"],4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
m["p10_quartile"]=bins.astype(str)
byq=(m.groupby("p10_quartile",observed=True)
       .agg(n=("case_id","size"),
            mean_p10=("p10","mean"),
            actual10_rate=("actual10","mean"),
            candidate_rate=("decision",lambda s:s.isin(["CANDIDATE","HIGH_CONVICTION"]).mean()),
            watch_rate=("decision",lambda s:(s=="WATCH").mean()),
            reject_rate=("decision",lambda s:(s=="REJECT").mean()))
       .reset_index())

# Inspect repeated caution language.
textcols=["decision_reason","bear_thesis","invalidation","hypothesis"]
alltext=m[textcols].fillna("").astype(str).agg(" ".join,axis=1).str.lower()
patterns={
  "missing_or_insufficient":r"missing|insufficient|unavailable|lack|缺少|不足|無法|沒有",
  "uncertain_or_caution":r"uncertain|caution|wait|confirmation|risk|不確定|觀望|等待|風險",
  "valuation":r"valuation|pe|pb|估值|本益比|股價淨值比",
  "revenue":r"revenue|營收",
  "institutional":r"institutional|foreign|trust|法人|外資|投信",
  "price_structure":r"price|support|resistance|volume|trend|價格|支撐|壓力|成交量|趨勢",
}
pattern_counts={k:int(alltext.str.contains(v,regex=True).sum()) for k,v in patterns.items()}

# Actual winners that AI rejected/watched: show strongest numerical examples without changing prompt.
fn=m[(m["actual10"]==1)&(~m["decision"].isin(["CANDIDATE","HIGH_CONVICTION"]))].copy()
fn=fn.sort_values("p10",ascending=False)
false_neg=fn[["case_id","date","code","decision","evidence_quality","p10","p20","p30","launch30","mfe_h120","mae_h120","decision_reason"]].head(12)

# Compare available evidence with actual winners/nonwinners.
ev=(m.groupby("actual10")
      .agg(n=("case_id","size"),mean_p10=("p10","mean"),mean_avail_n=("avail_n","mean"),
           has_mops_rate=("has_mops","mean"),has_inst_rate=("has_inst","mean"),
           has_rev_rate=("has_rev","mean"),has_val_rate=("has_val","mean"))
      .reset_index())

summary={
  "status":"AUDIT_COMPLETE",
  "n":int(len(m)),
  "decisions":m["decision"].value_counts().to_dict(),
  "actual_hit10_h120_rate":float(m["actual10"].mean()),
  "actual_hit20_h120_rate":float(m["actual20"].mean()),
  "actual_hit30_h120_rate":float(m["actual30"].mean()),
  "candidate_rate":float(m["decision"].isin(["CANDIDATE","HIGH_CONVICTION"]).mean()),
  "watch_rate":float((m["decision"]=="WATCH").mean()),
  "reject_rate":float((m["decision"]=="REJECT").mean()),
  "pattern_counts":pattern_counts,
  "top_quartile_actual10_rate":float(m.loc[m["p10_quartile"]=="Q4","actual10"].mean()),
  "top_quartile_candidate_rate":float(m.loc[m["p10_quartile"]=="Q4","decision"].isin(["CANDIDATE","HIGH_CONVICTION"]).mean()),
  "false_negative_count":int(len(fn)),
  "scientific_diagnosis":"DECISION_COLLAPSE" if m["decision"].isin(["CANDIDATE","HIGH_CONVICTION"]).sum()==0 else "MIXED",
  "2025_opened":False
}
(OUT/"DECISION_COLLAPSE_AUDIT.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
byq.to_csv(OUT/"BY_P10_QUARTILE.csv",index=False)
ev.to_csv(OUT/"EVIDENCE_BY_OUTCOME.csv",index=False)
false_neg.to_csv(OUT/"TOP_FALSE_NEGATIVES.csv",index=False)
m.to_json(OUT/"CASE_LEVEL_AUDIT.json",orient="records",force_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False),flush=True)
print("BY_QUARTILE")
print(byq.to_string(index=False),flush=True)
print("EVIDENCE_BY_OUTCOME")
print(ev.to_string(index=False),flush=True)
print("TOP_FALSE_NEG")
print(false_neg.head(5).to_string(index=False),flush=True)
