#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

V1=Path("inputs/v1/EVIDENCE_BUNDLE_V1.parquet")
OUT=Path("out"); OUT.mkdir(exist_ok=True)
HORIZON=120
TARGET=0.10
RESET=0.20

def date_int(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.strftime("%Y%m%d").astype(np.int64)
    n=pd.to_numeric(s,errors="raise").astype(np.int64)
    return n

x=pd.read_parquet(V1,columns=["date","code","open","high","low","close"])
x["date"]=date_int(x["date"])
x["code"]=x["code"].astype(str).str.strip().str.replace(r"\.0$","",regex=True).str.zfill(4)
if int(x["date"].max())>20241231:
    raise RuntimeError("sealed year leak")
x=x.sort_values(["code","date"]).reset_index(drop=True)
prev=x.groupby("code")["close"].shift(1)
x["_reset"]=(x["close"]/prev-1).abs()>RESET
x["_seg"]=x.groupby("code")["_reset"].cumsum().astype(np.int32)

rows=[]
for (code,seg),g in x.groupby(["code","_seg"],sort=False):
    g=g.reset_index(drop=True)
    n=len(g)
    close=g["close"].to_numpy(float)
    high=g["high"].to_numpy(float)
    low=g["low"].to_numpy(float)
    opn=g["open"].to_numpy(float)
    dates=g["date"].to_numpy(int)
    for i in range(n):
        # Full 120-session observation only, so no late-sample optimism.
        if i+HORIZON>=n:
            continue
        entry=close[i]
        fut_hi=high[i+1:i+HORIZON+1]/entry-1
        hit=np.flatnonzero(fut_hi>=TARGET)
        if len(hit)==0:
            rows.append({
                "date":int(dates[i]),"code":code,"success_h120":0,
                "first_hit_session":np.nan,"launch_session":np.nan,
                "prelaunch_mae":np.nan,"equal_trough_count":0,
                "same_day_trough_and_hit":False,"gap_hit":False
            })
            continue

        hit_s=int(hit[0]+1)
        # Include entry close as session 0 baseline. Trough is the last minimum
        # before the first +10% hit. The successful leg starts next session.
        path_lows=np.concatenate(([entry],low[i+1:i+hit_s+1]))
        min_low=float(np.min(path_lows))
        trough_idx=np.flatnonzero(np.isclose(path_lows,min_low,rtol=1e-10,atol=max(1e-8,abs(min_low)*1e-12)))
        trough_s=int(trough_idx[-1])
        equal_count=int(len(trough_idx))
        same_day=bool(trough_s==hit_s)
        if same_day:
            launch=np.nan
        else:
            launch=int(max(1,trough_s+1))
        if math_isfinite:=np.isfinite(launch):
            pre_slice=low[i+1:i+int(launch)+1]/entry-1
            pre_mae=float(np.min(pre_slice)) if len(pre_slice) else 0.0
        else:
            pre_mae=np.nan
        gap_hit=bool(opn[i+hit_s]/entry-1>=TARGET)
        rows.append({
            "date":int(dates[i]),"code":code,"success_h120":1,
            "first_hit_session":hit_s,"launch_session":launch,
            "prelaunch_mae":pre_mae,"equal_trough_count":equal_count,
            "same_day_trough_and_hit":same_day,"gap_hit":gap_hit
        })

o=pd.DataFrame(rows)
o.to_parquet(OUT/"V6_2_LAUNCH_LABEL_AUDIT_ROWS.parquet",index=False)
w=o[o["success_h120"]==1].copy()
valid=w[w["launch_session"].notna()].copy()

summary={
  "status":"AUDIT_PASS" if len(o)>10000 and len(valid)>1000 else "AUDIT_FAIL",
  "definition":"Among stock-dates that hit +10% within 120 sessions, realized launch anchor is the session after the final path trough between entry and first +10% hit. If the trough occurs on the same daily bar as the first +10% hit, label is ambiguous and excluded.",
  "rows_full_h120":int(len(o)),
  "winners_h120":int(len(w)),
  "non_winners_h120":int((o["success_h120"]==0).sum()),
  "valid_launch_labels":int(len(valid)),
  "ambiguous_same_day_trough_hit":int(w["same_day_trough_and_hit"].sum()),
  "equal_trough_multi_occurrence":int((w["equal_trough_count"]>1).sum()),
  "gap_first_hit":int(w["gap_hit"].sum()),
  "launch_by":{
    str(h):float((valid["launch_session"]<=h).mean()) for h in [1,3,5,10,20,30]
  },
  "launch_session_median":float(valid["launch_session"].median()) if len(valid) else None,
  "launch_session_p90":float(valid["launch_session"].quantile(.90)) if len(valid) else None,
  "prelaunch_mae_median":float(valid["prelaunch_mae"].median()) if len(valid) else None,
  "prelaunch_mae_p10":float(valid["prelaunch_mae"].quantile(.10)) if len(valid) else None,
  "2025_opened":False,
  "candidate_gate":"NONE"
}
(OUT/"V6_2_LAUNCH_LABEL_AUDIT_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False),flush=True)
if summary["status"]!="AUDIT_PASS":
    raise SystemExit(2)
