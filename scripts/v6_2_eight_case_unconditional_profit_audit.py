#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

V1=Path("inputs/v1/EVIDENCE_BUNDLE_V1.parquet")
REASON=Path("inputs/reason/REAL_EVIDENCE_REASONING_RESULTS.csv")
OUT=Path("out"); OUT.mkdir(exist_ok=True)
HORIZONS=[20,40,60,120]
TARGETS=[0.10,0.20,0.30,0.50]
RESET=0.20

x=pd.read_parquet(V1)
if pd.api.types.is_datetime64_any_dtype(x["date"]):
    x["date"]=pd.to_datetime(x["date"]).dt.strftime("%Y%m%d").astype(int)
else:
    x["date"]=pd.to_numeric(x["date"],errors="raise").astype(int)
x["code"]=x["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
x=x.sort_values(["code","date"]).reset_index(drop=True)
if int(x["date"].max())>20241231: raise RuntimeError("sealed year leak")
prev=x.groupby("code")["close"].shift(1)
x["_reset"]=(x["close"]/prev-1).abs()>RESET
x["_seg"]=x.groupby("code")["_reset"].cumsum().astype(int)

r=pd.read_csv(REASON,dtype={"code":str})
r["code"]=r["code"].str.zfill(4)
r["date"]=pd.to_numeric(r["date"],errors="raise").astype(int)

groups={c:g.sort_values("date").reset_index(drop=True) for c,g in x.groupby("code",sort=False)}
posmap={c:{int(d):i for i,d in enumerate(g["date"].to_numpy())} for c,g in groups.items()}
rows=[]
for _,rr in r.iterrows():
    code=str(rr["code"]); date=int(rr["date"])
    g=groups[code]; pos=posmap[code][date]
    entry=float(g.iloc[pos]["close"]); seg=int(g.iloc[pos]["_seg"])
    rec={"date":date,"code":code,"decision":rr["decision"],"entry_close":entry}
    for h in HORIZONS:
        future=g.iloc[pos+1:min(len(g),pos+h+1)].copy()
        future=future[(future["_seg"]==seg)&(future["date"]<=20241231)]
        rec[f"h{h}_sessions_observed"]=int(len(future))
        if len(future)==0:
            rec[f"h{h}_mfe"]=None; rec[f"h{h}_mae"]=None; rec[f"h{h}_peak_session"]=None
            for t in TARGETS: rec[f"h{h}_hit_{int(t*100)}"]=None
            continue
        fav=future["high"].astype(float)/entry-1
        adv=future["low"].astype(float)/entry-1
        rec[f"h{h}_mfe"]=float(fav.max())
        rec[f"h{h}_mae"]=float(adv.min())
        rec[f"h{h}_peak_session"]=int(np.argmax(fav.to_numpy())+1)
        for t in TARGETS:
            hits=np.flatnonzero(fav.to_numpy()>=t)
            rec[f"h{h}_hit_{int(t*100)}"]=bool(len(hits))
            rec[f"h{h}_time_to_{int(t*100)}"]=int(hits[0]+1) if len(hits) else None
    rows.append(rec)

o=pd.DataFrame(rows)
o.to_csv(OUT/"EIGHT_CASE_UNCONDITIONAL_PROFIT_OUTCOMES.csv",index=False)
cand=o[o["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])]
other=o[~o["decision"].isin(["CANDIDATE","HIGH_CONVICTION"])]
summary={"status":"DIAGNOSTIC_ONLY","n":int(len(o)),"2025_opened":False,"primary_definition":"unconditional future upside; interim drawdown does not invalidate winner"}
for h in HORIZONS:
    summary[f"h{h}_candidate_n"]=int(len(cand))
    summary[f"h{h}_candidate_mean_mfe"]=float(cand[f"h{h}_mfe"].mean()) if len(cand) else None
    summary[f"h{h}_other_mean_mfe"]=float(other[f"h{h}_mfe"].mean()) if len(other) else None
    for t in [10,20,30,50]:
        ck=f"h{h}_hit_{t}"
        summary[f"h{h}_candidate_hit{t}_rate"]=float(cand[ck].dropna().mean()) if cand[ck].notna().any() else None
        summary[f"h{h}_other_hit{t}_rate"]=float(other[ck].dropna().mean()) if other[ck].notna().any() else None
(OUT/"EIGHT_CASE_UNCONDITIONAL_PROFIT_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False),flush=True)
print(o[["date","code","decision"]+[f"h{h}_mfe" for h in HORIZONS]+[f"h120_hit_{t}" for t in [10,20,30,50]]].to_string(index=False),flush=True)
