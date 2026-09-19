#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd

V2=Path("inputs/v2/EVIDENCE_BUNDLE_V2.parquet")
PANEL=Path("inputs/panel/V6_2_NUMERICAL_DECISION_PANEL_2024_RECONCILED.parquet")
OUT=Path("prepared"); OUT.mkdir(exist_ok=True)
N_DATES=8; N_STRATA=8; RESET=0.20

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception: return None

def norm_date(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.strftime("%Y%m%d").astype(np.int64)
    return pd.to_numeric(s,errors="raise").astype(np.int64)

def hashed_pick(d,code):
    return hashlib.sha256(f"{d}|{code}|V6.2-AI-REPLAY-64".encode()).hexdigest()

p=pd.read_parquet(PANEL)
p["date"]=norm_date(p["date"])
p["code"]=p["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
if int(p["date"].max())>20241231: raise RuntimeError("2025 panel leak")
# Full 120-session labels only. Label availability is used only to ensure evaluability,
# never label value for case selection.
eligible=p[p["y_hit10_h120"].isin([0,1])].copy()
dates=np.array(sorted(eligible["date"].unique()),dtype=np.int64)
if len(dates)<N_DATES: raise RuntimeError("not enough eligible dates")
pick_dates=sorted(set(int(dates[int(i)]) for i in np.linspace(0,len(dates)-1,N_DATES,dtype=int)))
if len(pick_dates)!=N_DATES: raise RuntimeError("date de-dup reduced sample design")

selected=[]
for d in pick_dates:
    q=eligible[eligible["date"]==d].copy()
    if len(q)<N_STRATA*10: raise RuntimeError(f"date {d} too small")
    q["p_stratum"]=pd.qcut(q["p_hit10_h120"],N_STRATA,labels=False,duplicates="drop")
    if q["p_stratum"].nunique()!=N_STRATA:
        # deterministic rank bins only for validation sampling; never candidate admission.
        q=q.sort_values(["p_hit10_h120","code"]).reset_index(drop=True)
        q["p_stratum"]=np.minimum(N_STRATA-1,(np.arange(len(q))*N_STRATA//len(q))).astype(int)
    for s in range(N_STRATA):
        z=q[q["p_stratum"]==s].copy()
        z["_pick"]=z["code"].map(lambda c: hashed_pick(d,c))
        selected.append(z.sort_values("_pick").iloc[0])
sel=pd.DataFrame(selected).drop(columns=["_pick"],errors="ignore")
if len(sel)!=N_DATES*N_STRATA or sel.duplicated(["date","code"]).any():
    raise RuntimeError("bad deterministic case selection")

x=pd.read_parquet(V2)
x["date"]=norm_date(x["date"])
x["code"]=x["code"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
if int(x["date"].max())>20241231: raise RuntimeError("2025 evidence leak")
x=x.sort_values(["code","date"]).reset_index(drop=True)

# PIT price/volume structure known at T close.
g=x.groupby("code",group_keys=False)
prev=g["close"].shift(1)
x["_reset"]=(x["close"]/prev-1).abs()>RESET
x["_seg"]=x.groupby("code")["_reset"].cumsum().astype(np.int32)
sg=x.groupby(["code","_seg"],group_keys=False)
for w in [5,10,20,60]:
    x[f"low{w}"]=sg["low"].transform(lambda s:s.rolling(w,min_periods=max(2,w//2)).min())
    x[f"high{w}"]=sg["high"].transform(lambda s:s.rolling(w,min_periods=max(2,w//2)).max())
for w in [5,20,60]:
    x[f"ret{w}"]=sg["close"].pct_change(w)
x["vol_mean20"]=sg["volume"].transform(lambda s:s.rolling(20,min_periods=10).mean())
x["volume_ratio20"]=x["volume"]/x["vol_mean20"].replace(0,np.nan)
tr=pd.concat([(x["high"]-x["low"]).abs(),(x["high"]-prev).abs(),(x["low"]-prev).abs()],axis=1).max(axis=1)
x["atr20"]=tr.groupby([x["code"],x["_seg"]]).transform(lambda s:s.rolling(20,min_periods=10).mean())
x["close_pos60"]=(x["close"]-x["low60"])/(x["high60"]-x["low60"]).replace(0,np.nan)
for c in ["inst_foreign_net_ratio","inst_trust_net_ratio","inst_dealer_net_ratio"]:
    if c in x:
        for w in [5,20]:
            x[f"{c}_mean{w}"]=x.groupby(["code","_seg"])[c].transform(lambda s:s.rolling(w,min_periods=max(2,w//2)).mean())

idx=x.set_index(["date","code"],drop=False)
missing=[(int(r.date),str(r.code)) for _,r in sel.iterrows() if (int(r.date),str(r.code)) not in idx.index]
if missing: raise RuntimeError(f"evidence merge misses {missing[:5]}")

prob_cols=[f"p_hit{t}_h{h}" for t in [10,20,30,50] for h in [20,30,40,60,120]]
launch_cols=[f"p_successful_launch_by_{h}" for h in [1,3,5,10,20,30]]
cases=[]; outcomes=[]; paths=[]
for case_id,(_,r) in enumerate(sel.sort_values(["date","p_stratum","code"]).iterrows()):
    d=int(r["date"]); code=str(r["code"]); er=idx.loc[(d,code)]
    if isinstance(er,pd.DataFrame): er=er.iloc[0]
    event_titles=[]
    try:
        raw=er.get("event_titles_json")
        if pd.notna(raw):
            event_titles=json.loads(raw)
            if not isinstance(event_titles,list): event_titles=[]
    except Exception: event_titles=[]

    available=["price_volume_structure"]
    if any(pd.notna(er.get(c)) for c in ["inst_foreign_net_ratio","inst_trust_net_ratio","inst_dealer_net_ratio"]): available.append("institutional_flow")
    if any(pd.notna(er.get(c)) for c in ["rev_yoy_pct","rev_mom_pct"]): available.append("revenue")
    if any(pd.notna(er.get(c)) for c in ["valuation_pe","valuation_pb","valuation_dividend_yield_pct"]): available.append("valuation")
    if event_titles: available.append("mops_material_information")

    evidence={
      "available_evidence_ids":available,
      "missing_evidence_ids":["eps_revisions","analyst_consensus","industry_pricing","inventory_supply_demand","broad_news_semantics"],
      "price_volume_structure":{
        "current_price":finite(er.get("close")),"ret5_pct":None if finite(er.get("ret5")) is None else finite(er.get("ret5"))*100,
        "ret20_pct":None if finite(er.get("ret20")) is None else finite(er.get("ret20"))*100,
        "ret60_pct":None if finite(er.get("ret60")) is None else finite(er.get("ret60"))*100,
        "volume_ratio20":finite(er.get("volume_ratio20")),"atr20":finite(er.get("atr20")),
        "support_low5":finite(er.get("low5")),"support_low10":finite(er.get("low10")),
        "support_low20":finite(er.get("low20")),"support_low60":finite(er.get("low60")),
        "high20":finite(er.get("high20")),"high60":finite(er.get("high60")),
        "close_position_60_range":finite(er.get("close_pos60"))
      },
      "institutional_flow":{
        "foreign_net_ratio_prev_session":finite(er.get("inst_foreign_net_ratio")),
        "trust_net_ratio_prev_session":finite(er.get("inst_trust_net_ratio")),
        "dealer_net_ratio_prev_session":finite(er.get("inst_dealer_net_ratio")),
        "foreign_mean5":finite(er.get("inst_foreign_net_ratio_mean5")),
        "foreign_mean20":finite(er.get("inst_foreign_net_ratio_mean20")),
        "trust_mean5":finite(er.get("inst_trust_net_ratio_mean5")),
        "trust_mean20":finite(er.get("inst_trust_net_ratio_mean20"))
      },
      "revenue":{"yoy_pct":finite(er.get("rev_yoy_pct")),"mom_pct":finite(er.get("rev_mom_pct"))},
      "valuation":{"pe":finite(er.get("valuation_pe")),"pb":finite(er.get("valuation_pb")),
                   "dividend_yield_pct":finite(er.get("valuation_dividend_yield_pct"))},
      "mops_material_information":{"titles":event_titles[:6],"event_age_hours":finite(er.get("event_last_age_hours"))}
    }
    numerical={c:finite(r[c]) for c in prob_cols+launch_cols}
    packet={"case_id":case_id,"decision_date":d,"code":code,"validation_stratum":int(r["p_stratum"]),
            "evidence":evidence,"numerical_reference_read_only":numerical}
    cases.append(packet)

    # Outcome data stays physically separate from LLM packets.
    o={"case_id":case_id,"date":d,"code":code,"validation_stratum":int(r["p_stratum"])}
    for t in [10,20,30,50]:
        for h in [20,30,40,60,120]:
            o[f"y_hit{t}_h{h}"]=int(r[f"y_hit{t}_h{h}"]) if int(r[f"y_hit{t}_h{h}"]) in [0,1] else -1
            o[f"p_hit{t}_h{h}"]=float(r[f"p_hit{t}_h{h}"])
    # Realized path from V2, restricted to same price segment.
    pos=int(er.name[0]) if False else None
    gx=x[(x["code"]==code)&(x["_seg"]==er["_seg"])].reset_index(drop=True)
    loc=np.flatnonzero(gx["date"].to_numpy()==d)
    if len(loc)!=1: raise RuntimeError(f"path origin miss {d} {code}")
    i=int(loc[0]); entry=float(gx.loc[i,"close"])
    for h in [20,40,60,120]:
        if i+h>=len(gx):
            o[f"mfe_h{h}"]=np.nan; o[f"mae_h{h}"]=np.nan
            continue
        hi=gx.loc[i+1:i+h,"high"].to_numpy(float)/entry-1
        lo=gx.loc[i+1:i+h,"low"].to_numpy(float)/entry-1
        o[f"mfe_h{h}"]=float(np.max(hi)); o[f"mae_h{h}"]=float(np.min(lo))
    for t in [10,20,30,50]:
        fut=gx.loc[i+1:min(i+120,len(gx)-1),"high"].to_numpy(float)/entry-1
        hit=np.flatnonzero(fut>=t/100)
        o[f"time_to_hit{t}"]=int(hit[0]+1) if len(hit) else np.nan
    outcomes.append(o)

    for j in range(1,min(120,len(gx)-i-1)+1):
        rr=gx.loc[i+j]
        paths.append({"case_id":case_id,"session":j,"open":float(rr["open"]),"high":float(rr["high"]),
                      "low":float(rr["low"]),"close":float(rr["close"])})

# Four shards of 16 cases, deterministic.
for shard in range(4):
    shard_cases=[c for c in cases if c["case_id"]%4==shard]
    with (OUT/f"cases_shard_{shard}.jsonl").open("w") as f:
        for c in shard_cases: f.write(json.dumps(c,ensure_ascii=False,separators=(",",":"))+"\n")
pd.DataFrame(outcomes).to_csv(OUT/"HIDDEN_OUTCOMES.csv",index=False)
pd.DataFrame(paths).to_parquet(OUT/"HIDDEN_FUTURE_PATHS.parquet",index=False)
pd.DataFrame([{"case_id":c["case_id"],"decision_date":c["decision_date"],"code":c["code"],
               "validation_stratum":c["validation_stratum"]} for c in cases]).to_csv(OUT/"CASE_MANIFEST.csv",index=False)
summary={"status":"PASS","cases":len(cases),"dates":pick_dates,"strata_per_date":N_STRATA,
         "selection_uses_future_outcome":False,"selection_rule":"8 dates x 8 p_hit10_h120 validation strata; within stratum deterministic hash pick, not rank pick",
         "2025_opened":False}
(OUT/"PREP_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False),flush=True)
