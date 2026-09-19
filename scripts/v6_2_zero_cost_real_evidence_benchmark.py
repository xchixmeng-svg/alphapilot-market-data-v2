#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, time, urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

MODEL=os.environ.get("V62_LOCAL_MODEL","qwen2.5:3b")
V1=Path("inputs/v1/EVIDENCE_BUNDLE_V1.parquet")
NUM=Path("inputs/num/u10_d5_h20_AUDIT_SAMPLE.csv")
OUT=Path("out"); OUT.mkdir(exist_ok=True)

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

x=pd.read_parquet(V1)
if pd.api.types.is_datetime64_any_dtype(x["date"]):
    x["date"]=pd.to_datetime(x["date"]).dt.strftime("%Y%m%d").astype(int)
else:
    x["date"]=pd.to_numeric(x["date"],errors="raise").astype(int)
x["code"]=x["code"].astype(str).str.replace(r"\\.0$","",regex=True).str.zfill(4)
x=x.sort_values(["code","date"]).reset_index(drop=True)
if int(x["date"].max())>20241231: raise RuntimeError("sealed year leak")
g=x.groupby("code",group_keys=False)
x["return_20_sessions_pct"]=(g["close"].pct_change(20)*100).astype(float)
x["return_60_sessions_pct"]=(g["close"].pct_change(60)*100).astype(float)
hi60=g["high"].transform(lambda s:s.rolling(60,min_periods=40).max())
lo60=g["low"].transform(lambda s:s.rolling(60,min_periods=40).min())
x["close_position_in_60_session_range_pct"]=((x["close"]-lo60)/(hi60-lo60).replace(0,np.nan)*100).astype(float)

n=pd.read_csv(NUM,dtype={"code":str})
n["code"]=n["code"].str.zfill(4)
n["date"]=pd.to_numeric(n["date"],errors="raise").astype(int)
n=n.sort_values(["date","p_up","code"],ascending=[True,False,True])
dates=sorted(n["date"].unique())
pick_dates=[dates[int(i)] for i in np.linspace(0,len(dates)-1,8,dtype=int)]
sel=[]
for j,d in enumerate(pick_dates):
    q=n[n["date"]==d].sort_values(["p_up","code"],ascending=[False,True])
    sel.append(q.iloc[min(len(q)-1,j%max(1,len(q)))])
s=pd.DataFrame(sel)
m=s.merge(x,on=["date","code"],how="left",validate="one_to_one")
if m["close"].isna().any(): raise RuntimeError("real evidence merge miss")

system="""You are V6.2 Taiwan-stock opportunity reasoning layer.
Judge each stock independently from supplied point-in-time evidence. Never rank stocks.
Return exactly one JSON object and nothing else.
Never invent unavailable evidence. Revenue growth is not EPS revision evidence.
Decision must be REJECT, WATCH, CANDIDATE, HIGH_CONVICTION, or INSUFFICIENT_INPUT.
Every non-INSUFFICIENT decision requires why_now, non-empty evidence_used, non-empty counter_evidence, and a concrete invalidation.
Numerical engine values are immutable and read-only. Do not copy or emit them.
Do not use TopK, rank, quotas, or forced candidate counts.
A missing evidence family is uncertainty/counter-evidence; do not convert it into a positive claim.
20-session and 60-session returns are explicitly trading-session returns, not annual returns.
This is historical research output, not a trading recommendation."""

rows=[]; errors=[]
for _,r in m.iterrows():
    evidence={
      "decision_date":int(r["date"]),"code":str(r["code"]),
      "price_volume_structure":{
        "status":"AVAILABLE","close":finite(r.get("close")),"volume":finite(r.get("volume")),
        "return_20_sessions_pct":finite(r.get("return_20_sessions_pct")),
        "return_60_sessions_pct":finite(r.get("return_60_sessions_pct")),
        "close_position_in_60_session_range_pct":finite(r.get("close_position_in_60_session_range_pct"))
      },
      "institutional_flow":{
        "status":"AVAILABLE" if pd.notna(r.get("inst_foreign_net_ratio")) else "UNAVAILABLE",
        "foreign_net_ratio_prev_session":finite(r.get("inst_foreign_net_ratio")),
        "trust_net_ratio_prev_session":finite(r.get("inst_trust_net_ratio")),
        "dealer_net_ratio_prev_session":finite(r.get("inst_dealer_net_ratio"))
      },
      "revenue_earnings":{
        "status":"PARTIAL",
        "latest_revenue_yoy_pct":finite(r.get("rev_yoy_pct")),
        "latest_revenue_mom_pct":finite(r.get("rev_mom_pct")),
        "missing":["earnings","margins","eps"]
      },
      "eps_revisions":{"status":"UNAVAILABLE"},
      "valuation":{
        "status":"AVAILABLE",
        "pe":finite(r.get("valuation_pe")),"pb":finite(r.get("valuation_pb")),
        "dividend_yield_pct":finite(r.get("valuation_dividend_yield_pct"))
      },
      "corporate_events":{
        "status":"PARTIAL",
        "event_cum_count":finite(r.get("event_cum_count")),
        "event_last_age_hours":finite(r.get("event_last_age_hours")),
        "semantic_event_text":"UNAVAILABLE"
      },
      "industry_pricing_supply_demand":{"status":"UNAVAILABLE"},
      "news_disclosed_catalysts":{"status":"UNAVAILABLE"}
    }
    numerical={"task":"u10_d5_h20","p_up10_before_down5_h20":finite(r["p_up"]),"p_down_first_h20":finite(r["p_down"]),"p_neither_h20":finite(r["p_neither"])}
    payload={"evidence":evidence,"numerical_reference_read_only":numerical,
             "response_schema":{"decision":"enum","why_now":"string","evidence_used":"list[string]","counter_evidence":"list[string]","invalidation":"string"}}
    body=json.dumps({"model":MODEL,"messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,ensure_ascii=False,separators=(",",":"))}],
                     "stream":False,"format":"json","options":{"temperature":0,"num_predict":360}}).encode()
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=900) as resp: out=json.loads(resp.read().decode())
        elapsed=time.time()-t0
        obj=json.loads(out["message"]["content"].strip())
        allowed={"decision","why_now","evidence_used","counter_evidence","invalidation"}
        if set(obj)!=allowed: raise RuntimeError(f"schema keys {sorted(obj)}")
        if obj["decision"] not in {"REJECT","WATCH","CANDIDATE","HIGH_CONVICTION","INSUFFICIENT_INPUT"}: raise RuntimeError("bad decision")
        if obj["decision"]!="INSUFFICIENT_INPUT":
            if not isinstance(obj["why_now"],str) or not obj["why_now"].strip(): raise RuntimeError("empty why_now")
            if not isinstance(obj["evidence_used"],list) or not obj["evidence_used"]: raise RuntimeError("empty evidence")
            if not isinstance(obj["counter_evidence"],list) or not obj["counter_evidence"]: raise RuntimeError("empty counter")
            if not isinstance(obj["invalidation"],str) or not obj["invalidation"].strip(): raise RuntimeError("empty invalidation")
        txt=json.dumps(obj,ensure_ascii=False).lower()
        forbidden=["eps revisions are rising","eps revision is rising","eps revisions improving","eps revision improving","past year","over the last year","yearly return"]
        if any(z in txt for z in forbidden): raise RuntimeError("semantic hallucination")
        rows.append({"date":int(r["date"]),"code":str(r["code"]),"elapsed_seconds":elapsed,**numerical,**obj})
        print(json.dumps({"date":int(r["date"]),"code":str(r["code"]),"decision":obj["decision"],"elapsed_seconds":round(elapsed,3)},ensure_ascii=False),flush=True)
    except Exception as e:
        errors.append({"date":int(r["date"]),"code":str(r["code"]),"error":repr(e)})
        print("ERROR",errors[-1],flush=True)

res=pd.DataFrame(rows)
res.to_csv(OUT/"REAL_EVIDENCE_REASONING_RESULTS.csv",index=False)
(OUT/"REAL_EVIDENCE_REASONING_ERRORS.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2))
summary={"status":"PASS" if not errors and len(rows)==8 else "FAIL","model":MODEL,"api_cost_usd":0,"requested":8,"completed":len(rows),"errors":len(errors),
         "mean_seconds":float(res["elapsed_seconds"].mean()) if len(res) else None,"p50_seconds":float(res["elapsed_seconds"].median()) if len(res) else None,
         "decisions":res["decision"].value_counts().to_dict() if len(res) else {},"2025_opened":False}
(OUT/"REAL_EVIDENCE_REASONING_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print("SUMMARY="+json.dumps(summary,ensure_ascii=False),flush=True)
if summary["status"]!="PASS": raise SystemExit(2)
