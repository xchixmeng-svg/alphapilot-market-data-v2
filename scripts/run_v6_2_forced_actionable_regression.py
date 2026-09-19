#!/usr/bin/env python3
from __future__ import annotations
import json,math,os,time,urllib.request
from pathlib import Path

MODEL=os.environ.get("V62_LOCAL_MODEL","qwen2.5:7b")
CASE_ID=int(os.environ.get("V62_CASE_ID","4"))
OUT=Path("out"); OUT.mkdir(exist_ok=True)

def load_case():
    for p in Path("prepared").glob("cases_shard_*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                x=json.loads(line)
                if int(x["case_id"])==CASE_ID: return x
    raise RuntimeError("case missing")

def call(pkt):
    current=float(pkt["evidence"]["price_volume_structure"]["current_price"])
    schema={"type":"object","additionalProperties":False,"required":["entry","failure_exit"],"properties":{
      "entry":{"type":"object","additionalProperties":False,"required":["status","ideal_low","ideal_high"],"properties":{
        "status":{"type":"string","enum":["NOW","WAIT_FOR_PULLBACK","WAIT_FOR_CONFIRMATION"]},
        "ideal_low":{"type":"number","minimum":max(.01,current*.65),"maximum":current*1.30},
        "ideal_high":{"type":"number","minimum":max(.01,current*.65),"maximum":current*1.30}}},
      "failure_exit":{"type":"object","additionalProperties":False,"required":["exit_price","reason","trigger_type"],"properties":{
        "exit_price":{"type":"number","exclusiveMinimum":0,"maximum":current*1.30},
        "reason":{"type":"string","minLength":1},
        "trigger_type":{"type":"string","enum":["PRICE_STRUCTURE_BREAK","THESIS_INVALIDATION","CATALYST_FAILURE","VALUATION_EXPECTATION_BREAK","MULTI_EVIDENCE_FAILURE"]}}}
    }}
    system="""This is a V6.2 engineering regression of the actionable branch only.
The upstream decision is deliberately LOCKED to CANDIDATE for this regression. Do not reassess or downgrade it.
Using only supplied point-in-time evidence, produce a coherent entry plan and a stock-specific AI Failure Exit.
This forced CANDIDATE is NOT a scientific label and must never be used as model-performance evidence.
Return only schema-valid JSON."""
    body=json.dumps({"model":MODEL,"messages":[{"role":"system","content":system},{"role":"user","content":json.dumps({"case":pkt,"locked_decision":"CANDIDATE"},ensure_ascii=False,separators=(",",":"))}],"stream":False,"format":schema,"options":{"temperature":0,"num_predict":500}},ensure_ascii=False).encode()
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=body,headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(req,timeout=900) as r: raw=json.loads(r.read().decode())
    o=json.loads(raw["message"]["content"])
    e=o["entry"]; f=o["failure_exit"]
    if not (0<float(e["ideal_low"])<=float(e["ideal_high"])): raise ValueError("entry_order")
    if not (0<float(f["exit_price"])<float(e["ideal_high"])): raise ValueError("failure_exit_not_below_entry")
    return o,time.time()-t

pkt=load_case()
o,sec=call(pkt)
res={"status":"PASS","case_id":CASE_ID,"forced_decision_for_engineering_only":"CANDIDATE","seconds":sec,**o}
(OUT/"FORCED_ACTIONABLE_BRANCH_RESULT.json").write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(res,ensure_ascii=False),flush=True)
