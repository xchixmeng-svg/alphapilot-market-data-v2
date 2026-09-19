#!/usr/bin/env python3
from __future__ import annotations
import json,math,os,time,urllib.request
from pathlib import Path

MODEL=os.environ.get("V62_LOCAL_MODEL","qwen2.5:7b")
CASE_ID=int(os.environ.get("V62_CASE_ID","4"))
OUT=Path("out"); OUT.mkdir(exist_ok=True)

DECISIONS=["REJECT","WATCH","CANDIDATE","HIGH_CONVICTION"]
ACTIONABLE={"CANDIDATE","HIGH_CONVICTION"}

SYSTEM_DECISION="""You are AlphaPilot V6.2's AI decision layer.
Evaluate exactly one Taiwan stock using only supplied point-in-time evidence and immutable numerical forecasts.
Do not rank stocks. Do not use TopK, quotas, or fixed probability thresholds.
Never invent unavailable evidence. Missing evidence may be mentioned only as a limitation/counter-evidence, never as positive support.
Decide whether the stock is forming a genuine >=10% opportunity. Return only JSON matching the schema."""

SYSTEM_ACTION="""You are AlphaPilot V6.2's execution-planning layer.
The prior AI decision is already actionable (CANDIDATE or HIGH_CONVICTION). Do not change that decision.
Using only the same supplied evidence, produce a stock-specific entry plan and AI Failure Exit.
Do not use a universal fixed-percent stop. Return only JSON matching the schema."""

def load_case():
    for p in sorted(Path("prepared").glob("cases_shard_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                x=json.loads(line)
                if int(x["case_id"])==CASE_ID:
                    return x
    raise RuntimeError(f"case {CASE_ID} not found")

def call(system,user,schema):
    body=json.dumps({
      "model":MODEL,
      "messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(user,ensure_ascii=False,separators=(",",":"))}],
      "stream":False,
      "format":schema,
      "options":{"temperature":0,"num_predict":900}
    },ensure_ascii=False).encode()
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as resp:
        raw=json.loads(resp.read().decode())
    return json.loads(raw["message"]["content"]),time.time()-t0

def decision_schema(pkt):
    avail=list(pkt["evidence"]["available_evidence_ids"])
    counter=avail+["missing:"+x for x in pkt["evidence"]["missing_evidence_ids"]]
    return {
      "type":"object","additionalProperties":False,
      "required":["decision","evidence_quality","hypothesis_type","hypothesis","primary_evidence_ids","secondary_evidence_ids","counter_evidence_ids","bull_thesis","bear_thesis","invalidation","decision_reason"],
      "properties":{
        "decision":{"type":"string","enum":DECISIONS},
        "evidence_quality":{"type":"string","enum":["STRONG","MODERATE","WEAK"]},
        "hypothesis_type":{"type":"string","minLength":1},
        "hypothesis":{"type":"string","minLength":1},
        "primary_evidence_ids":{"type":"array","minItems":1,"uniqueItems":True,"items":{"type":"string","enum":avail}},
        "secondary_evidence_ids":{"type":"array","uniqueItems":True,"items":{"type":"string","enum":avail}},
        "counter_evidence_ids":{"type":"array","uniqueItems":True,"items":{"type":"string","enum":counter}},
        "bull_thesis":{"type":"string","minLength":1},
        "bear_thesis":{"type":"string","minLength":1},
        "invalidation":{"type":"string","minLength":1},
        "decision_reason":{"type":"string","minLength":1}
      }
    }

def action_schema(pkt):
    current=float(pkt["evidence"]["price_volume_structure"]["current_price"])
    lo=max(0.01,current*0.65); hi=current*1.30
    return {
      "type":"object","additionalProperties":False,
      "required":["entry","failure_exit"],
      "properties":{
        "entry":{
          "type":"object","additionalProperties":False,
          "required":["status","ideal_low","ideal_high"],
          "properties":{
            "status":{"type":"string","enum":["NOW","WAIT_FOR_PULLBACK","WAIT_FOR_CONFIRMATION"]},
            "ideal_low":{"type":"number","minimum":lo,"maximum":hi},
            "ideal_high":{"type":"number","minimum":lo,"maximum":hi}
          }
        },
        "failure_exit":{
          "type":"object","additionalProperties":False,
          "required":["exit_price","reason","trigger_type"],
          "properties":{
            "exit_price":{"type":"number","exclusiveMinimum":0,"maximum":hi},
            "reason":{"type":"string","minLength":1},
            "trigger_type":{"type":"string","enum":["PRICE_STRUCTURE_BREAK","THESIS_INVALIDATION","CATALYST_FAILURE","VALUATION_EXPECTATION_BREAK","MULTI_EVIDENCE_FAILURE"]}
          }
        }
      }
    }

def validate_action(pkt,a):
    e=a["entry"]; f=a["failure_exit"]
    if not (0<float(e["ideal_low"])<=float(e["ideal_high"])):
        raise ValueError("entry_order")
    if not (0<float(f["exit_price"])<float(e["ideal_high"])):
        raise ValueError("failure_exit_not_below_entry_high")

def main():
    pkt=load_case()
    d,dt=call(SYSTEM_DECISION,{"case":pkt},decision_schema(pkt))
    result={"case_id":CASE_ID,"date":pkt["decision_date"],"code":pkt["code"],"decision_seconds":dt,**d}
    if d["decision"] in ACTIONABLE:
        a,at=call(SYSTEM_ACTION,{"case":pkt,"locked_decision":d},action_schema(pkt))
        validate_action(pkt,a)
        result.update(a); result["action_seconds"]=at
    else:
        result["entry"]={"status":"NOT_ACTIONABLE","ideal_low":None,"ideal_high":None}
        result["failure_exit"]={"exit_price":None,"reason":None,"trigger_type":"UNAVAILABLE"}
        result["action_seconds"]=0.0
    (OUT/"SINGLE_CASE_RESULT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    summary={"status":"PASS","case_id":CASE_ID,"decision":result["decision"],"decision_seconds":round(dt,2),
             "action_seconds":round(result["action_seconds"],2),"two_stage":True,"api_cost_usd":0}
    (OUT/"SINGLE_CASE_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
