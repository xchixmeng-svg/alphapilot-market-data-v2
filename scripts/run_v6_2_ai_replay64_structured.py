#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,os,time,urllib.request
from pathlib import Path

MODEL=os.environ.get("V62_LOCAL_MODEL","qwen2.5:7b")
OUT=Path("out"); OUT.mkdir(exist_ok=True)
DECISIONS=["REJECT","WATCH","CANDIDATE","HIGH_CONVICTION"]
ENTRY=["NOW","WAIT_FOR_PULLBACK","WAIT_FOR_CONFIRMATION","NOT_ACTIONABLE"]
TRIGGERS=["PRICE_STRUCTURE_BREAK","THESIS_INVALIDATION","CATALYST_FAILURE","VALUATION_EXPECTATION_BREAK","MULTI_EVIDENCE_FAILURE","UNAVAILABLE"]

SYSTEM="""You are AlphaPilot V6.2's AI decision layer.
Evaluate exactly one Taiwan stock using only the supplied point-in-time evidence and immutable numerical forecasts.
Do not rank stocks. Do not use TopK, quotas, or fixed probability thresholds. Numerical forecasts are evidence, not admission rules.
Never invent unavailable evidence. Missing evidence may be mentioned only as a limitation/counter-evidence, never as positive support.
For CANDIDATE/HIGH_CONVICTION, give a stock-specific entry zone and a pre-entry AI Failure Exit Price grounded in supplied evidence.
For WATCH/REJECT, entry must be NOT_ACTIONABLE with null prices, and failure_exit must be unavailable with null exit price.
Return only JSON matching the provided JSON schema."""

def finite(v):
    return isinstance(v,(int,float)) and math.isfinite(float(v))

def schema_for(pkt):
    avail=list(pkt["evidence"]["available_evidence_ids"])
    counter=avail+["missing:"+x for x in pkt["evidence"]["missing_evidence_ids"]]
    s={
      "type":"object","additionalProperties":False,
      "required":["decision","evidence_quality","hypothesis_type","hypothesis","primary_evidence_ids","secondary_evidence_ids","counter_evidence_ids","bull_thesis","bear_thesis","invalidation","decision_reason","entry","failure_exit"],
      "properties":{
        "decision":{"type":"string","enum":DECISIONS},
        "evidence_quality":{"type":"string","enum":["STRONG","MODERATE","WEAK"]},
        "hypothesis_type":{"type":"string"},
        "hypothesis":{"type":"string"},
        "primary_evidence_ids":{"type":"array","minItems":1,"uniqueItems":True,"items":{"type":"string","enum":avail}},
        "secondary_evidence_ids":{"type":"array","uniqueItems":True,"items":{"type":"string","enum":avail}},
        "counter_evidence_ids":{"type":"array","uniqueItems":True,"items":{"type":"string","enum":counter}},
        "bull_thesis":{"type":"string"},
        "bear_thesis":{"type":"string"},
        "invalidation":{"type":"string"},
        "decision_reason":{"type":"string"},
        "entry":{"type":"object","additionalProperties":False,"required":["status","ideal_low","ideal_high"],
          "properties":{"status":{"type":"string","enum":ENTRY},"ideal_low":{"type":["number","null"]},"ideal_high":{"type":["number","null"]}}},
        "failure_exit":{"type":"object","additionalProperties":False,"required":["exit_price","reason","trigger_type"],
          "properties":{"exit_price":{"type":["number","null"]},"reason":{"type":["string","null"]},"trigger_type":{"type":"string","enum":TRIGGERS}}}
      }
    }
    return s

def semantic_validate(pkt,o):
    for k in ["hypothesis_type","hypothesis","bull_thesis","bear_thesis","invalidation","decision_reason"]:
        if not isinstance(o.get(k),str) or not o[k].strip():
            raise ValueError(f"empty_{k}")
    d=o["decision"]; e=o["entry"]; f=o["failure_exit"]
    current=float(pkt["evidence"]["price_volume_structure"]["current_price"])
    actionable=d in {"CANDIDATE","HIGH_CONVICTION"}
    if actionable:
        if e["status"]=="NOT_ACTIONABLE": raise ValueError("actionable_entry_not_actionable")
        if not finite(e["ideal_low"]) or not finite(e["ideal_high"]): raise ValueError("actionable_entry_missing")
        if not (0<float(e["ideal_low"])<=float(e["ideal_high"])): raise ValueError("actionable_entry_order")
        if not (0.65*current<=float(e["ideal_low"])<=1.30*current and 0.65*current<=float(e["ideal_high"])<=1.30*current):
            raise ValueError("actionable_entry_implausible")
        if not finite(f["exit_price"]) or not (0<float(f["exit_price"])<float(e["ideal_high"])):
            raise ValueError("actionable_failure_exit_invalid")
        if f["trigger_type"]=="UNAVAILABLE" or not isinstance(f["reason"],str) or not f["reason"].strip():
            raise ValueError("actionable_failure_exit_missing_reason")
    else:
        if e["status"]!="NOT_ACTIONABLE" or e["ideal_low"] is not None or e["ideal_high"] is not None:
            raise ValueError("nonactionable_entry_must_be_null")
        if f["exit_price"] is not None or f["trigger_type"]!="UNAVAILABLE":
            raise ValueError("nonactionable_failure_exit_must_be_unavailable")
    # Hard evidence safety is structural: schema enums prevent missing families from entering support IDs.
    support=set(o["primary_evidence_ids"])|set(o["secondary_evidence_ids"])
    missing=set(pkt["evidence"]["missing_evidence_ids"])
    if support & missing:
        raise ValueError("missing_evidence_used_as_support")

def call(pkt,repair_error=None):
    sch=schema_for(pkt)
    user={
      "case":pkt,
      "contract_notes":[
        "Primary/secondary evidence IDs must come from available evidence only.",
        "Counter-evidence may cite available evidence or missing:<family>.",
        "WATCH/REJECT => NOT_ACTIONABLE entry with null prices and UNAVAILABLE failure exit.",
        "CANDIDATE/HIGH_CONVICTION => actionable entry zone and stock-specific failure exit price."
      ]
    }
    if repair_error:
        user["repair_only"]=f"Your prior answer failed semantic validation: {repair_error}. Keep the substantive thesis but repair only this contract violation."
    body=json.dumps({
      "model":MODEL,
      "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(user,ensure_ascii=False,separators=(",",":"))}],
      "stream":False,
      "format":sch,
      "options":{"temperature":0,"num_predict":900}
    },ensure_ascii=False).encode()
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as resp:
        raw=json.loads(resp.read().decode())
    obj=json.loads(raw["message"]["content"])
    semantic_validate(pkt,obj)
    return obj,time.time()-t0

def load_all():
    cases=[]
    for shard in range(4):
        p=Path(f"prepared/cases_shard_{shard}.jsonl")
        for line in p.read_text().splitlines():
            if line.strip(): cases.append(json.loads(line))
    return sorted(cases,key=lambda x:x["case_id"])

def choose_canary(cases):
    # Contract canary only: cover low/mid/high numerical regimes without looking at outcomes.
    s=sorted(cases,key=lambda x:float(x["numerical_reference_read_only"]["p_hit10_h120"]))
    idx=[0,len(s)//3,(2*len(s))//3,len(s)-1]
    return [s[i] for i in idx]

def run_cases(cases,label):
    rows=[]; errors=[]
    checkpoint=OUT/f"{label}_responses.jsonl"
    errfile=OUT/f"{label}_errors.jsonl"
    for pkt in cases:
        obj=None; err=None; elapsed=None
        for attempt in (1,2):
            try:
                obj,elapsed=call(pkt,err)
                break
            except Exception as e:
                err=repr(e)
        if obj is None:
            rec={"case_id":pkt["case_id"],"date":pkt["decision_date"],"code":pkt["code"],"error":err}
            errors.append(rec)
            with errfile.open("a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            print("ERROR "+json.dumps(rec,ensure_ascii=False),flush=True)
            continue
        rec={"case_id":pkt["case_id"],"date":pkt["decision_date"],"code":pkt["code"],
             "elapsed_seconds":elapsed,"p_hit10_h120":pkt["numerical_reference_read_only"]["p_hit10_h120"],**obj}
        rows.append(rec)
        with checkpoint.open("a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        print("OK "+json.dumps({"case_id":pkt["case_id"],"decision":obj["decision"],"seconds":round(elapsed,1)},ensure_ascii=False),flush=True)
    summary={"label":label,"requested":len(cases),"completed":len(rows),"errors":len(errors),"model":MODEL,"api_cost_usd":0}
    (OUT/f"{label}_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("SUMMARY="+json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["canary","shard"],required=True)
    ap.add_argument("--shard",type=int)
    a=ap.parse_args()
    cases=load_all()
    if a.mode=="canary":
        chosen=choose_canary(cases)
        s=run_cases(chosen,"canary")
        if s["errors"] or s["completed"]!=4: raise SystemExit(2)
    else:
        if a.shard not in [0,1,2,3]: raise SystemExit("--shard 0..3 required")
        chosen=[x for x in cases if x["case_id"]%4==a.shard]
        s=run_cases(chosen,f"shard_{a.shard}")
        if s["errors"] or s["completed"]!=16: raise SystemExit(2)

if __name__=="__main__": main()
