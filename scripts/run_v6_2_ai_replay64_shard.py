#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,os,re,time,urllib.request
from pathlib import Path

MODEL=os.environ.get("V62_LOCAL_MODEL","qwen2.5:7b")
OUT=Path("out"); OUT.mkdir(exist_ok=True)
DECISIONS={"REJECT","WATCH","CANDIDATE","HIGH_CONVICTION"}
ENTRY={"NOW","WAIT_FOR_PULLBACK","WAIT_FOR_CONFIRMATION","NOT_ACTIONABLE"}
TRIGGERS={"PRICE_STRUCTURE_BREAK","THESIS_INVALIDATION","CATALYST_FAILURE","VALUATION_EXPECTATION_BREAK","MULTI_EVIDENCE_FAILURE","UNAVAILABLE"}

SYSTEM="""You are the AlphaPilot V6.2 Taiwan-stock AI decision layer.
You receive one stock at one historical decision date with point-in-time evidence and immutable calibrated numerical forecasts.
Judge this stock independently. Do not rank it against other stocks and do not use TopK, quotas, or fixed probability thresholds.
The question is whether this stock is forming a genuine >=10% opportunity, why now, whether the entry is attractive, and what would prove the thesis wrong.
Different stocks may rely on different evidence families. Numerical forecasts are evidence, not admission rules.
Never invent evidence. Evidence listed as missing is unavailable. Revenue is not EPS revision. Price strength is not proof of news, industry pricing, inventory, or supply-demand.
Do not echo, rewrite, or alter numerical probabilities in your response.
For CANDIDATE/HIGH_CONVICTION, provide a stock-specific entry zone and a pre-entry AI Failure Exit Price based on actual evidence; never use a universal fixed stop percentage.
For WATCH/REJECT, failure_exit.exit_price must be null and trigger_type UNAVAILABLE.
Return exactly one JSON object and no markdown."""

def finite(v):
    return isinstance(v,(int,float)) and math.isfinite(float(v))

def _canon(s):
    return re.sub(r"[^a-z0-9]+","",str(s).lower())

def normalize_ids(pkt,o):
    """Normalize harmless LLM spelling/case/punctuation variants only.
    No fuzzy semantic matching: an ID must canonicalize uniquely to an exact supplied ID.
    This preserves fail-closed no-hallucination semantics while avoiding false failures such
    as `price-volume-structure` vs `price_volume_structure`.
    """
    avail=list(pkt["evidence"]["available_evidence_ids"])
    missing=list(pkt["evidence"]["missing_evidence_ids"])
    canonical={}
    for x in avail+missing:
        canonical.setdefault(_canon(x),[]).append(x)
    def one(x,allow_missing=False):
        pref=str(x).lower().startswith("missing:")
        raw=str(x).split(":",1)[1] if pref else str(x)
        pool=missing if pref else (avail+missing if allow_missing else avail)
        hits=[v for v in canonical.get(_canon(raw),[]) if v in pool]
        if len(hits)==1:
            v=hits[0]
            return "missing:"+v if (pref or (allow_missing and v in missing)) else v
        return x
    for k in ["primary_evidence_ids","secondary_evidence_ids"]:
        if isinstance(o.get(k),list): o[k]=[one(x,False) for x in o[k]]
    if isinstance(o.get("counter_evidence_ids"),list):
        o["counter_evidence_ids"]=[one(x,True) for x in o["counter_evidence_ids"]]

def validate(pkt,o):
    normalize_ids(pkt,o)
    keys={"decision","evidence_quality","hypothesis_type","hypothesis","primary_evidence_ids","secondary_evidence_ids",
          "counter_evidence_ids","bull_thesis","bear_thesis","invalidation","decision_reason","entry","failure_exit"}
    if set(o)!=keys: raise ValueError(f"schema keys {sorted(o)}")
    if o["decision"] not in DECISIONS: raise ValueError("bad decision")
    if o["evidence_quality"] not in {"STRONG","MODERATE","WEAK"}: raise ValueError("bad evidence_quality")
    for k in ["hypothesis_type","hypothesis","bull_thesis","bear_thesis","invalidation","decision_reason"]:
        if not isinstance(o[k],str) or not o[k].strip(): raise ValueError(f"empty {k}")
    avail=set(pkt["evidence"]["available_evidence_ids"])
    missing_raw=set(pkt["evidence"]["missing_evidence_ids"])
    missing_prefixed={"missing:"+x for x in missing_raw}
    for k in ["primary_evidence_ids","secondary_evidence_ids"]:
        if not isinstance(o[k],list) or any(x not in avail for x in o[k]): raise ValueError(f"unsupported {k}")
    if not o["primary_evidence_ids"]: raise ValueError("empty primary evidence")
    if not isinstance(o["counter_evidence_ids"],list): raise ValueError("counter_evidence_ids must be list")
    allowed_counter=avail|missing_raw|missing_prefixed
    bad_counter=[x for x in o["counter_evidence_ids"] if x not in allowed_counter]
    if bad_counter: raise ValueError("unsupported counter evidence: "+",".join(map(str,bad_counter)))
    o["counter_evidence_ids"]=[("missing:"+x if x in missing_raw else x) for x in o["counter_evidence_ids"]]
    e=o["entry"]
    if set(e)!={"status","ideal_low","ideal_high"} or e["status"] not in ENTRY: raise ValueError("bad entry")
    f=o["failure_exit"]
    if set(f)!={"exit_price","reason","trigger_type"} or f["trigger_type"] not in TRIGGERS: raise ValueError("bad failure_exit")
    actionable=o["decision"] in {"CANDIDATE","HIGH_CONVICTION"}
    current=float(pkt["evidence"]["price_volume_structure"]["current_price"])
    if actionable:
        if e["status"]=="NOT_ACTIONABLE": raise ValueError("actionable decision has NOT_ACTIONABLE entry")
        if not finite(e["ideal_low"]) or not finite(e["ideal_high"]) or not (0<e["ideal_low"]<=e["ideal_high"]): raise ValueError("bad entry zone")
        if not (0.65*current <= e["ideal_low"] <= 1.30*current and 0.65*current <= e["ideal_high"] <= 1.30*current): raise ValueError("entry zone implausibly far from current")
        if not finite(f["exit_price"]) or not (0<float(f["exit_price"])<float(e["ideal_high"])): raise ValueError("bad failure exit price")
        if f["trigger_type"]=="UNAVAILABLE" or not isinstance(f["reason"],str) or not f["reason"].strip(): raise ValueError("missing failure exit reasoning")
    else:
        if f["exit_price"] is not None or f["trigger_type"]!="UNAVAILABLE": raise ValueError("non-actionable must not fabricate failure exit")
    txt=json.dumps(o,ensure_ascii=False).lower()
    forbidden={"eps_revisions":["eps revision","eps revisions","eps上修","eps 預估上修"],"analyst_consensus":["analyst consensus","consensus estimate","分析師共識"],"industry_pricing":["industry pricing","spot price","產業報價","現貨價"],"inventory_supply_demand":["inventory destocking","supply shortage","供不應求","庫存去化"],"broad_news_semantics":["news sentiment","媒體情緒"]}
    miss=set(pkt["evidence"]["missing_evidence_ids"])
    bad=[fam for fam,terms in forbidden.items() if fam in miss and any(t in txt for t in terms)]
    if bad: raise ValueError("hallucinated missing evidence "+",".join(bad))

def call(pkt,repair=None):
    response_schema={"decision":"REJECT|WATCH|CANDIDATE|HIGH_CONVICTION","evidence_quality":"STRONG|MODERATE|WEAK","hypothesis_type":"string","hypothesis":"string","primary_evidence_ids":"list of supplied available evidence IDs only","secondary_evidence_ids":"list of supplied available evidence IDs only","counter_evidence_ids":"list of supplied available IDs or missing:<family>","bull_thesis":"string","bear_thesis":"string","invalidation":"string","decision_reason":"string","entry":{"status":"NOW|WAIT_FOR_PULLBACK|WAIT_FOR_CONFIRMATION|NOT_ACTIONABLE","ideal_low":"number|null","ideal_high":"number|null"},"failure_exit":{"exit_price":"number|null","reason":"string|null","trigger_type":"PRICE_STRUCTURE_BREAK|THESIS_INVALIDATION|CATALYST_FAILURE|VALUATION_EXPECTATION_BREAK|MULTI_EVIDENCE_FAILURE|UNAVAILABLE"}}
    user={"case":pkt,"response_schema":response_schema,"allowed_primary_secondary_evidence_ids":pkt["evidence"]["available_evidence_ids"],"allowed_counter_evidence_ids":pkt["evidence"]["available_evidence_ids"]+pkt["evidence"]["missing_evidence_ids"]+["missing:"+x for x in pkt["evidence"]["missing_evidence_ids"]]}
    if repair:
        user["previous_schema_error"]=repair
        user["repair_instruction"]="Return the same analysis but use ONLY exact IDs from the allowed ID arrays. Copy IDs verbatim; do not invent descriptive evidence IDs."
    body=json.dumps({"model":MODEL,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(user,ensure_ascii=False,separators=(",",":"))}],"stream":False,"format":"json","options":{"temperature":0,"num_predict":950}}).encode()
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as resp: raw=json.loads(resp.read().decode())
    obj=json.loads(raw["message"]["content"].strip()); validate(pkt,obj)
    return obj,time.time()-t0

def main(shard):
    cases=[json.loads(x) for x in Path(f"prepared/cases_shard_{shard}.jsonl").read_text().splitlines() if x.strip()]
    rows=[]; errors=[]
    for pkt in cases:
        err=None; obj=None; elapsed=None
        for attempt in [1,2]:
            try: obj,elapsed=call(pkt,err); break
            except Exception as e: err=repr(e)
        if obj is None:
            errors.append({"case_id":pkt["case_id"],"date":pkt["decision_date"],"code":pkt["code"],"error":err}); print("ERROR",errors[-1],flush=True); continue
        rows.append({"case_id":pkt["case_id"],"date":pkt["decision_date"],"code":pkt["code"],"elapsed_seconds":elapsed,"p_hit10_h120":pkt["numerical_reference_read_only"]["p_hit10_h120"],**obj})
        print(json.dumps({"case_id":pkt["case_id"],"code":pkt["code"],"decision":obj["decision"],"seconds":round(elapsed,2)},ensure_ascii=False),flush=True)
    (OUT/f"responses_shard_{shard}.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
    (OUT/f"errors_shard_{shard}.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2)+"\n")
    summary={"shard":shard,"requested":len(cases),"completed":len(rows),"errors":len(errors),"model":MODEL,"api_cost_usd":0}
    (OUT/f"summary_shard_{shard}.json").write_text(json.dumps(summary,indent=2)+"\n"); print("SUMMARY="+json.dumps(summary),flush=True)
    if errors: raise SystemExit(2)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",type=int,required=True); main(ap.parse_args().shard)
