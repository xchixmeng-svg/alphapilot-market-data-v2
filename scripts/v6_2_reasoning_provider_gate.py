#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

DECISIONS={"REJECT","WATCH","CANDIDATE","HIGH_CONVICTION","INSUFFICIENT_INPUT"}
NUMERIC_PREFIXES=("p_up","probability_","expected_","base_","bull_","mfe_","mae_")


def sha256_obj(x):
    b=json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(b).hexdigest()


def validate(evidence, numerical, response):
    # LLM is not allowed to run when required upstream inputs are absent.
    missing=[]
    for name,obj in (("evidence",evidence),("numerical",numerical)):
        if not isinstance(obj,dict) or not obj:
            missing.append(name)
    if missing:
        if response.get("decision")!="INSUFFICIENT_INPUT":
            raise RuntimeError("missing upstream input must force INSUFFICIENT_INPUT")
        return {"status":"PASS","decision":"INSUFFICIENT_INPUT","missing":missing}

    decision=response.get("decision")
    if decision not in DECISIONS:
        raise RuntimeError(f"invalid decision {decision!r}")

    # Numerical engine owns all probabilities/ranges. Response may echo them only exactly.
    tampered=[]
    for k,v in response.items():
        if k in numerical or k.startswith(NUMERIC_PREFIXES):
            if k not in numerical or v!=numerical[k]:
                tampered.append(k)
    if tampered:
        raise RuntimeError("LLM numerical mutation/invention: "+",".join(sorted(tampered)))

    # No rank/TopK field is allowed to participate in admission.
    forbidden=[k for k in response if "rank" in k.lower() or "topk" in k.lower() or "top_k" in k.lower()]
    if forbidden:
        raise RuntimeError("rank/TopK admission field forbidden: "+",".join(sorted(forbidden)))

    # Reasoning must identify evidence, counter-evidence and invalidation for an admitted name.
    if decision in {"CANDIDATE","HIGH_CONVICTION"}:
        required=("why_now","evidence_used","counter_evidence","invalidation")
        absent=[k for k in required if not response.get(k)]
        if absent:
            raise RuntimeError("admitted candidate missing reasoning fields: "+",".join(absent))

    return {
      "status":"PASS","decision":decision,
      "evidence_sha256":sha256_obj(evidence),
      "numerical_sha256":sha256_obj(numerical),
      "response_sha256":sha256_obj(response),
      "numerical_fields_checked":len(numerical),
      "rank_admission_fields":0
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",required=True); ap.add_argument("--numerical",required=True)
    ap.add_argument("--response",required=True); ap.add_argument("--out",required=True)
    ns=ap.parse_args()
    load=lambda p: json.loads(Path(p).read_text())
    audit=validate(load(ns.evidence),load(ns.numerical),load(ns.response))
    Path(ns.out).write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    print(json.dumps(audit,sort_keys=True))

if __name__=="__main__": main()
