#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[1]
schema=json.loads((ROOT/"research"/"V6_2_LLM_REASONING_SCHEMA.json").read_text(encoding="utf-8"))
v=Draft202012Validator(schema)

base={
  "analysis_status":"INSUFFICIENT_INPUT",
  "missing_required_inputs":["numerical_outputs.barrier_probabilities"],
  "hypothesis_type":None,
  "hypothesis":None,
  "primary_evidence_ids":[],
  "secondary_evidence_ids":[],
  "counter_evidence_ids":[],
  "bull_thesis":None,
  "bear_thesis":None,
  "invalidation":None,
  "evidence_quality":"NOT_ASSESSABLE",
  "decision":None,
  "decision_reason":None
}
errs=list(v.iter_errors(base))
if errs:
    raise RuntimeError("valid abstention rejected: "+str(errs[0]))

bad=dict(base)
bad["decision"]="CANDIDATE"
errs=list(v.iter_errors(bad))
if not errs:
    raise RuntimeError("invalid forced CANDIDATE was accepted for INSUFFICIENT_INPUT")

complete=dict(base)
complete.update({
  "analysis_status":"COMPLETE",
  "missing_required_inputs":[],
  "hypothesis_type":"VALUATION_RERATING",
  "hypothesis":"Structured test hypothesis.",
  "bull_thesis":"Structured bull evidence.",
  "bear_thesis":"Structured counter evidence.",
  "invalidation":"Structured invalidation.",
  "evidence_quality":"MODERATE",
  "decision":"WATCH",
  "decision_reason":"Structured test decision."
})
errs=list(v.iter_errors(complete))
if errs:
    raise RuntimeError("valid complete decision rejected: "+str(errs[0]))

invented=dict(complete)
invented["ai_confidence_0_100"]=84
errs=list(v.iter_errors(invented))
if not errs:
    raise RuntimeError("uncalibrated confidence backdoor field was accepted")

print(json.dumps({
  "status":"PASS",
  "abstention_allowed":True,
  "forced_candidate_when_incomplete_rejected":True,
  "complete_four_level_decision_allowed":True,
  "uncalibrated_confidence_field_rejected":True
},ensure_ascii=False))
