#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, urllib.request
from pathlib import Path

MODEL=os.environ.get("V62_LOCAL_MODEL","qwen2.5:3b")

evidence={
  "decision_date":"2024-12-31","code":"2330",
  "price_structure":{"status":"AVAILABLE","return_20_sessions_pct":8.0,"close_position_in_60_session_range_pct":83.0},
  "institutional_flow":{"status":"AVAILABLE","foreign_net_shares_over_volume_mean_5_sessions_pct":2.1},
  "revenue_earnings":{"status":"PARTIAL","latest_revenue_yoy_pct":18.4,"missing":["earnings","margins","eps"]},
  "eps_revisions":{"status":"UNAVAILABLE","missing_reason":"No point-in-time analyst EPS revision series."},
  "valuation":{"status":"AVAILABLE","pe":24.1},
  "industry_pricing_supply_demand":{"status":"UNAVAILABLE","missing_reason":"No admitted point-in-time industry pricing/supply-demand series."}
}
numerical={"p_up10_before_down5_h20":0.61,"expected_mae_low_pct":-0.04,"expected_mae_high_pct":-0.01}

system="""You are the V6.2 Taiwan-stock opportunity reasoning layer. Judge THIS stock independently; never rank stocks.
Return exactly one JSON object and nothing else.
Evidence rules: never invent unavailable evidence; revenue growth is NOT analyst EPS revision evidence. Explicitly acknowledge unavailable EPS revision and industry pricing evidence when relevant.
Decision rules: decision is REJECT, WATCH, CANDIDATE, HIGH_CONVICTION, or INSUFFICIENT_INPUT. Do not use TopK, ranks, quotas, or forced-candidate logic.
For every non-INSUFFICIENT decision return non-empty why_now, evidence_used list, counter_evidence list, and concrete invalidation.
The numerical object is immutable machine output. DO NOT restate, copy, round, transform, explain as a different number, or generate numerical-engine values. Your response schema intentionally contains NO probability or MAE fields. A deterministic validator attaches the original immutable numerical object after validating your reasoning; that is transport integrity, not AI inference.
Do not turn a 20-session return into a yearly return. This is a capability benchmark, not a trading recommendation."""

payload={"task":"Assess whether the current opportunity is sufficiently evidenced.","evidence":evidence,"numerical_reference_read_only":numerical,
         "response_schema":{"decision":"enum","why_now":"string","evidence_used":"list[string]","counter_evidence":"list[string]","invalidation":"string"}}
body=json.dumps({"model":MODEL,"messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,ensure_ascii=False,separators=(",",":"))}],"stream":False,"format":"json","options":{"temperature":0,"num_predict":420}}).encode()
req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=body,headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req,timeout=900) as r: out=json.loads(r.read().decode())
elapsed=time.time()-t0
raw=out["message"]["content"].strip()
print("LOCAL_MODEL="+MODEL); print(f"ELAPSED_SECONDS={elapsed:.3f}"); print("RAW_RESPONSE="+raw.replace("\n"," ")[:3000])
obj=json.loads(raw)
allowed={"decision","why_now","evidence_used","counter_evidence","invalidation"}
extra=sorted(set(obj)-allowed)
if extra: raise SystemExit("SEMANTIC_FAIL unexpected_or_numerical_keys="+",".join(extra))
missing=sorted(allowed-set(obj))
if missing: raise SystemExit("SEMANTIC_FAIL missing_required_keys="+",".join(missing))
if obj["decision"] not in {"REJECT","WATCH","CANDIDATE","HIGH_CONVICTION","INSUFFICIENT_INPUT"}: raise SystemExit("SEMANTIC_FAIL invalid_decision")
if obj["decision"]!="INSUFFICIENT_INPUT":
    if not isinstance(obj["why_now"],str) or not obj["why_now"].strip(): raise SystemExit("SEMANTIC_FAIL empty_why_now")
    for k in ("evidence_used","counter_evidence"):
        if not isinstance(obj[k],list) or not obj[k]: raise SystemExit("SEMANTIC_FAIL empty_"+k)
    if not isinstance(obj["invalidation"],str) or not obj["invalidation"].strip(): raise SystemExit("SEMANTIC_FAIL empty_invalidation")

txt=json.dumps(obj,ensure_ascii=False).lower()
bad_eps=["eps revisions are rising","eps revisions are improving","eps revision is rising","eps revision is improving","eps revisions upgraded","eps revision upgraded","analyst eps estimates are rising","analyst estimates are rising"]
if any(s in txt for s in bad_eps): raise SystemExit("SEMANTIC_FAIL hallucinated_eps_revision")
bad_time=["over the last year","past year","yearly return","one-year return"]
if any(s in txt for s in bad_time): raise SystemExit("SEMANTIC_FAIL misread_20_session_return")
# Fail closed if unavailable families are positively asserted as evidence.
used=" ".join(str(x).lower() for x in obj.get("evidence_used",[]))
if "eps_revision" in used or "eps revision" in used: raise SystemExit("SEMANTIC_FAIL unavailable_eps_used_as_positive_evidence")
if "industry_pricing" in used or "industry pricing" in used or "supply_demand" in used: raise SystemExit("SEMANTIC_FAIL unavailable_industry_data_used_as_positive_evidence")

# Immutable machine fields are attached only after AI semantic validation.
final={**obj,"numerical":numerical}
Path("out").mkdir(exist_ok=True)
Path("out/evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
Path("out/numerical.json").write_text(json.dumps(numerical,ensure_ascii=False,indent=2))
Path("out/reasoning_response.json").write_text(json.dumps(obj,ensure_ascii=False,indent=2))
Path("out/final_validated_packet.json").write_text(json.dumps(final,ensure_ascii=False,indent=2))
Path("out/provider_meta.json").write_text(json.dumps({"provider":"local_ollama_open_weights","model":MODEL,"api_cost_usd":0,"elapsed_seconds":elapsed,"prompt_eval_count":out.get("prompt_eval_count"),"eval_count":out.get("eval_count"),"eval_duration_ns":out.get("eval_duration"),"total_duration_ns":out.get("total_duration")},indent=2))
print("V6_2_ZERO_COST_SEMANTIC_PROBE=PASS")
