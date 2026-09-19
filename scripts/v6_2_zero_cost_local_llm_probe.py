#!/usr/bin/env python3
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"

evidence={
  "date":"2024-12-31",
  "code":"2330",
  "price_structure":{"status":"AVAILABLE","ret20":0.08,"close_pos60":0.83},
  "institutional_flow":{"status":"AVAILABLE","foreign_net_ratio_5d":0.021},
  "revenue_earnings":{"status":"PARTIAL","revenue_yoy_pct":18.4},
  "eps_revisions":{"status":"UNAVAILABLE"},
  "valuation":{"status":"AVAILABLE","pe":24.1},
  "industry_pricing_supply_demand":{"status":"UNAVAILABLE"}
}
numerical={
  "p_up10_before_down5_h20":0.61,
  "expected_mae_low_pct":-0.04,
  "expected_mae_high_pct":-0.01
}

system="""You are the V6.2 stock opportunity reasoning layer.
Return ONE JSON object only, no markdown.
You may reason from evidence but MUST NOT invent or alter any numerical value.
Do not use rank, TopK, or forced quota logic.
Allowed decision: REJECT, WATCH, CANDIDATE, HIGH_CONVICTION, INSUFFICIENT_INPUT.
For CANDIDATE/HIGH_CONVICTION include non-empty why_now, evidence_used, counter_evidence, invalidation.
If an evidence family is unavailable, explicitly treat it as missing rather than inventing facts.
Echo numerical fields exactly if you include them."""

user={
  "task":"Judge this stock independently. This is a provider-capability probe, not a trading recommendation.",
  "evidence":evidence,
  "numerical":numerical,
  "required_output_example":{
    "decision":"CANDIDATE",
    "why_now":"...",
    "evidence_used":["..."],
    "counter_evidence":["..."],
    "invalidation":"A concrete evidence-based condition that would falsify the setup.",
    "p_up10_before_down5_h20":0.61,
    "expected_mae_low_pct":-0.04,
    "expected_mae_high_pct":-0.01
  }
}

tok=AutoTokenizer.from_pretrained(MODEL_ID)
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
messages=[
  {"role":"system","content":system},
  {"role":"user","content":json.dumps(user,ensure_ascii=False,separators=(",",":"))}
]
text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
inputs=tok([text],return_tensors="pt")
t0=time.time()
with torch.no_grad():
    out=model.generate(
        **inputs,
        max_new_tokens=420,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.eos_token_id,
    )
gen=tok.decode(out[0][inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
elapsed=time.time()-t0
print(f"LOCAL_MODEL={MODEL_ID}")
print(f"ELAPSED_SECONDS={elapsed:.3f}")
print("RAW_RESPONSE="+gen.replace("\n"," ")[:2000])

m=re.search(r"\{.*\}",gen,re.S)
if not m:
    raise SystemExit("no JSON object in local-model response")
obj=json.loads(m.group(0))
Path("out").mkdir(exist_ok=True)
Path("out/evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
Path("out/numerical.json").write_text(json.dumps(numerical,ensure_ascii=False,indent=2))
Path("out/response.json").write_text(json.dumps(obj,ensure_ascii=False,indent=2))
Path("out/provider_meta.json").write_text(json.dumps({
  "provider":"local_open_weights",
  "model":MODEL_ID,
  "api_cost_usd":0,
  "elapsed_seconds":elapsed,
  "torch_version":torch.__version__
},indent=2))
