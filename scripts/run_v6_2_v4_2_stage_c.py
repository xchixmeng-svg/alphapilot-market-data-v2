#!/usr/bin/env python3
"""Production Stage-C runner for V6.2/V4.2 upgraded packets.

The runner is local-model only.  It never opens outcomes and makes exactly one
model call per decision stage; retries are intentionally absent.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from v6_2_v4_2_contract import ContractError, load_jsonl, validate_decision_semantics, validate_packet

DECISIONS = ["REJECT", "WATCH", "CANDIDATE", "HIGH_CONVICTION"]
ACTIONABLE = {"CANDIDATE", "HIGH_CONVICTION"}

DECISION_SYSTEM = """You are AlphaPilot V6.2/V4.2 Stage C.
Use only the supplied point-in-time packet. Frozen numerical forecasts are read-only.
causal_context_read_only is background context, not company evidence: never put a context
field in bull_evidence_ids, bear_evidence_ids, primary_evidence_ids, or secondary_evidence_ids.
Do not infer the company's industry from industry_index_context; those 37 indices are global context only.
Do not invent missing evidence, raw PE/PB, future paths, labels, or outcomes. Evidence families absent from the packet are neutral: never mention them or use their absence as support, counter-evidence, invalidation, or veto.\nJudge a genuine >=10% opportunity without TopK, quotas, or a fixed probability threshold. Keep every narrative field concise and do not repeat packet contents.\nReturn exactly one JSON object matching the supplied schema."""

ACTION_SYSTEM = """You are AlphaPilot V6.2/V4.2 execution planning.
The Stage-C decision is already actionable and locked. Use supplied point-in-time company evidence
for the entry and thesis-specific failure exit. Causal context may adjust timing/risk but may not
masquerade as bull or bear company evidence. Do not use a universal fixed-percent stop.
Return exactly one JSON object matching the supplied schema."""


def decision_schema(packet: dict[str, Any]) -> dict[str, Any]:
    available = list(packet["evidence"].get("available_evidence_ids", []))
    if not available:
        raise ContractError("no available company evidence IDs")
    ids = {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": available}}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["decision", "hypothesis", "primary_evidence_ids", "secondary_evidence_ids", "bull_evidence_ids", "bear_evidence_ids", "bull_thesis", "bear_thesis", "invalidation", "context_assessment", "decision_reason"],
        "properties": {
            "decision": {"type": "string", "enum": DECISIONS},
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": 280},
            "primary_evidence_ids": {**ids, "minItems": 1},
            "secondary_evidence_ids": ids,
            "bull_evidence_ids": ids,
            "bear_evidence_ids": ids,
            "bull_thesis": {"type": "string", "minLength": 1, "maxLength": 420},
            "bear_thesis": {"type": "string", "minLength": 1, "maxLength": 420},
            "invalidation": {"type": "string", "minLength": 1, "maxLength": 320},
            "context_assessment": {"type": "string", "minLength": 1, "maxLength": 360},
            "decision_reason": {"type": "string", "minLength": 1, "maxLength": 420},
        },
    }


def action_schema(packet: dict[str, Any]) -> dict[str, Any]:
    current = float(packet["evidence"]["price_volume_structure"]["current_price"])
    lo, hi = max(0.01, current * 0.65), current * 1.30
    return {
        "type": "object", "additionalProperties": False, "required": ["entry", "failure_exit"],
        "properties": {
            "entry": {"type": "object", "additionalProperties": False, "required": ["status", "ideal_low", "ideal_high"], "properties": {
                "status": {"type": "string", "enum": ["NOW", "WAIT_FOR_PULLBACK", "WAIT_FOR_CONFIRMATION"]},
                "ideal_low": {"type": "number", "minimum": lo, "maximum": hi},
                "ideal_high": {"type": "number", "minimum": lo, "maximum": hi},
            }},
            "failure_exit": {"type": "object", "additionalProperties": False, "required": ["exit_price", "reason", "trigger_type"], "properties": {
                "exit_price": {"type": "number", "exclusiveMinimum": 0, "maximum": hi},
                "reason": {"type": "string", "minLength": 1, "maxLength": 360},
                "trigger_type": {"type": "string", "enum": ["PRICE_STRUCTURE_BREAK", "THESIS_INVALIDATION", "CATALYST_FAILURE", "VALUATION_EXPECTATION_BREAK", "MULTI_EVIDENCE_FAILURE"]},
            }},
        },
    }


def call_ollama(url: str, model: str, system: str, user: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], float]:
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))}], "stream": False, "format": schema, "options": {"temperature": 0, "num_predict": 2048}}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=1200) as response:
        raw = json.loads(response.read().decode())
    content = raw.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"local model returned invalid JSON; done_reason={raw.get('done_reason')!r}; "
            f"chars={len(content)}; error={exc}"
        ) from exc
    return parsed, time.time() - started


def assert_evidence_ids(packet: dict[str, Any], result: dict[str, Any]) -> None:
    allowed = set(packet["evidence"].get("available_evidence_ids", []))
    for field in ("primary_evidence_ids", "secondary_evidence_ids", "bull_evidence_ids", "bear_evidence_ids"):
        values = result.get(field)
        if not isinstance(values, list) or any(value not in allowed for value in values):
            raise ContractError(f"context or invented evidence leaked into {field}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packets", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("V62_LOCAL_MODEL", "qwen2.5:7b"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transport-only", action="store_true")
    args = ap.parse_args()
    packets = load_jsonl(args.packets)
    if args.limit > 0:
        packets = packets[:args.limit]
    rows = []
    for packet in packets:
        validate_packet(packet)
        if not isinstance(packet.get("causal_context_read_only"), dict):
            raise ContractError("Stage C packet missing causal_context_read_only")
        base = {"case_id": packet["case_id"], "decision_date": packet["decision_date"], "code": packet["code"], "stage_c_received_causal_context_read_only": True}
        if args.transport_only:
            rows.append({**base, "status": "TRANSPORT_VERIFIED"})
            continue
        decision, seconds = call_ollama(args.ollama_url, args.model, DECISION_SYSTEM, {"case": packet}, decision_schema(packet))
        assert_evidence_ids(packet, decision)
        validate_decision_semantics(decision)
        result = {**base, "status": "MODEL_COMPLETED", "model": args.model, "decision_seconds": seconds, **decision}
        if decision["decision"] in ACTIONABLE:
            action, action_seconds = call_ollama(args.ollama_url, args.model, ACTION_SYSTEM, {"case": packet, "locked_decision": decision}, action_schema(packet))
            entry, failure = action["entry"], action["failure_exit"]
            if not (0 < float(entry["ideal_low"]) <= float(entry["ideal_high"])):
                raise ContractError("invalid entry order")
            if not (0 < float(failure["exit_price"]) < float(entry["ideal_high"])):
                raise ContractError("failure exit must be below entry high")
            result.update(action)
            result["action_seconds"] = action_seconds
        else:
            result["entry"] = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
            result["failure_exit"] = {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}
            result["action_seconds"] = 0.0
        rows.append(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "transport_only": args.transport_only, "paid_model_called": False}))


if __name__ == "__main__":
    main()
