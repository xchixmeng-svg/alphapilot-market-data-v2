#!/usr/bin/env python3
"""Production Stage-C runner for V6.2/V4.2 upgraded packets.

The runner is local-model only. It never opens outcomes. Stage C uses a compact
coded decision contract so the local model cannot expand into unbounded prose.
There are no retries and no deterministic admission thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from v6_2_v4_2_contract import (
    ContractError,
    load_jsonl,
    validate_decision_semantics,
    validate_packet,
)

DECISIONS = ["REJECT", "WATCH", "CANDIDATE", "HIGH_CONVICTION"]
ACTIONABLE = {"CANDIDATE", "HIGH_CONVICTION"}
HYPOTHESIS_CODES = [
    "REVENUE_ACCELERATION",
    "PRICE_VOLUME_CONTINUATION",
    "EVENT_CATALYST",
    "INSTITUTIONAL_FLOW_CONFIRMATION",
    "VALUATION_RERATING",
    "MULTI_FACTOR_UPSIDE",
    "NO_CLEAR_UPSIDE_THESIS",
]
INVALIDATION_CODES = [
    "PRICE_STRUCTURE_BREAK",
    "REVENUE_MOMENTUM_BREAK",
    "EVENT_CATALYST_FAILURE",
    "FLOW_REVERSAL",
    "VALUATION_EXPECTATION_BREAK",
    "MULTI_EVIDENCE_FAILURE",
    "THESIS_NOT_ESTABLISHED",
]
DECISION_REASON_CODES = [
    "EVIDENCE_CONVERGENCE",
    "CONSTRUCTIVE_BUT_NOT_ACTIONABLE",
    "CONFLICTING_AVAILABLE_EVIDENCE",
    "INSUFFICIENT_AVAILABLE_EDGE",
    "AVAILABLE_RISK_DOMINATES",
    "NO_TIMELY_SETUP",
]
TIMING_CODES = ["NOW", "DEVELOPING", "WAIT_FOR_CONFIRMATION", "NO_SETUP"]
CONTEXT_STANCES = ["SUPPORTIVE", "NEUTRAL", "CAUTIONARY"]

DECISION_SYSTEM = """You are AlphaPilot V6.2/V4.2 Stage C.
Use only the supplied point-in-time packet. Frozen numerical forecasts are read-only.
causal_context_read_only is background context, not company evidence and may never appear
in evidence-id arrays. Do not infer company industry from the 37 global industry indices.
Evidence families absent from the packet are neutral and must not affect the decision.
Judge a genuine >=10% opportunity without TopK, quotas, or a fixed probability threshold.
Return only the compact coded JSON object required by the schema. Do not add prose."""

ACTION_SYSTEM = """You are AlphaPilot V6.2/V4.2 execution planning.
The decision is already actionable and locked. Use only the supplied point-in-time packet
and locked decision to choose entry prices and a thesis-specific failure exit.
Causal context may affect timing/risk but is not company bull/bear evidence.
Do not use a universal fixed-percent stop. Return only the compact JSON required by schema."""


def decision_schema(packet: dict[str, Any]) -> dict[str, Any]:
    available = list(packet["evidence"].get("available_evidence_ids", []))
    if not available:
        raise ContractError("no available company evidence IDs")
    evidence_item = {"type": "string", "enum": available}
    ids = {"type": "array", "uniqueItems": True, "maxItems": min(4, len(available)), "items": evidence_item}
    required = [
        "decision",
        "hypothesis_code",
        "primary_evidence_ids",
        "secondary_evidence_ids",
        "bull_evidence_ids",
        "bear_evidence_ids",
        "invalidation_code",
        "invalidation_evidence_ids",
        "timing_code",
        "market_context_stance",
        "macro_context_stance",
        "industry_index_context_stance",
        "event_timeliness_context_stance",
        "revenue_context_stance",
        "decision_reason_code",
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "decision": {"type": "string", "enum": DECISIONS},
            "hypothesis_code": {"type": "string", "enum": HYPOTHESIS_CODES},
            "primary_evidence_ids": {**ids, "minItems": 1},
            "secondary_evidence_ids": ids,
            "bull_evidence_ids": ids,
            "bear_evidence_ids": ids,
            "invalidation_code": {"type": "string", "enum": INVALIDATION_CODES},
            "invalidation_evidence_ids": {**ids, "minItems": 1},
            "timing_code": {"type": "string", "enum": TIMING_CODES},
            "market_context_stance": {"type": "string", "enum": CONTEXT_STANCES},
            "macro_context_stance": {"type": "string", "enum": CONTEXT_STANCES},
            "industry_index_context_stance": {"type": "string", "enum": CONTEXT_STANCES},
            "event_timeliness_context_stance": {"type": "string", "enum": CONTEXT_STANCES},
            "revenue_context_stance": {"type": "string", "enum": CONTEXT_STANCES},
            "decision_reason_code": {"type": "string", "enum": DECISION_REASON_CODES},
        },
    }


def action_schema(packet: dict[str, Any]) -> dict[str, Any]:
    current = float(packet["evidence"]["price_volume_structure"]["current_price"])
    lo, hi = max(0.01, current * 0.65), current * 1.30
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["entry", "failure_exit"],
        "properties": {
            "entry": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "ideal_low", "ideal_high"],
                "properties": {
                    "status": {"type": "string", "enum": ["NOW", "WAIT_FOR_PULLBACK", "WAIT_FOR_CONFIRMATION"]},
                    "ideal_low": {"type": "number", "minimum": lo, "maximum": hi},
                    "ideal_high": {"type": "number", "minimum": lo, "maximum": hi},
                },
            },
            "failure_exit": {
                "type": "object",
                "additionalProperties": False,
                "required": ["exit_price", "trigger_type"],
                "properties": {
                    "exit_price": {"type": "number", "exclusiveMinimum": 0, "maximum": hi},
                    "trigger_type": {
                        "type": "string",
                        "enum": [
                            "PRICE_STRUCTURE_BREAK",
                            "THESIS_INVALIDATION",
                            "CATALYST_FAILURE",
                            "VALUATION_EXPECTATION_BREAK",
                            "REVENUE_MOMENTUM_BREAK",
                            "FLOW_REVERSAL",
                            "MULTI_EVIDENCE_FAILURE",
                        ],
                    },
                },
            },
        },
    }


def call_ollama(
    url: str,
    model: str,
    system: str,
    user: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(user, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": 900},
        },
        ensure_ascii=False,
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=1200) as response:
        raw = json.loads(response.read().decode())
    content = raw.get("message", {}).get("content", "")
    if raw.get("done_reason") == "length":
        raise ContractError(f"compact local-model contract exceeded output budget; chars={len(content)}")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"local model returned invalid compact JSON; done_reason={raw.get('done_reason')!r}; "
            f"chars={len(content)}; error={exc}"
        ) from exc
    return parsed, time.time() - started


def assert_evidence_ids(packet: dict[str, Any], result: dict[str, Any]) -> None:
    allowed = set(packet["evidence"].get("available_evidence_ids", []))
    for field in (
        "primary_evidence_ids",
        "secondary_evidence_ids",
        "bull_evidence_ids",
        "bear_evidence_ids",
        "invalidation_evidence_ids",
    ):
        values = result.get(field)
        if not isinstance(values, list) or any(value not in allowed for value in values):
            raise ContractError(f"context or invented evidence leaked into {field}")


def render_decision_narratives(compact: dict[str, Any]) -> dict[str, str]:
    primary = ",".join(compact["primary_evidence_ids"]) or "NONE"
    bull = ",".join(compact["bull_evidence_ids"]) or "NONE"
    bear = ",".join(compact["bear_evidence_ids"]) or "NONE"
    invalid = ",".join(compact["invalidation_evidence_ids"]) or "NONE"
    return {
        "hypothesis": f"{compact['hypothesis_code']}; primary_evidence={primary}",
        "bull_thesis": f"bull_evidence={bull}",
        "bear_thesis": f"bear_evidence={bear}",
        "invalidation": f"{compact['invalidation_code']}; evidence={invalid}",
        "context_assessment": (
            f"market={compact['market_context_stance']}; "
            f"macro={compact['macro_context_stance']}; "
            f"industry_indices={compact['industry_index_context_stance']}; "
            f"event_timeliness={compact['event_timeliness_context_stance']}; "
            f"revenue_context={compact['revenue_context_stance']}"
        ),
        "decision_reason": (
            f"{compact['decision_reason_code']}; timing={compact['timing_code']}"
        ),
    }


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
        packets = packets[: args.limit]

    rows = []
    for packet in packets:
        validate_packet(packet)
        if not isinstance(packet.get("causal_context_read_only"), dict):
            raise ContractError("Stage C packet missing causal_context_read_only")
        base = {
            "case_id": packet["case_id"],
            "decision_date": packet["decision_date"],
            "code": packet["code"],
            "stage_c_received_causal_context_read_only": True,
        }
        if args.transport_only:
            rows.append({**base, "status": "TRANSPORT_VERIFIED"})
            continue

        compact, seconds = call_ollama(
            args.ollama_url,
            args.model,
            DECISION_SYSTEM,
            {"case": packet},
            decision_schema(packet),
        )
        assert_evidence_ids(packet, compact)
        narratives = render_decision_narratives(compact)
        validate_decision_semantics(narratives)
        decision = {**compact, **narratives}

        result = {
            **base,
            "status": "MODEL_COMPLETED",
            "model": args.model,
            "decision_seconds": seconds,
            **decision,
        }

        if decision["decision"] in ACTIONABLE:
            action, action_seconds = call_ollama(
                args.ollama_url,
                args.model,
                ACTION_SYSTEM,
                {"case": packet, "locked_decision": decision},
                action_schema(packet),
            )
            entry, failure = action["entry"], action["failure_exit"]
            if not (0 < float(entry["ideal_low"]) <= float(entry["ideal_high"])):
                raise ContractError("invalid entry order")
            if not (0 < float(failure["exit_price"]) < float(entry["ideal_high"])):
                raise ContractError("failure exit must be below entry high")
            failure["reason"] = (
                f"{failure['trigger_type']}; "
                f"locked_invalidation={decision['invalidation_code']}; "
                f"evidence={','.join(decision['invalidation_evidence_ids'])}"
            )
            result["entry"] = entry
            result["failure_exit"] = failure
            result["action_seconds"] = action_seconds
        else:
            result["entry"] = {
                "status": "NOT_ACTIONABLE",
                "ideal_low": None,
                "ideal_high": None,
            }
            result["failure_exit"] = {
                "exit_price": None,
                "reason": None,
                "trigger_type": "UNAVAILABLE",
            }
            result["action_seconds"] = 0.0

        rows.append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "transport_only": args.transport_only,
                "paid_model_called": False,
                "decision_contract": "COMPACT_CODED_V1",
            }
        )
    )


if __name__ == "__main__":
    main()
