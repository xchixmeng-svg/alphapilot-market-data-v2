#!/usr/bin/env python3
"""Canonical prompt, payload construction, normalization and validation for V6.2."""
from __future__ import annotations

import copy
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from evidence_capability import derive_case_evidence_sets


class ContractError(ValueError):
    pass


SYSTEM = """You are the AlphaPilot V6.2 Taiwan-stock AI decision layer.
You receive one stock at one historical decision date with point-in-time evidence and immutable calibrated numerical forecasts.
Judge this stock independently. Do not rank it against other stocks and do not use TopK, quotas, fixed candidate counts, or fixed probability thresholds.
The question is whether this stock is forming a genuine >=10% opportunity, why now, whether the entry is attractive, and what would prove the thesis wrong.
Numerical forecasts are a calibrated base-rate prior: use them as evidence, but never echo, rewrite, or alter their probabilities in your response.
Never invent evidence. Revenue is not EPS revision. Price strength is not proof of news, industry pricing, inventory, or supply-demand.

Evidence availability has two distinct meanings:
1. system_unavailable_evidence_ids are permanent pipeline limitations shared by every case. They are not evidence or counter-evidence, never lower evidence_quality, and can never by themselves justify WATCH or REJECT. Do not mention them as a reason for caution.
2. case_missing_but_system_supported_evidence_ids are available in the system generally but absent for this case. They are case-specific and may legitimately affect evidence_quality or the decision.

Score evidence_quality only from case_available_evidence_ids, considering point-in-time validity, timeliness, internal consistency, independent corroboration, thesis sufficiency, and support for a concrete invalidation. Do not score it by counting families in the full registry. Three timely, consistent and sufficient available families may be STRONG; five stale or contradictory families may be WEAK.

Removing the system-limitations penalty does not turn available evidence into positive evidence. Availability is not direction. Inspect the actual values and text. Neutral, mixed, stale, merely present, or internally contradictory observations are not corroboration and must not be described as positive. Do not call valuation cheap or expensive without a supplied historical, peer, or growth-relative benchmark. Do not treat one short-period revenue change as durable growth by itself. Do not treat routine or procedural corporate disclosures as an economic catalyst unless their supplied content establishes a concrete earnings, demand, pricing, capital-allocation, or risk change. Price support by itself is not a sufficient why-now mechanism for a >=10% thesis.

STRONG means the chosen thesis is genuinely corroborated by timely, directionally supportive, independent case-available evidence and has a well-supported invalidation. It does not mean that many fields are populated. Bear-thesis facts that concretely contradict the chosen thesis must be represented in counter_evidence_ids rather than dismissed as merely neutral.

Decision semantics are qualitative contracts, not thresholds:
- REJECT: available evidence actively contradicts a credible >=10% opportunity, or no coherent profit thesis can be formed from available evidence.
- WATCH: a credible >=10% thesis exists, but available evidence shows mixed confirmation, poor present entry, immature timing, or a concrete unresolved contradiction.
- CANDIDATE: available evidence forms a coherent >=10% thesis, materially aligns with the calibrated forecasts, supports an actionable entry, and supports an invalidation/failure exit.
- HIGH_CONVICTION: CANDIDATE conditions hold with unusually strong coherence across independent available families, limited concrete counter-evidence, clear launch/entry logic, and a well-supported failure exit.

If your decision is materially weaker than the numerical forecasts imply, decision_reason must identify a specific contradiction in case_available_evidence_ids. A system limitation may never be the reason for discounting the forecast.
primary_evidence_ids, secondary_evidence_ids, and counter_evidence_ids may contain only supplied case_available_evidence_ids. counter_evidence_ids must identify evidence that concretely contradicts the thesis. Echo system_unavailable_evidence_ids exactly in system_limitations for transparency only.
Before finalizing REJECT or WATCH, verify that the reason comes from case-available evidence or a case-specific supported gap, not from permanent system limitations.
Before finalizing CANDIDATE or HIGH_CONVICTION, verify that you have identified a concrete economic or market launch mechanism, not merely a collection of available fields. The calibrated prior informs plausibility but does not make the qualitative evidence automatically bullish.
For CANDIDATE/HIGH_CONVICTION, provide a stock-specific entry zone and pre-entry AI Failure Exit Price based on case-available evidence; never use a universal fixed stop percentage.
For WATCH/REJECT, entry must be NOT_ACTIONABLE with null ideal_low and ideal_high; failure_exit.exit_price must be null and trigger_type must be UNAVAILABLE.
For CANDIDATE/HIGH_CONVICTION, entry bounds and failure_exit.exit_price must be finite positive numbers, ideal_low <= ideal_high, and failure_exit.exit_price must be strictly below ideal_high.
Return exactly one JSON object and no markdown."""


SYSTEM_LIMITATION_ALIASES = {
    "eps_revisions": ("eps revision", "eps revisions", "eps修正", "eps預估", "eps预估"),
    "analyst_consensus": ("analyst consensus", "分析師共識", "分析师共识"),
    "industry_pricing": ("industry pricing", "產業定價", "产业定价", "行業定價", "行业定价"),
    "inventory_supply_demand": ("inventory supply demand", "inventory and supply", "庫存供需", "库存供需"),
    "broad_news_semantics": ("broad news semantics", "news sentiment", "新聞情緒", "新闻情绪"),
}


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "research" / "V6_2_LLM_REASONING_SCHEMA.json"


@lru_cache(maxsize=1)
def load_response_schema() -> dict[str, Any]:
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ContractError("canonical response schema must be a closed object")
    return schema


def build_user_payload(pkt: dict[str, Any]) -> dict[str, Any]:
    sets = derive_case_evidence_sets(pkt["evidence"])
    model_case = copy.deepcopy(pkt)
    model_evidence = model_case["evidence"]
    # Remove the legacy two-way labels before prompting; only the explicit
    # three-way split is shown to the model.
    model_evidence.pop("available_evidence_ids", None)
    model_evidence.pop("missing_evidence_ids", None)
    model_evidence["evidence_availability"] = sets
    return {
        "case": model_case,
        "response_schema": load_response_schema(),
        "allowed_primary_secondary_evidence_ids": sets["case_available_evidence_ids"],
        "allowed_counter_evidence_ids": sets["case_available_evidence_ids"],
        **sets,
    }


def _canon(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _closed_enum(value: Any, allowed: set[str]) -> Any:
    hits = [x for x in allowed if _canon(x) == _canon(value)]
    return hits[0] if len(hits) == 1 else value


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _required_keys(schema: dict[str, Any]) -> set[str]:
    return set(schema["required"])


def normalize_output(pkt: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Apply harmless representation normalization; never infer semantics."""
    if not isinstance(raw, dict):
        raise ContractError("model output must be a JSON object")
    schema = load_response_schema()
    required = _required_keys(schema)
    if set(raw) != required:
        raise ContractError(f"schema keys: expected {sorted(required)}, got {sorted(raw)}")
    out = copy.deepcopy(raw)

    decisions = set(schema["properties"]["decision"]["enum"])
    qualities = set(schema["properties"]["evidence_quality"]["enum"])
    entry_statuses = set(schema["properties"]["entry"]["properties"]["status"]["enum"])
    trigger_types = set(schema["properties"]["failure_exit"]["properties"]["trigger_type"]["enum"])
    out["decision"] = _closed_enum(out["decision"], decisions)
    out["evidence_quality"] = _closed_enum(out["evidence_quality"], qualities)

    for field, nested_schema, enum_field, allowed in (
        ("entry", schema["properties"]["entry"], "status", entry_statuses),
        ("failure_exit", schema["properties"]["failure_exit"], "trigger_type", trigger_types),
    ):
        if not isinstance(out.get(field), dict) or set(out[field]) != _required_keys(nested_schema):
            raise ContractError(f"{field} must contain exactly {sorted(_required_keys(nested_schema))}")
        out[field][enum_field] = _closed_enum(out[field][enum_field], allowed)

    sets = derive_case_evidence_sets(pkt["evidence"])
    known = sets["case_available_evidence_ids"] + sets["system_unavailable_evidence_ids"] + sets["case_missing_but_system_supported_evidence_ids"]
    canonical: dict[str, list[str]] = {}
    for item in known:
        canonical.setdefault(_canon(item), []).append(item)

    for field in ("primary_evidence_ids", "secondary_evidence_ids", "counter_evidence_ids", "system_limitations"):
        value = out.get(field)
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ContractError(f"{field} must be a list of strings")
        normalized = []
        for item in value:
            hits = canonical.get(_canon(item), [])
            normalized.append(hits[0] if len(hits) == 1 else item)
        out[field] = normalized
    return out


def _family_mentioned(text: str, family: str) -> bool:
    lowered = text.lower()
    literal = family.replace("_", " ")
    if family.lower() in lowered or literal in lowered:
        return True
    words = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in family.split("_")]
    return all(word in lowered for word in words)


def _named_system_limitations(text: str, families: set[str]) -> list[str]:
    lowered = text.lower()
    found = []
    for family in families:
        aliases = SYSTEM_LIMITATION_ALIASES.get(family, ())
        if _family_mentioned(text, family) or any(alias.lower() in lowered for alias in aliases):
            found.append(family)
    return sorted(found)


def validate(pkt: dict[str, Any], out: dict[str, Any]) -> list[str]:
    """Raise ContractError on hard violations and return audit warnings."""
    schema = load_response_schema()
    props = schema["properties"]
    errors: list[str] = []
    warnings: list[str] = []
    if out.get("decision") not in set(props["decision"]["enum"]):
        errors.append("bad decision")
    if out.get("evidence_quality") not in set(props["evidence_quality"]["enum"]):
        errors.append("bad evidence_quality")

    for field in ("hypothesis_type", "hypothesis", "bull_thesis", "bear_thesis", "invalidation", "decision_reason"):
        value = out.get(field)
        spec = props[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"empty {field}")
        elif len(value) > spec.get("maxLength", 10**9):
            errors.append(f"{field} exceeds maxLength")

    sets = derive_case_evidence_sets(pkt["evidence"])
    available = set(sets["case_available_evidence_ids"])
    system_unavailable = set(sets["system_unavailable_evidence_ids"])
    case_missing = set(sets["case_missing_but_system_supported_evidence_ids"])
    if set(out.get("system_limitations", [])) != system_unavailable:
        errors.append("system_limitations must exactly equal system_unavailable_evidence_ids")
    if len(out.get("system_limitations", [])) != len(set(out.get("system_limitations", []))):
        errors.append("duplicate system_limitations")

    for field in ("primary_evidence_ids", "secondary_evidence_ids", "counter_evidence_ids"):
        ids = out.get(field)
        if not isinstance(ids, list):
            errors.append(f"{field} must be a list")
            continue
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate {field}")
        unsupported = set(ids) - available
        if unsupported:
            errors.append(f"{field} contains unavailable/unknown evidence: {sorted(unsupported)}")
    if not out.get("primary_evidence_ids"):
        errors.append("empty primary evidence")

    decision = out.get("decision")
    entry = out.get("entry", {})
    failure = out.get("failure_exit", {})
    entry_statuses = set(props["entry"]["properties"]["status"]["enum"])
    trigger_types = set(props["failure_exit"]["properties"]["trigger_type"]["enum"])
    if entry.get("status") not in entry_statuses:
        errors.append("bad entry status")
    if failure.get("trigger_type") not in trigger_types:
        errors.append("bad failure-exit trigger")

    actionable = decision in {"CANDIDATE", "HIGH_CONVICTION"}
    if actionable:
        low, high = entry.get("ideal_low"), entry.get("ideal_high")
        price = pkt["evidence"].get("price_volume_structure", {}).get("current_price")
        if entry.get("status") == "NOT_ACTIONABLE":
            errors.append("actionable decision has NOT_ACTIONABLE entry")
        if not _finite(low) or not _finite(high) or not (0 < float(low) <= float(high)):
            errors.append("bad actionable entry zone")
        elif not _finite(price) or not (0.65 * float(price) <= float(low) <= 1.30 * float(price) and 0.65 * float(price) <= float(high) <= 1.30 * float(price)):
            errors.append("entry zone implausibly far from current price")
        exit_price = failure.get("exit_price")
        if not _finite(exit_price) or not _finite(high) or not (0 < float(exit_price) < float(high)):
            errors.append("bad failure-exit price")
        if failure.get("trigger_type") == "UNAVAILABLE":
            errors.append("actionable decision has unavailable failure-exit trigger")
        if not isinstance(failure.get("reason"), str) or not failure["reason"].strip():
            errors.append("missing failure-exit reasoning")
    elif decision in {"REJECT", "WATCH"}:
        if entry.get("status") != "NOT_ACTIONABLE" or entry.get("ideal_low") is not None or entry.get("ideal_high") is not None:
            errors.append("non-actionable decision must use NOT_ACTIONABLE with null entry bounds")
        if failure.get("exit_price") is not None or failure.get("trigger_type") != "UNAVAILABLE":
            errors.append("non-actionable decision must not fabricate a failure exit")

        basis_text = " ".join(str(out.get(k, "")) for k in ("decision_reason", "bear_thesis"))
        review_text = basis_text + " " + str(out.get("invalidation", ""))
        mentions_system = bool(_named_system_limitations(basis_text, system_unavailable))
        mentions_available = any(_family_mentioned(basis_text, family) for family in available)
        has_counter = bool(out.get("counter_evidence_ids"))
        if mentions_system and not mentions_available and not has_counter:
            if decision == "REJECT":
                errors.append("REJECT appears justified only by permanent system limitations")
            else:
                warnings.append("WATCH text may rely only on permanent system limitations")
        generic_absence = re.search(r"\b(unavailable|not available|no data|lack of data|missing data)\b", review_text.lower())
        if generic_absence and not mentions_available and not case_missing:
            warnings.append("non-actionable reason uses generic missing-data language without a case-specific supported gap")

    all_narrative = " ".join(str(out.get(k, "")) for k in ("hypothesis", "bull_thesis", "bear_thesis", "invalidation", "decision_reason"))
    named_system_limitations = _named_system_limitations(all_narrative, system_unavailable)
    if named_system_limitations:
        errors.append(
            "system-unavailable families mentioned outside system_limitations: "
            + ",".join(named_system_limitations)
        )

    if errors:
        raise ContractError("; ".join(errors))
    return warnings
