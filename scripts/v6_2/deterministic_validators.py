"""
deterministic_validators.py

All checks in this file are pure functions over already-parsed JSON dicts.
No LLM calls here. Split into:

  validate_stage1_output(raw, expected_case_id, case_available_evidence_ids, pkt_evidence)
      -> (normalized, errors)
  validate_stage2_output(raw, stage1_observations, pkt_evidence) -> (normalized, errors, warnings)
  validate_stage3_output(raw, stage1_evidence_ids) -> (normalized, errors)

`errors` are hard contract violations: the case must be retried/revised,
never persisted as a final result while errors is non-empty.
`warnings` are soft, best-effort text-pattern signals for human/audit
review; they never block finalization by themselves.

EXCEPTION SAFETY (fixed after external review): every public validate_*
function in this module wraps its body in try/except Exception and
converts any unexpected exception (e.g. a malformed field of the wrong
Python type, such as primary_evidence_ids being an int instead of a list)
into a normal contract error string. No exception may propagate out of
this module -- a single malformed model response must never crash the
orchestrator or abort a batch.

GROUNDING (added after external review): Stage 1 is no longer trusted on
the strength of "the field is a non-empty list of strings". Numeric claims
inside exact_observations are cross-checked against the actual numeric
values present in the corresponding raw evidence family of the packet
(see _extract_numbers, _ground_numeric_claims). A claim whose numbers do
not appear anywhere in that family's raw data is flagged UNGROUNDED. Free
text with no extractable number is passed through (this mechanical check
cannot verify prose meaning -- that remains Stage 3's job) rather than
silently trusted as fact.

CASE_ID CHECK (added after external review): Stage 1's returned case_id
must match the case_id the request was actually built for, with strict
type and value equality. A Stage1 response for the wrong case must never
be accepted.
"""

from __future__ import annotations

import re
from typing import Any

from evidence_capability import derive_case_evidence_sets

# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

DIRECTION_VALUES = {"SUPPORTIVE", "CONTRADICTORY", "NEUTRAL", "MIXED", "NOT_INTERPRETABLE"}
TIMELINESS_VALUES = {"CURRENT", "STALE", "UNKNOWN_AGE"}
RELEVANCE_VALUES = {"HIGH", "MEDIUM", "LOW"}

DECISION_VALUES = {"REJECT", "WATCH", "CANDIDATE", "HIGH_CONVICTION"}
EVIDENCE_QUALITY_VALUES = {"STRONG", "MODERATE", "WEAK"}
ENTRY_STATUS_VALUES = {"NOW", "WAIT_FOR_PULLBACK", "WAIT_FOR_CONFIRMATION", "NOT_ACTIONABLE"}
FAILURE_TRIGGER_VALUES = {
    "PRICE_STRUCTURE_BREAK", "THESIS_INVALIDATION", "CATALYST_FAILURE",
    "VALUATION_EXPECTATION_BREAK", "MULTI_EVIDENCE_FAILURE", "UNAVAILABLE",
}
NON_ACTIONABLE_DECISIONS = {"REJECT", "WATCH"}
ACTIONABLE_DECISIONS = {"CANDIDATE", "HIGH_CONVICTION"}

CRITIC_VERDICT_VALUES = {"APPROVE", "REVISE"}
CRITIC_PROBLEM_TYPES = {
    "UNGROUNDED_CLAIM", "CONTRADICTS_EXACT_OBSERVATIONS", "OVERSTATES_DIRECTION",
    "BENCHMARK_MISSING_BUT_VALUE_JUDGMENT_MADE", "SINGLE_PERIOD_TREATED_AS_TREND",
    "ROUTINE_FILING_TREATED_AS_CATALYST", "MIXED_EVIDENCE_TREATED_AS_UNIFORM",
    "CLAIM_NOT_REFLECTED_IN_EVIDENCE_ID_LISTS", "DECISION_INCONSISTENT_WITH_EVIDENCE_BALANCE",
}
CRITIC_PROBLEM_REQUIRED_KEYS = {"field", "claim_text", "cited_evidence_id", "problem_type", "explanation"}
CRITIC_PROBLEM_FIELD_VALUES = {
    "bull_thesis", "bear_thesis", "decision_reason", "invalidation", "hypothesis",
    "evidence_quality", "decision",
}

# Narrow, enumerable phrase sets for the HARD-ERROR text checks (EN + ZH).
VALUE_JUDGMENT_PHRASES = {
    "cheap", "expensive", "undervalued", "overvalued", "reasonable",
    "fairly valued", "attractively valued", "reasonable valuation",
    "便宜", "昂貴", "低估", "高估", "合理", "估值合理", "估值便宜", "估值偏低", "估值偏高",
}
DURATIVE_TREND_PHRASES = {
    "sustained growth", "continued growth", "ongoing growth", "continuation of growth",
    "trend of growth", "consistent growth",
    "持續成長", "持續增長", "延續成長", "穩定成長", "持續性成長",
}
CATALYST_PHRASES = {
    "catalyst", "positive catalyst", "new catalyst", "driving factor", "key driver",
    "催化", "催化劑", "正面消息", "利多",
}

# Match standalone numeric values, but not digits embedded in field names
# such as ret5_pct, ret20_pct, foreign_mean5, or support_low60.  Those
# digits describe a lookback window; they are not claimed packet values.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])")

_VALUATION_BENCHMARK_KEYS = {
    "peer_pe", "peer_pe_median", "historical_pe_band", "historical_pe_low",
    "historical_pe_high", "sector_median_pe", "peer_pb", "historical_pb_band",
}


# ---------------------------------------------------------------------------
# Text-matching helpers
# ---------------------------------------------------------------------------


def _text_mentions_any_phrase(text: str, phrases: set[str]) -> list[str]:
    lowered = text.lower()
    return [p for p in phrases if p.lower() in lowered]


def _text_mentions_family(text_lower: str, family_id: str) -> bool:
    words = [w for w in family_id.split("_") if len(w) >= 3] or family_id.split("_")
    for w in words:
        singular = w[:-1] if w.endswith("s") and len(w) > 1 else w
        if singular in text_lower or w in text_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Grounding helpers (numeric cross-check against raw packet evidence)
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def _flatten_raw_values(raw_family: Any) -> list[float]:
    numbers: list[float] = []
    if isinstance(raw_family, bool):
        return numbers
    if isinstance(raw_family, (int, float)):
        numbers.append(float(raw_family))
    elif isinstance(raw_family, dict):
        for v in raw_family.values():
            numbers.extend(_flatten_raw_values(v))
    elif isinstance(raw_family, (list, tuple)):
        for v in raw_family:
            numbers.extend(_flatten_raw_values(v))
    elif isinstance(raw_family, str):
        numbers.extend(_extract_numbers(raw_family))
    return numbers


def _number_matches_any(claimed: float, raw_numbers: list[float], rel_tol: float = 0.01, abs_tol: float = 0.011) -> bool:
    """~1% tolerance, tight on purpose -- this exists to catch fabrication,
    not to be lenient about it. Also tolerates a fraction/percent mismatch
    (0.1984 vs 19.84) since raw data may store either form."""
    for r in raw_numbers:
        if abs(claimed - r) <= max(abs_tol, abs(r) * rel_tol):
            return True
        if abs(claimed - r * 100) <= max(abs_tol, abs(r * 100) * rel_tol):
            return True
        if abs(claimed * 100 - r) <= max(abs_tol, abs(r) * rel_tol):
            return True
    return False


def _ground_numeric_claims(exact_observations: list[str], evidence_id: str, pkt_evidence: dict) -> list[str]:
    """
    Empty list if every number in exact_observations traces back to THIS
    SAME family's raw data in pkt_evidence. Cross-family number-borrowing
    is caught naturally: a number is only accepted if it appears in the
    raw data of the family named by evidence_id, not any other family.
    Non-numeric prose is not checked here (left for Stage 3's judgment).
    """
    raw_family = pkt_evidence.get(evidence_id)
    if raw_family is None:
        # No raw numeric data available for this family in the packet at
        # all -- cannot mechanically verify; do not fabricate a false
        # failure for a family that is legitimately non-numeric (e.g. a
        # pure text/id family).
        return []
    raw_numbers = _flatten_raw_values(raw_family)
    if not raw_numbers:
        return []
    problems = []
    for obs_text in exact_observations:
        for claimed in _extract_numbers(obs_text):
            if not _number_matches_any(claimed, raw_numbers):
                problems.append(
                    f"exact_observations claim '{obs_text}' for evidence_id={evidence_id!r} contains "
                    f"number {claimed} not found in this family's raw packet data {raw_numbers}."
                )
    return problems


# ---------------------------------------------------------------------------
# Stage 1 validation
# ---------------------------------------------------------------------------


def validate_stage1_output(
    raw: Any,
    expected_case_id: int,
    case_available_evidence_ids: list[str],
    pkt_evidence: dict | None = None,
) -> tuple[dict, list[str]]:
    try:
        return _validate_stage1_output_impl(raw, expected_case_id, case_available_evidence_ids, pkt_evidence)
    except Exception as e:  # noqa: BLE001 -- never let a malformed field crash the caller
        return {}, [f"Stage1 validator crashed on malformed input: {type(e).__name__}: {e}"]


def _validate_stage1_output_impl(
    raw: Any,
    expected_case_id: int,
    case_available_evidence_ids: list[str],
    pkt_evidence: dict | None,
) -> tuple[dict, list[str]]:
    errors: list[str] = []

    if not isinstance(raw, dict):
        return {}, [f"Stage1 output must be a JSON object, got {type(raw)}"]

    extra = set(raw.keys()) - {"case_id", "evidence_observations"}
    if extra:
        errors.append(f"Stage1 output has unexpected top-level key(s): {sorted(extra)}")
    missing = {"case_id", "evidence_observations"} - set(raw.keys())
    if missing:
        errors.append(f"Stage1 output missing key(s): {sorted(missing)}")
    if errors:
        return {}, errors

    returned_case_id = raw["case_id"]
    if not isinstance(returned_case_id, int) or isinstance(returned_case_id, bool) or returned_case_id != expected_case_id:
        return {}, [
            f"Stage1 output case_id={returned_case_id!r} (type={type(returned_case_id).__name__}) does not "
            f"match expected_case_id={expected_case_id!r}. This output cannot be trusted for any case."
        ]

    obs_list = raw["evidence_observations"]
    if not isinstance(obs_list, list) or not obs_list:
        return {}, ["Stage1 evidence_observations must be a non-empty list"]

    seen_ids: set[str] = set()
    normalized_obs = []
    for i, obs in enumerate(obs_list):
        prefix = f"evidence_observations[{i}]"
        if not isinstance(obs, dict):
            errors.append(f"{prefix} must be an object")
            continue

        required = {
            "evidence_id", "exact_observations", "direction", "timeliness",
            "relevance_to_hypothesis_space", "limitations", "benchmark_available",
        }
        extra_keys = set(obs.keys()) - required
        missing_keys = required - set(obs.keys())
        if extra_keys:
            errors.append(f"{prefix} has unexpected key(s): {sorted(extra_keys)}")
        if missing_keys:
            errors.append(f"{prefix} missing key(s): {sorted(missing_keys)}")
            continue

        eid = obs["evidence_id"]
        if not isinstance(eid, str):
            errors.append(f"{prefix}.evidence_id must be a string, got {type(eid).__name__}")
            continue
        if eid not in case_available_evidence_ids:
            errors.append(f"{prefix}.evidence_id={eid!r} is not in case_available_evidence_ids {case_available_evidence_ids}")
        if eid in seen_ids:
            errors.append(f"{prefix}.evidence_id={eid!r} is duplicated")
        seen_ids.add(eid)

        exact_obs = obs["exact_observations"]
        if not isinstance(exact_obs, list) or not exact_obs or not all(isinstance(x, str) for x in exact_obs):
            errors.append(f"{prefix}.exact_observations must be a non-empty list of strings")
            exact_obs = []

        direction = obs["direction"]
        if direction not in DIRECTION_VALUES:
            errors.append(f"{prefix}.direction={direction!r} not in {sorted(DIRECTION_VALUES)}")

        timeliness = obs["timeliness"]
        if timeliness not in TIMELINESS_VALUES:
            errors.append(f"{prefix}.timeliness={timeliness!r} not in {sorted(TIMELINESS_VALUES)}")

        relevance = obs["relevance_to_hypothesis_space"]
        if relevance not in RELEVANCE_VALUES:
            errors.append(f"{prefix}.relevance_to_hypothesis_space={relevance!r} not in {sorted(RELEVANCE_VALUES)}")

        limitations = obs["limitations"]
        if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
            errors.append(f"{prefix}.limitations must be a list of strings")
            limitations = []

        benchmark_available = obs["benchmark_available"]
        if not isinstance(benchmark_available, bool):
            errors.append(f"{prefix}.benchmark_available must be boolean")

        # For valuation this flag is a packet fact, not a model judgment.
        # Enforce it deterministically whenever the raw family is present so
        # the model cannot bypass the no-benchmark rule by echoing true.
        if eid == "valuation" and pkt_evidence is not None:
            raw_valuation = pkt_evidence.get("valuation")
            if isinstance(raw_valuation, dict):
                expected_benchmark = bool(_VALUATION_BENCHMARK_KEYS & set(raw_valuation))
                if benchmark_available is not expected_benchmark:
                    errors.append(
                        f"{prefix}.benchmark_available={benchmark_available!r} contradicts the raw "
                        f"valuation packet; expected {expected_benchmark!r} from explicit benchmark fields."
                    )

        if eid == "valuation" and benchmark_available is False:
            has_caveat = any("benchmark" in lim.lower() or "基準" in lim or "比較" in lim for lim in limitations)
            if not has_caveat:
                errors.append(
                    f"{prefix}: evidence_id='valuation' has benchmark_available=false but limitations "
                    f"does not state the no-benchmark caveat required by the contract."
                )

        if pkt_evidence is not None and eid in case_available_evidence_ids and exact_obs:
            grounding_problems = _ground_numeric_claims(exact_obs, eid, pkt_evidence)
            errors.extend(f"{prefix}: {p}" for p in grounding_problems)

        normalized_obs.append(
            {
                "evidence_id": eid,
                "exact_observations": exact_obs,
                "direction": direction,
                "timeliness": timeliness,
                "relevance_to_hypothesis_space": relevance,
                "limitations": limitations,
                "benchmark_available": benchmark_available,
            }
        )

    missing_families = set(case_available_evidence_ids) - seen_ids
    if missing_families:
        errors.append(f"Stage1 output does not cover all case_available_evidence_ids; missing: {sorted(missing_families)}")

    normalized = {"case_id": returned_case_id, "evidence_observations": normalized_obs}
    return normalized, errors


# ---------------------------------------------------------------------------
# Stage 2 validation
# ---------------------------------------------------------------------------


def validate_stage2_output(raw: Any, stage1_observations: list[dict], pkt_evidence: dict) -> tuple[dict, list[str], list[str]]:
    try:
        return _validate_stage2_output_impl(raw, stage1_observations, pkt_evidence)
    except Exception as e:  # noqa: BLE001
        return {}, [f"Stage2 validator crashed on malformed input: {type(e).__name__}: {e}"], []


def _as_str_list_safe(value: Any, field_name: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"'{field_name}' must be a list, got {type(value).__name__}: {value!r}")
        return []
    out = []
    for item in value:
        if not isinstance(item, str):
            errors.append(f"'{field_name}' must contain only strings, found {type(item).__name__}: {item!r}")
            continue
        out.append(item)
    return list(dict.fromkeys(out))


def _validate_stage2_output_impl(raw: Any, stage1_observations: list[dict], pkt_evidence: dict) -> tuple[dict, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw, dict):
        return {}, [f"Stage2 output must be a JSON object, got {type(raw)}"], []

    required_top = {
        "decision", "evidence_quality", "hypothesis_type", "hypothesis",
        "primary_evidence_ids", "secondary_evidence_ids", "counter_evidence_ids",
        "system_limitations", "bull_thesis", "bear_thesis", "invalidation",
        "decision_reason", "entry", "failure_exit",
    }
    extra = set(raw.keys()) - required_top
    missing = required_top - set(raw.keys())
    if extra:
        errors.append(f"Stage2 output has unexpected top-level key(s): {sorted(extra)}")
    if missing:
        errors.append(f"Stage2 output missing key(s): {sorted(missing)}")
    if errors:
        return {}, errors, []

    entry = raw.get("entry")
    failure_exit = raw.get("failure_exit")
    if not isinstance(entry, dict) or set(entry.keys()) != {"status", "ideal_low", "ideal_high"}:
        errors.append("'entry' must be an object with exactly status/ideal_low/ideal_high")
        entry = {"status": None, "ideal_low": None, "ideal_high": None}
    if not isinstance(failure_exit, dict) or set(failure_exit.keys()) != {"exit_price", "reason", "trigger_type"}:
        errors.append("'failure_exit' must be an object with exactly exit_price/reason/trigger_type")
        failure_exit = {"exit_price": None, "reason": None, "trigger_type": None}

    for str_field in ("decision", "evidence_quality", "hypothesis_type", "hypothesis",
                       "bull_thesis", "bear_thesis", "invalidation", "decision_reason"):
        if not isinstance(raw.get(str_field), str):
            errors.append(f"'{str_field}' must be a string, got {type(raw.get(str_field)).__name__}")

    normalized = {
        "decision": raw.get("decision", "").strip().upper() if isinstance(raw.get("decision"), str) else "",
        "evidence_quality": raw.get("evidence_quality", "").strip().upper() if isinstance(raw.get("evidence_quality"), str) else "",
        "hypothesis_type": raw.get("hypothesis_type", "").strip() if isinstance(raw.get("hypothesis_type"), str) else "",
        "hypothesis": raw.get("hypothesis", "").strip() if isinstance(raw.get("hypothesis"), str) else "",
        "primary_evidence_ids": _as_str_list_safe(raw.get("primary_evidence_ids"), "primary_evidence_ids", errors),
        "secondary_evidence_ids": _as_str_list_safe(raw.get("secondary_evidence_ids"), "secondary_evidence_ids", errors),
        "counter_evidence_ids": _as_str_list_safe(raw.get("counter_evidence_ids"), "counter_evidence_ids", errors),
        "system_limitations": _as_str_list_safe(raw.get("system_limitations"), "system_limitations", errors),
        "bull_thesis": raw.get("bull_thesis", "").strip() if isinstance(raw.get("bull_thesis"), str) else "",
        "bear_thesis": raw.get("bear_thesis", "").strip() if isinstance(raw.get("bear_thesis"), str) else "",
        "invalidation": raw.get("invalidation", "").strip() if isinstance(raw.get("invalidation"), str) else "",
        "decision_reason": raw.get("decision_reason", "").strip() if isinstance(raw.get("decision_reason"), str) else "",
        "entry": {
            "status": entry.get("status", "").strip().upper() if isinstance(entry.get("status"), str) else entry.get("status"),
            "ideal_low": entry.get("ideal_low"),
            "ideal_high": entry.get("ideal_high"),
        },
        "failure_exit": {
            "exit_price": failure_exit.get("exit_price"),
            "reason": failure_exit.get("reason"),
            "trigger_type": (
                failure_exit.get("trigger_type", "").strip().upper()
                if isinstance(failure_exit.get("trigger_type"), str) else failure_exit.get("trigger_type")
            ),
        },
    }

    decision = normalized["decision"]
    evidence_quality = normalized["evidence_quality"]
    if decision not in DECISION_VALUES:
        errors.append(f"decision={decision!r} not in {sorted(DECISION_VALUES)}")
    if evidence_quality not in EVIDENCE_QUALITY_VALUES:
        errors.append(f"evidence_quality={evidence_quality!r} not in {sorted(EVIDENCE_QUALITY_VALUES)}")

    obs_by_id = {o["evidence_id"]: o for o in stage1_observations if isinstance(o, dict) and isinstance(o.get("evidence_id"), str)}
    supportive_or_mixed = {eid for eid, o in obs_by_id.items() if o.get("direction") in ("SUPPORTIVE", "MIXED")}
    contradictory_or_mixed = {eid for eid, o in obs_by_id.items() if o.get("direction") in ("CONTRADICTORY", "MIXED")}

    for fld in ("primary_evidence_ids", "secondary_evidence_ids"):
        bad = set(normalized[fld]) - supportive_or_mixed
        if bad:
            errors.append(f"{fld} contains id(s) not SUPPORTIVE/MIXED per Stage1: {sorted(bad)}")
    bad_counter = set(normalized["counter_evidence_ids"]) - contradictory_or_mixed
    if bad_counter:
        errors.append(f"counter_evidence_ids contains id(s) not CONTRADICTORY/MIXED per Stage1: {sorted(bad_counter)}")

    evidence_sets = derive_case_evidence_sets(pkt_evidence)
    expected_system_unavailable = set(evidence_sets["system_unavailable_evidence_ids"])
    if set(normalized["system_limitations"]) != expected_system_unavailable:
        errors.append(
            f"system_limitations does not match system_unavailable_evidence_ids. "
            f"Expected {sorted(expected_system_unavailable)}, got {sorted(normalized['system_limitations'])}"
        )

    bear_text_lower = normalized["bear_thesis"].lower()
    for eid in contradictory_or_mixed:
        if _text_mentions_family(bear_text_lower, eid) and eid not in normalized["counter_evidence_ids"]:
            errors.append(
                f"bear_thesis references evidence family '{eid}' (Stage1 direction={obs_by_id[eid]['direction']}) "
                f"but counter_evidence_ids does not include it."
            )
    if decision in NON_ACTIONABLE_DECISIONS and not normalized["counter_evidence_ids"]:
        if any(kw in bear_text_lower for kw in ("risk", "concern", "weak", "negative", "風險", "疑慮", "轉弱", "負向", "下滑")):
            warnings.append(
                "bear_thesis uses negative-sounding language but counter_evidence_ids is empty and no "
                "specific CONTRADICTORY/MIXED family was matched by name -- recommend human review."
            )

    for eid, obs in obs_by_id.items():
        if obs.get("benchmark_available") is False:
            for fld in ("bull_thesis", "bear_thesis", "decision_reason"):
                text_lower = normalized[fld].lower()
                if _text_mentions_family(text_lower, eid):
                    hits = _text_mentions_any_phrase(normalized[fld], VALUE_JUDGMENT_PHRASES)
                    if hits:
                        errors.append(f"{fld} makes a value judgment ({hits}) about '{eid}', which has benchmark_available=false.")

    for eid, obs in obs_by_id.items():
        has_single_period_caveat = any(
            "single-period" in lim.lower() or "single period" in lim.lower() or "單期" in lim
            for lim in obs.get("limitations", [])
        )
        if has_single_period_caveat:
            for fld in ("bull_thesis", "decision_reason"):
                text_lower = normalized[fld].lower()
                if _text_mentions_family(text_lower, eid):
                    hits = _text_mentions_any_phrase(normalized[fld], DURATIVE_TREND_PHRASES)
                    if hits:
                        errors.append(f"{fld} describes '{eid}' with durative/trend language ({hits}), but Stage1 flagged it as single-period.")

    for eid, obs in obs_by_id.items():
        has_routine_caveat = any(
            "routine" in lim.lower() or "periodic" in lim.lower() or "例行" in lim
            for lim in obs.get("limitations", [])
        )
        if has_routine_caveat:
            for fld in ("bull_thesis", "decision_reason"):
                text_lower = normalized[fld].lower()
                if _text_mentions_family(text_lower, eid):
                    hits = _text_mentions_any_phrase(normalized[fld], CATALYST_PHRASES)
                    if hits:
                        errors.append(f"{fld} describes '{eid}' as a catalyst ({hits}), but Stage1 flagged it as routine/periodic.")

    for eid, obs in obs_by_id.items():
        if obs.get("direction") == "MIXED":
            for fld in ("bull_thesis", "bear_thesis"):
                text_lower = normalized[fld].lower()
                if _text_mentions_family(text_lower, eid):
                    if any(kw in text_lower for kw in ("clearly positive", "uniformly positive", "全面正向", "全面看多", "clearly negative", "uniformly negative", "全面負向", "全面看空")):
                        warnings.append(f"{fld} describes MIXED-direction family '{eid}' in uniform terms.")

    if normalized["entry"]["status"] not in ENTRY_STATUS_VALUES:
        errors.append(f"entry.status={normalized['entry']['status']!r} not in {sorted(ENTRY_STATUS_VALUES)}")
    if normalized["failure_exit"]["trigger_type"] not in FAILURE_TRIGGER_VALUES:
        errors.append(f"failure_exit.trigger_type={normalized['failure_exit']['trigger_type']!r} not in {sorted(FAILURE_TRIGGER_VALUES)}")

    if decision in NON_ACTIONABLE_DECISIONS:
        if normalized["entry"]["status"] != "NOT_ACTIONABLE":
            errors.append(f"decision={decision} requires entry.status == 'NOT_ACTIONABLE'")
        if normalized["entry"]["ideal_low"] is not None or normalized["entry"]["ideal_high"] is not None:
            errors.append(f"decision={decision} requires entry.ideal_low/ideal_high == null")
        if normalized["failure_exit"]["exit_price"] is not None:
            errors.append(f"decision={decision} requires failure_exit.exit_price == null")
        if normalized["failure_exit"]["trigger_type"] != "UNAVAILABLE":
            errors.append(f"decision={decision} requires failure_exit.trigger_type == 'UNAVAILABLE'")

    if decision in ACTIONABLE_DECISIONS:
        if normalized["entry"]["status"] == "NOT_ACTIONABLE":
            errors.append(f"decision={decision} must not have entry.status == 'NOT_ACTIONABLE'")
        il, ih = normalized["entry"]["ideal_low"], normalized["entry"]["ideal_high"]
        price_family = obs_by_id.get("price_volume_structure")
        raw_price_numbers: list[float] = []
        if price_family:
            for txt in price_family.get("exact_observations", []):
                raw_price_numbers.extend(_extract_numbers(txt))
        if il is None or ih is None:
            errors.append(f"decision={decision} requires non-null entry.ideal_low/ideal_high")
        elif isinstance(il, bool) or isinstance(ih, bool) or not isinstance(il, (int, float)) or not isinstance(ih, (int, float)):
            errors.append("entry.ideal_low/ideal_high must be numeric for an actionable decision")
        else:
            if il > ih:
                errors.append(f"entry.ideal_low ({il}) must be <= entry.ideal_high ({ih})")
            if raw_price_numbers:
                lo, hi = min(raw_price_numbers) * 0.5, max(raw_price_numbers) * 2.0
                if not (lo <= il <= hi) or not (lo <= ih <= hi):
                    errors.append(
                        f"entry.ideal_low/ideal_high ({il}/{ih}) is not within a plausible range of the "
                        f"price_volume_structure numbers Stage1 observed ({raw_price_numbers})."
                    )
        exit_price = normalized["failure_exit"]["exit_price"]
        if exit_price is None:
            errors.append(f"decision={decision} requires non-null failure_exit.exit_price")
        elif isinstance(exit_price, bool) or not isinstance(exit_price, (int, float)):
            errors.append("failure_exit.exit_price must be numeric for an actionable decision")
        elif raw_price_numbers:
            lo, hi = min(raw_price_numbers) * 0.5, max(raw_price_numbers) * 2.0
            if not (lo <= exit_price <= hi):
                errors.append(f"failure_exit.exit_price ({exit_price}) is not within a plausible range of {raw_price_numbers}.")
        if normalized["failure_exit"]["trigger_type"] == "UNAVAILABLE":
            errors.append(f"decision={decision} requires a real failure_exit.trigger_type")
        if not normalized["failure_exit"]["reason"]:
            errors.append(f"decision={decision} requires a non-empty failure_exit.reason")

    for text_field in ("hypothesis", "bull_thesis", "bear_thesis", "invalidation", "decision_reason"):
        if not normalized[text_field]:
            errors.append(f"'{text_field}' must not be empty")

    return normalized, errors, warnings


# ---------------------------------------------------------------------------
# Stage 3 validation
# ---------------------------------------------------------------------------


def validate_stage3_output(raw: Any, stage1_evidence_ids: list[str] | None = None) -> tuple[dict, list[str]]:
    try:
        return _validate_stage3_output_impl(raw, stage1_evidence_ids or [])
    except Exception as e:  # noqa: BLE001
        return {}, [f"Stage3 validator crashed on malformed input: {type(e).__name__}: {e}"]


def _validate_stage3_output_impl(raw: Any, stage1_evidence_ids: list[str]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return {}, [f"Stage3 output must be a JSON object, got {type(raw)}"]

    extra = set(raw.keys()) - {"verdict", "problems"}
    missing = {"verdict", "problems"} - set(raw.keys())
    if extra:
        errors.append(f"Stage3 output has unexpected top-level key(s): {sorted(extra)}")
    if missing:
        errors.append(f"Stage3 output missing key(s): {sorted(missing)}")
    if errors:
        return {}, errors

    verdict = raw["verdict"]
    if verdict not in CRITIC_VERDICT_VALUES:
        errors.append(f"verdict={verdict!r} not in {sorted(CRITIC_VERDICT_VALUES)}")

    problems = raw["problems"]
    if not isinstance(problems, list):
        return {}, [f"'problems' must be a list, got {type(problems).__name__}"]

    normalized_problems = []
    for i, p in enumerate(problems):
        prefix = f"problems[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{prefix} must be an object, got {type(p).__name__}")
            continue
        if set(p.keys()) != CRITIC_PROBLEM_REQUIRED_KEYS:
            errors.append(f"{prefix} keys must be exactly {sorted(CRITIC_PROBLEM_REQUIRED_KEYS)}, got {sorted(p.keys())}")
            continue

        field_val, claim_text, cited_evidence_id = p["field"], p["claim_text"], p["cited_evidence_id"]
        problem_type, explanation = p["problem_type"], p["explanation"]
        ok = True

        if field_val not in CRITIC_PROBLEM_FIELD_VALUES:
            errors.append(f"{prefix}.field={field_val!r} not in {sorted(CRITIC_PROBLEM_FIELD_VALUES)}")
            ok = False
        if not isinstance(claim_text, str) or not claim_text.strip():
            errors.append(f"{prefix}.claim_text must be a non-empty string, got {claim_text!r}")
            ok = False
        if cited_evidence_id is not None:
            if not isinstance(cited_evidence_id, str):
                errors.append(f"{prefix}.cited_evidence_id must be null or a string, got {type(cited_evidence_id).__name__}: {cited_evidence_id!r}")
                ok = False
            elif stage1_evidence_ids and cited_evidence_id not in stage1_evidence_ids:
                errors.append(f"{prefix}.cited_evidence_id={cited_evidence_id!r} is not one of Stage1's evidence_ids {stage1_evidence_ids}")
                ok = False
        if problem_type not in CRITIC_PROBLEM_TYPES:
            errors.append(f"{prefix}.problem_type={problem_type!r} not in {sorted(CRITIC_PROBLEM_TYPES)}")
            ok = False
        if not isinstance(explanation, str) or not explanation.strip():
            errors.append(f"{prefix}.explanation must be a non-empty string, got {explanation!r}")
            ok = False

        if ok:
            normalized_problems.append(p)

    if verdict == "APPROVE" and normalized_problems:
        errors.append("verdict=APPROVE but problems is non-empty; these are contradictory")
    if verdict == "REVISE" and not normalized_problems and not errors:
        errors.append("verdict=REVISE but problems is empty (or none were valid); must list at least one concrete problem")

    return {"verdict": verdict, "problems": normalized_problems}, errors
