"""
test_three_stage_contract.py

Synthetic contract tests, rewritten after real-model canary runs
35482706294 / 35485028854 exposed real bugs no synthetic test had caught
(the schema wasn't being sent to Ollama at all in the first run; once
fixed, Stage2's single combined contract turned out to allow
decision=CANDIDATE + entry.status=NOT_ACTIONABLE + null prices). This
file tests the resulting architecture change: Stage2 is now two separate
validators (validate_stage2_decision_output / validate_stage2_
actionability_output), plus the alias-matching fix, the unified valuation
rule, the new evidence-role hard rules, and the repair-request builder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_capability import SYSTEM_AVAILABLE_FAMILIES, SYSTEM_UNAVAILABLE_FAMILIES
from deterministic_validators import (
    ENTRY_STATUS_ACTIONABLE_VALUES,
    FAILURE_TRIGGER_ACTIONABLE_VALUES,
    validate_stage1_output,
    validate_stage2_actionability_output,
    validate_stage2_decision_output,
    validate_stage3_output,
    errors_to_repair_request,
    _text_mentions_family,
)
from safe_finalizer import assemble_actionable_stage2, assemble_non_actionable_stage2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_pkt(available=None):
    available = available or list(SYSTEM_AVAILABLE_FAMILIES)
    missing = (set(SYSTEM_AVAILABLE_FAMILIES) - set(available)) | set(SYSTEM_UNAVAILABLE_FAMILIES)
    return {
        "case_id": 9001,
        "decision_date": 20240610,
        "code": "TEST1",
        "evidence": {
            "available_evidence_ids": sorted(available),
            "missing_evidence_ids": sorted(missing),
        },
    }


def obs(evidence_id, exact_observations, direction, timeliness="CURRENT",
        relevance="HIGH", limitations=None, benchmark_available=True):
    return {
        "evidence_id": evidence_id,
        "exact_observations": exact_observations,
        "direction": direction,
        "timeliness": timeliness,
        "relevance_to_hypothesis_space": relevance,
        "limitations": limitations or [],
        "benchmark_available": benchmark_available,
    }


def base_decision(decision="WATCH", **overrides):
    d = {
        "decision": decision,
        "evidence_quality": "MODERATE",
        "hypothesis_type": "test",
        "hypothesis": "test hypothesis",
        "primary_evidence_ids": ["revenue"],
        "secondary_evidence_ids": [],
        "counter_evidence_ids": [],
        "system_limitations": sorted(SYSTEM_UNAVAILABLE_FAMILIES),
        "bull_thesis": "revenue supports the thesis.",
        "bear_thesis": "no material counter evidence identified.",
        "invalidation": "revenue reverses.",
        "decision_reason": "revenue supports the thesis with adequate strength.",
    }
    d.update(overrides)
    return d


def base_actionability(**overrides):
    d = {
        "entry": {"status": "NOW", "ideal_low": 95.0, "ideal_high": 105.0},
        "failure_exit": {"exit_price": 88.0, "reason": "support break", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Alias-matching fix (review point 2)
# ---------------------------------------------------------------------------


def test_reasonable_valuation_does_not_false_positive_on_other_families():
    """Reproduces the exact reported bug: 'reasonable valuation' was
    detected as mentioning institutional_flow, mops_material_information,
    price_volume_structure, AND revenue, via generic single-word matching
    ('flow', 'information', 'structure', 'price'). None of those should
    match now."""
    text = "reasonable valuation given the current numbers".lower()
    assert _text_mentions_family(text, "valuation") is True
    assert _text_mentions_family(text, "institutional_flow") is False
    assert _text_mentions_family(text, "mops_material_information") is False
    assert _text_mentions_family(text, "price_volume_structure") is False
    assert _text_mentions_family(text, "revenue") is False


def test_generic_words_alone_never_match_any_family():
    for word in ("information", "structure", "flow", "price", "revenue growth potential"):
        text = f"the {word} looks fine".lower()
        # only revenue-related text containing the actual alias "revenue" should match revenue
        if "revenue" in word:
            assert _text_mentions_family(text, "revenue") is True
        else:
            assert _text_mentions_family(text, "institutional_flow") is False
            assert _text_mentions_family(text, "mops_material_information") is False
            assert _text_mentions_family(text, "price_volume_structure") is False


def test_alias_matching_still_detects_real_mentions():
    assert _text_mentions_family("foreign investor flow turned negative".lower(), "institutional_flow") is True
    assert _text_mentions_family("this mops filing is routine".lower(), "mops_material_information") is True
    assert _text_mentions_family("technical structure shows a breakdown".lower(), "price_volume_structure") is True
    assert _text_mentions_family("monthly revenue jumped".lower(), "revenue") is True


def test_value_judgment_regression_no_longer_over_triggers_across_families():
    """The false-positive alias bug meant a value-judgment error about
    valuation used to also (spuriously) fire against unrelated families
    whose text happened to share a generic word. Confirm decision output
    referencing only valuation's value judgment produces errors scoped to
    valuation, not phantom errors about other families."""
    pkt = make_pkt()
    stage1_obs = [
        obs("valuation", ["PE = 20"], direction="NOT_INTERPRETABLE", benchmark_available=False,
            limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"]),
        obs("institutional_flow", ["foreign flow +0.3%"], direction="SUPPORTIVE"),
    ]
    decision = base_decision(
        decision="WATCH", primary_evidence_ids=["institutional_flow"],
        bull_thesis="reasonable valuation and positive institutional flow support this.",
    )
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    valuation_errors = [e for e in errors if "'valuation'" in e]
    assert valuation_errors, "should still catch the real valuation value-judgment violation"
    # And it should NOT also spuriously complain about institutional_flow's benchmark (it has none stated, not false)
    assert not any("institutional_flow'" in e and "benchmark_available=false" in e for e in errors)


# ---------------------------------------------------------------------------
# Unified valuation rule (review point 3)
# ---------------------------------------------------------------------------


def test_stage1_valuation_without_benchmark_must_be_not_interpretable():
    pkt = make_pkt()
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 215.37"], direction="CONTRADICTORY", benchmark_available=False,
                limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"])
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert any("direction must be 'NOT_INTERPRETABLE'" in e for e in errors), errors


def test_stage1_valuation_without_benchmark_supportive_also_rejected():
    pkt = make_pkt()
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 8.0"], direction="SUPPORTIVE", benchmark_available=False,
                limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"])
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert any("direction must be 'NOT_INTERPRETABLE'" in e for e in errors), errors


def test_stage1_valuation_without_benchmark_not_interpretable_passes():
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 215.37"], direction="NOT_INTERPRETABLE", benchmark_available=False,
                limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"])
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert errors == [], errors


def test_stage1_valuation_with_benchmark_can_be_supportive_or_contradictory():
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 8.0, peer median PE = 20.0"], direction="SUPPORTIVE", benchmark_available=True)
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert errors == [], errors


# ---------------------------------------------------------------------------
# Evidence-role hard rules (review point 4)
# ---------------------------------------------------------------------------


def test_contradictory_family_must_appear_in_counter_evidence_ids():
    pkt = make_pkt()
    stage1_obs = [
        obs("revenue", ["revenue -15% YoY"], direction="CONTRADICTORY"),
        obs("price_volume_structure", ["ret20 +5%"], direction="SUPPORTIVE"),
    ]
    decision = base_decision(
        decision="WATCH", primary_evidence_ids=["price_volume_structure"],
        counter_evidence_ids=[],  # omits the CONTRADICTORY revenue family
    )
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert any("missing required CONTRADICTORY family" in e for e in errors), errors


def test_contradictory_family_correctly_listed_passes_that_check():
    pkt = make_pkt()
    stage1_obs = [
        obs("revenue", ["revenue -15% YoY"], direction="CONTRADICTORY"),
        obs("price_volume_structure", ["ret20 +5%"], direction="SUPPORTIVE"),
    ]
    decision = base_decision(
        decision="WATCH", primary_evidence_ids=["price_volume_structure"],
        counter_evidence_ids=["revenue"],
    )
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert not any("missing required CONTRADICTORY family" in e for e in errors), errors


def test_neutral_family_forbidden_in_any_role():
    pkt = make_pkt()
    stage1_obs = [
        obs("revenue", ["flat revenue"], direction="NEUTRAL"),
        obs("price_volume_structure", ["ret20 +5%"], direction="SUPPORTIVE"),
    ]
    decision = base_decision(decision="WATCH", primary_evidence_ids=["price_volume_structure", "revenue"])
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert any("NEUTRAL or NOT_INTERPRETABLE" in e for e in errors), errors


def test_not_interpretable_family_forbidden_in_any_role():
    pkt = make_pkt()
    stage1_obs = [
        obs("valuation", ["PE=20"], direction="NOT_INTERPRETABLE", benchmark_available=False,
            limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"]),
        obs("price_volume_structure", ["ret20 +5%"], direction="SUPPORTIVE"),
    ]
    decision = base_decision(decision="WATCH", secondary_evidence_ids=["valuation"], primary_evidence_ids=["price_volume_structure"])
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert any("NEUTRAL or NOT_INTERPRETABLE" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Stage2 decision-only contract (no entry/failure_exit fields accepted)
# ---------------------------------------------------------------------------


def test_stage2_decision_output_rejects_entry_field_entirely():
    """The core structural fix: this call must not even accept an entry/
    failure_exit field, let alone validate its content."""
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["+10% YoY"], direction="SUPPORTIVE")]
    decision = base_decision(decision="CANDIDATE")
    decision["entry"] = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}  # must be rejected
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert any("unexpected top-level key" in e and "entry" in e for e in errors), errors


def test_stage2_decision_output_valid_candidate_has_no_price_fields():
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["+10% YoY"], direction="SUPPORTIVE")]
    decision = base_decision(decision="CANDIDATE")
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert errors == [], errors


def test_actionable_or_watch_decision_requires_primary_support():
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["revenue YoY = 10"], direction="SUPPORTIVE")]
    for decision_name in ("WATCH", "CANDIDATE", "HIGH_CONVICTION"):
        decision = base_decision(decision=decision_name, primary_evidence_ids=[], secondary_evidence_ids=[])
        normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
        assert any("requires at least one primary_evidence_id" in e for e in errors), (decision_name, errors)


def test_high_conviction_requires_two_distinct_supportive_families():
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["revenue YoY = 10"], direction="SUPPORTIVE")]
    decision = base_decision(decision="HIGH_CONVICTION", primary_evidence_ids=["revenue"], secondary_evidence_ids=[])
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert any("two distinct" in e for e in errors), errors


def test_system_unavailable_family_cannot_appear_in_decision_prose():
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["revenue YoY = 10"], direction="SUPPORTIVE")]
    decision = base_decision(
        decision="WATCH",
        decision_reason="The revenue evidence is supportive, but analyst consensus is unavailable.",
    )
    normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
    assert any("system-unavailable" in e and "analyst_consensus" in e for e in errors), errors
    assert "entry" not in normalized
    assert "failure_exit" not in normalized


def test_stage2_decision_malformed_evidence_id_field_types_do_not_crash():
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["+10%"], direction="SUPPORTIVE")]
    for bad_value in (123, {"not": "a list"}, "a bare string"):
        decision = base_decision(primary_evidence_ids=bad_value)
        normalized, errors, warnings = validate_stage2_decision_output(decision, stage1_obs, pkt["evidence"])
        assert errors, f"bad_value={bad_value!r} should have produced errors"
        assert any("primary_evidence_ids" in e for e in errors), errors


def test_stage2_decision_output_not_a_dict_does_not_crash():
    pkt = make_pkt()
    stage1_obs = [obs("revenue", ["+10%"], direction="SUPPORTIVE")]
    for bad_raw in (None, "a string", 123, ["a", "list"]):
        normalized, errors, warnings = validate_stage2_decision_output(bad_raw, stage1_obs, pkt["evidence"])
        assert errors
        assert normalized == {}


# ---------------------------------------------------------------------------
# Stage2 actionability-only contract (the other half of the structural fix)
# ---------------------------------------------------------------------------


def test_actionability_schema_structurally_excludes_not_actionable():
    """entry.status=NOT_ACTIONABLE is not even in this validator's allowed
    enum -- reproducing that exact combination through this call is
    impossible, not merely discouraged."""
    assert "NOT_ACTIONABLE" not in ENTRY_STATUS_ACTIONABLE_VALUES
    assert "UNAVAILABLE" not in FAILURE_TRIGGER_ACTIONABLE_VALUES

    stage1_obs = [obs("price_volume_structure", ["current price = 100.0"], direction="SUPPORTIVE")]
    raw = base_actionability()
    raw["entry"]["status"] = "NOT_ACTIONABLE"
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs)
    assert any("not in" in e and "entry.status" in e for e in errors), errors


def test_actionability_schema_structurally_excludes_unavailable_trigger():
    stage1_obs = [obs("price_volume_structure", ["current price = 100.0"], direction="SUPPORTIVE")]
    raw = base_actionability()
    raw["failure_exit"]["trigger_type"] = "UNAVAILABLE"
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs)
    assert any("failure_exit.trigger_type" in e for e in errors), errors


def test_actionability_valid_response_passes():
    stage1_obs = [obs("price_volume_structure", ["current price = 100.0", "ret20 = +5.0%"], direction="SUPPORTIVE")]
    raw = base_actionability()
    packet_evidence = {"price_volume_structure": {
        "current_price": 100.0, "support_low20": 88.0, "high20": 105.0, "atr20": 2.0,
    }}
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs, packet_evidence)
    assert errors == [], errors
    assert normalized["entry"]["status"] == "NOW"


def test_actionability_price_not_grounded_is_rejected():
    stage1_obs = [obs("price_volume_structure", ["current price = 100.0"], direction="SUPPORTIVE")]
    raw = base_actionability(entry={"status": "NOW", "ideal_low": 5000.0, "ideal_high": 5100.0})
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs)
    assert any("not within a plausible range" in e for e in errors), errors


def test_actionability_uses_named_packet_price_anchors_not_returns_or_lookbacks():
    stage1_obs = [obs(
        "price_volume_structure",
        ["current_price = 100", "ret60_pct = 95", "support_low60 = 80", "high60 = 110"],
        direction="SUPPORTIVE",
    )]
    packet_evidence = {"price_volume_structure": {
        "current_price": 100.0, "ret60_pct": 95.0, "support_low60": 80.0,
        "high60": 110.0, "atr20": 2.0,
    }}
    raw = base_actionability(entry={"status": "NOW", "ideal_low": 50.0, "ideal_high": 55.0})
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs, packet_evidence)
    assert any("packet price anchors" in e for e in errors), errors


def test_actionability_failure_exit_must_be_below_entry_low_for_long_only():
    stage1_obs = [obs("price_volume_structure", ["current_price = 100"], direction="SUPPORTIVE")]
    packet_evidence = {"price_volume_structure": {
        "current_price": 100.0, "support_low20": 90.0, "high20": 110.0, "atr20": 2.0,
    }}
    raw = base_actionability(entry={"status": "NOW", "ideal_low": 95.0, "ideal_high": 100.0})
    raw["failure_exit"]["exit_price"] = 96.0
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs, packet_evidence)
    assert any("must be below entry.ideal_low" in e for e in errors), errors


def test_stage1_grounding_ignores_lookback_digits_embedded_in_field_names():
    pkt = make_pkt()
    pkt["evidence"]["price_volume_structure"] = {"ret5_pct": 1.25, "ret20_pct": -2.5}
    raw = {
        "case_id": pkt["case_id"],
        "evidence_observations": [
            obs("price_volume_structure", ["ret5_pct = 1.25", "ret20_pct = -2.5"], direction="MIXED")
        ],
    }
    normalized, errors = validate_stage1_output(
        raw, pkt["case_id"], ["price_volume_structure"], pkt["evidence"]
    )
    assert errors == [], errors


def test_stage1_valuation_benchmark_flag_is_derived_from_packet_not_model():
    pkt = make_pkt()
    pkt["evidence"]["valuation"] = {"pe": 20.0, "pb": 2.0}
    raw = {
        "case_id": pkt["case_id"],
        "evidence_observations": [
            obs("valuation", ["PE = 20.0"], direction="SUPPORTIVE", benchmark_available=True)
        ],
    }
    normalized, errors = validate_stage1_output(raw, pkt["case_id"], ["valuation"], pkt["evidence"])
    assert any("contradicts the raw valuation packet" in e for e in errors), errors


def test_actionability_null_prices_rejected():
    """Reproduces the reported failure mode's OTHER half: even if status
    were somehow actionable, null prices must still be rejected here."""
    stage1_obs = [obs("price_volume_structure", ["current price = 100.0"], direction="SUPPORTIVE")]
    raw = {"entry": {"status": "NOW", "ideal_low": None, "ideal_high": None},
           "failure_exit": {"exit_price": None, "reason": "x", "trigger_type": "PRICE_STRUCTURE_BREAK"}}
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs)
    assert any("non-null" in e for e in errors), errors


def test_actionability_output_not_a_dict_does_not_crash():
    stage1_obs = [obs("price_volume_structure", ["100.0"], direction="SUPPORTIVE")]
    for bad_raw in (None, "x", 123, [1, 2]):
        normalized, errors = validate_stage2_actionability_output(bad_raw, stage1_obs)
        assert errors
        assert normalized == {}


def test_actionability_malformed_types_do_not_crash():
    stage1_obs = [obs("price_volume_structure", ["100.0"], direction="SUPPORTIVE")]
    raw = {"entry": {"status": "NOW", "ideal_low": "not a number", "ideal_high": [1, 2]},
           "failure_exit": {"exit_price": {"nested": "dict"}, "reason": 123, "trigger_type": "PRICE_STRUCTURE_BREAK"}}
    normalized, errors = validate_stage2_actionability_output(raw, stage1_obs)
    assert errors  # should be rejected, not crash


# ---------------------------------------------------------------------------
# Deterministic assembly (safe_finalizer.py)
# ---------------------------------------------------------------------------


def test_assemble_non_actionable_never_calls_model_and_sets_canonical_nulls():
    decision_output = base_decision(decision="WATCH")
    del decision_output  # not actually needed by the function besides decision key check below
    decision_output = base_decision(decision="REJECT")
    result = assemble_non_actionable_stage2(decision_output)
    assert result.combined["entry"] == {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
    assert result.combined["failure_exit"] == {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}
    assert result.combined["decision"] == "REJECT"


def test_assemble_non_actionable_refuses_actionable_decision():
    decision_output = base_decision(decision="CANDIDATE")
    raised = False
    try:
        assemble_non_actionable_stage2(decision_output)
    except ValueError:
        raised = True
    assert raised


def test_assemble_actionable_combines_decision_and_actionability():
    decision_output = base_decision(decision="CANDIDATE")
    actionability_output = {"entry": {"status": "NOW", "ideal_low": 90.0, "ideal_high": 95.0},
                             "failure_exit": {"exit_price": 80.0, "reason": "x", "trigger_type": "PRICE_STRUCTURE_BREAK"}}
    result = assemble_actionable_stage2(decision_output, actionability_output)
    assert result.combined["entry"]["ideal_low"] == 90.0
    assert result.combined["decision"] == "CANDIDATE"


def test_assemble_actionable_refuses_non_actionable_decision():
    decision_output = base_decision(decision="WATCH")
    actionability_output = {"entry": {"status": "NOW", "ideal_low": 90.0, "ideal_high": 95.0},
                             "failure_exit": {"exit_price": 80.0, "reason": "x", "trigger_type": "PRICE_STRUCTURE_BREAK"}}
    raised = False
    try:
        assemble_actionable_stage2(decision_output, actionability_output)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# Structured repair requests (review point 5)
# ---------------------------------------------------------------------------


def test_repair_request_extracts_enum_violation():
    errors = ["decision='MAYBE' not in ['CANDIDATE', 'HIGH_CONVICTION', 'REJECT', 'WATCH']"]
    repairs = errors_to_repair_request(errors)
    assert len(repairs) == 1
    assert repairs[0]["field"] == "decision"
    assert repairs[0]["problem"] == "invalid_enum_value"
    assert "MAYBE" in repairs[0]["received"]


def test_repair_request_extracts_missing_counter_evidence():
    errors = ["counter_evidence_ids is missing required CONTRADICTORY family/families (Stage1 marked these CONTRADICTORY; they must be listed, not silently dropped): ['revenue']"]
    repairs = errors_to_repair_request(errors)
    assert repairs[0]["field"] == "counter_evidence_ids"
    assert repairs[0]["problem"] == "missing_required_contradictory_family"
    assert "revenue" in repairs[0]["must_add"]


def test_repair_request_never_drops_unmatched_errors():
    errors = ["some totally novel error string never seen before"]
    repairs = errors_to_repair_request(errors)
    assert len(repairs) == 1
    assert repairs[0]["problem"] == errors[0]


def test_repair_request_handles_empty_list():
    assert errors_to_repair_request([]) == []


# ---------------------------------------------------------------------------
# Stage 3 (unchanged core behavior, still exercised here for completeness)
# ---------------------------------------------------------------------------


def test_stage3_approve_with_problems_is_invalid():
    raw = {"verdict": "APPROVE", "problems": [{
        "field": "bull_thesis", "claim_text": "x", "cited_evidence_id": None,
        "problem_type": "OVERSTATES_DIRECTION", "explanation": "y",
    }]}
    normalized, errors = validate_stage3_output(raw)
    assert errors


def test_stage3_revise_with_no_problems_is_invalid():
    raw = {"verdict": "REVISE", "problems": []}
    normalized, errors = validate_stage3_output(raw)
    assert errors


def test_stage3_entry_or_failure_exit_field_is_valid():
    """New field value added for the actionability-price critique path."""
    raw = {"verdict": "REVISE", "problems": [{
        "field": "entry_or_failure_exit", "claim_text": "entry zone looks disconnected from price data",
        "cited_evidence_id": "price_volume_structure", "problem_type": "ACTIONABILITY_PRICE_NOT_GROUNDED",
        "explanation": "the stated entry zone is far from the observed price range",
    }]}
    normalized, errors = validate_stage3_output(raw, ["price_volume_structure"])
    assert errors == [], errors


def test_stage3_wrong_case_id_type_rejected_upstream_in_stage1():
    stage1_raw = {"case_id": "9001", "evidence_observations": [obs("revenue", ["+5%"], direction="SUPPORTIVE")]}
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["revenue"])
    assert errors
