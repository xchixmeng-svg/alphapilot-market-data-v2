"""
test_three_stage_contract.py

Synthetic contract tests. NO LLM calls, no hidden outcomes, no real case
data. Every fixture here is invented purely to exercise the validators.

The five "Case N / ticker" tests each reproduce the STRUCTURE of a real
failure pattern reported from the outcome-blind canary runs (35456479322 /
35458136917 / 35459949583), using made-up numbers, to prove the new
validators catch that failure pattern deterministically. These are NOT
the actual tickers/numbers from the canary and must never be used to
special-case those specific stocks -- they exist to test the *pattern*
(extreme PE with no benchmark, routine filing misread as catalyst, empty
counter_evidence_ids despite a stated bear thesis, single-period spike
misread as trend, and the WATCH/REJECT null-field contract), not any
specific case_id.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from evidence_capability import SYSTEM_AVAILABLE_FAMILIES, SYSTEM_UNAVAILABLE_FAMILIES
from deterministic_validators import (
    validate_stage1_output,
    validate_stage2_output,
    validate_stage3_output,
)
from safe_finalizer import attempt_safe_finalize, finalize_watch_reject_nullfields


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


def base_stage2(decision="CANDIDATE", **overrides):
    d = {
        "decision": decision,
        "evidence_quality": "STRONG",
        "hypothesis_type": "TEST",
        "hypothesis": "test hypothesis",
        "primary_evidence_ids": [],
        "secondary_evidence_ids": [],
        "counter_evidence_ids": [],
        "system_limitations": sorted(SYSTEM_UNAVAILABLE_FAMILIES),
        "bull_thesis": "test bull thesis",
        "bear_thesis": "test bear thesis",
        "invalidation": "test invalidation",
        "decision_reason": "test decision reason",
        "entry": {"status": "NOW", "ideal_low": 90.0, "ideal_high": 95.0},
        "failure_exit": {"exit_price": 80.0, "reason": "structure break", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    }
    d.update(overrides)
    return d


def watch_stage2(**overrides):
    d = base_stage2(decision="WATCH")
    d["entry"] = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
    d["failure_exit"] = {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Case 14/8027 pattern: extreme PE called "reasonable" with no benchmark
# ---------------------------------------------------------------------------


def test_pattern_extreme_pe_called_reasonable_without_benchmark_is_rejected():
    pkt = make_pkt()
    stage1_obs = [
        obs("valuation", ["PE = 215.37", "PB = 3.66"], direction="CONTRADICTORY",
            limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"],
            benchmark_available=False),
    ]
    s2 = base_stage2(
        bull_thesis="Valuation is reasonable given the company's position.",
        decision_reason="Valuation is reasonable given the company's position.",
    )
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("BENCHMARK" not in e and "value judgment" in e for e in errors), errors
    assert not normalized == {} or errors  # sanity: normalized always returned


def test_stage1_itself_must_state_benchmark_caveat_for_valuation():
    """If Stage1 forgets to state the required no-benchmark caveat for
    valuation, that is itself a Stage1 contract violation -- catching the
    problem one stage earlier than Stage2's text-based check."""
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 215.37"], direction="CONTRADICTORY",
                limitations=[], benchmark_available=False)  # missing required caveat
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert any("no-benchmark caveat" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Case 16/2109 pattern: routine loan-to-affiliate MOPS misread as catalyst
# ---------------------------------------------------------------------------


def test_pattern_routine_filing_misread_as_catalyst_is_rejected():
    pkt = make_pkt()
    stage1_obs = [
        obs("mops_material_information",
            ["MOPS title: subsidiary ordinary-course loan to affiliate"],
            direction="NEUTRAL",
            limitations=["routine/periodic filing -- not evidence of a new catalyst"]),
    ]
    s2 = base_stage2(
        bull_thesis="The mops filing shows a positive catalyst demonstrating strong financial support capability.",
    )
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("routine/periodic" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Case 24/1604 pattern: bear thesis claims counter-evidence, but the list
# is empty; single-period MoM spike called sustained growth; valuation
# called cheap without a benchmark.
# ---------------------------------------------------------------------------


def test_pattern_empty_counter_evidence_despite_named_bear_family():
    pkt = make_pkt()
    stage1_obs = [
        obs("institutional_flow", ["foreign_previous = -0.20"], direction="CONTRADICTORY"),
    ]
    s2 = base_stage2(
        decision="WATCH",
        bear_thesis="institutional_flow has been negative recently, a concern for the thesis.",
        counter_evidence_ids=[],  # empty despite naming the family
    )
    s2["entry"] = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
    s2["failure_exit"] = {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("bear_thesis references evidence family 'institutional_flow'" in e for e in errors), errors


def test_pattern_single_month_mom_spike_called_sustained_growth():
    pkt = make_pkt()
    stage1_obs = [
        obs("revenue", ["revenue MoM = +64.31%", "revenue YoY = +2.32%"], direction="MIXED",
            limitations=["single-period change -- does not by itself establish a sustained trend"]),
    ]
    s2 = base_stage2(
        bull_thesis="revenue shows sustained growth this quarter, supporting the thesis.",
        primary_evidence_ids=["revenue"],
    )
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("durative/trend language" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Case 32/4904 pattern: "valuation reasonable" + "recent performance good"
# while price/flow evidence is actually negative and no benchmark exists.
# ---------------------------------------------------------------------------


def test_pattern_valuation_reasonable_claim_without_benchmark_case32_style():
    pkt = make_pkt()
    stage1_obs = [
        obs("valuation", ["PE = 25.87", "PB = 3.21"], direction="NOT_INTERPRETABLE",
            limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"],
            benchmark_available=False),
        obs("price_volume_structure", ["ret5 = -1.58%", "ret60 = -0.25%"], direction="CONTRADICTORY"),
    ]
    s2 = base_stage2(
        bull_thesis="Valuation is reasonable and recent price performance is good.",
        primary_evidence_ids=["price_volume_structure"],
    )
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("value judgment" in e for e in errors), errors
    # price_volume_structure is CONTRADICTORY, not SUPPORTIVE/MIXED -- it
    # cannot legally appear in primary_evidence_ids at all.
    assert any("not SUPPORTIVE/MIXED" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Case 15/4402 pattern: WATCH decision that repeatedly fails to null out
# entry/failure_exit -- exercise both the hard validator AND the safe
# finalizer's narrow allowed fix.
# ---------------------------------------------------------------------------


def test_pattern_watch_with_non_null_entry_is_rejected():
    pkt = make_pkt()
    stage1_obs = [obs("price_volume_structure", ["ret20 = +27.13%"], direction="MIXED")]
    s2 = base_stage2(decision="WATCH")  # base_stage2 defaults entry/failure_exit to actionable-shaped values
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("requires entry.status == 'NOT_ACTIONABLE'" in e for e in errors)
    assert any("requires entry.ideal_low/ideal_high == null" in e for e in errors)
    assert any("requires failure_exit.exit_price == null" in e for e in errors)
    assert any("requires failure_exit.trigger_type == 'UNAVAILABLE'" in e for e in errors)


def test_safe_finalizer_fixes_watch_nullfields_when_that_is_the_only_problem():
    """This is exactly the class of fix the finalizer IS allowed to make:
    decision stays WATCH (untouched), only the null-field shape is forced."""
    pkt = make_pkt()
    stage1_obs = [obs("price_volume_structure", ["ret20 = +27.13%"], direction="MIXED")]
    s2 = base_stage2(decision="WATCH")  # decision text/thesis all otherwise valid; only null-fields wrong
    result = attempt_safe_finalize(s2, stage1_obs, pkt["evidence"])
    assert result.is_finalizable, result.remaining_errors
    assert result.normalized["decision"] == "WATCH"  # decision itself never touched
    assert result.normalized["entry"]["status"] == "NOT_ACTIONABLE"
    assert result.normalized["entry"]["ideal_low"] is None
    assert result.normalized["failure_exit"]["exit_price"] is None
    assert result.normalized["failure_exit"]["trigger_type"] == "UNAVAILABLE"
    assert len(result.applied_fixes) == 4


def test_safe_finalizer_refuses_to_fix_when_other_errors_are_also_present():
    """The finalizer must NOT silently repair null-fields if there is ALSO
    a substantive error present (e.g. a benchmark-violation) -- it must
    leave everything untouched and force a real model revision instead."""
    pkt = make_pkt()
    stage1_obs = [
        obs("valuation", ["PE = 215.37"], direction="CONTRADICTORY",
            limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"],
            benchmark_available=False),
    ]
    s2 = base_stage2(
        decision="WATCH",
        bull_thesis="Valuation is reasonable.",  # substantive violation
    )
    result = attempt_safe_finalize(s2, stage1_obs, pkt["evidence"])
    assert not result.is_finalizable
    assert any("value judgment" in e for e in result.remaining_errors)
    # decision must be untouched and entry/failure_exit must NOT have been
    # silently forced, since this case was not eligible for the safe fix.
    assert result.normalized["decision"] == "WATCH"


def test_safe_finalizer_never_touches_actionable_decisions():
    pkt = make_pkt()
    stage1_obs = [obs("valuation", ["PE = 20"], direction="SUPPORTIVE", benchmark_available=True)]
    s2 = base_stage2(decision="CANDIDATE", primary_evidence_ids=["valuation"])
    s2["entry"] = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}  # wrong for CANDIDATE
    result = attempt_safe_finalize(s2, stage1_obs, pkt["evidence"])
    # This is a CANDIDATE with a null entry zone -- finalize_watch_reject_nullfields
    # only applies to REJECT/WATCH, so this must remain unfixed.
    assert not result.is_finalizable
    assert any("must not have entry.status" in e for e in result.remaining_errors)


# ---------------------------------------------------------------------------
# Stage 3 critic schema/consistency checks
# ---------------------------------------------------------------------------


def test_stage3_approve_with_problems_is_invalid():
    raw = {"verdict": "APPROVE", "problems": [{
        "field": "bull_thesis", "claim_text": "x", "cited_evidence_id": "valuation",
        "problem_type": "OVERSTATES_DIRECTION", "explanation": "y",
    }]}
    normalized, errors = validate_stage3_output(raw)
    assert any("contradictory" in e for e in errors)


def test_stage3_revise_with_no_problems_is_invalid():
    raw = {"verdict": "REVISE", "problems": []}
    normalized, errors = validate_stage3_output(raw)
    assert any("must list at least one" in e for e in errors)


def test_stage3_valid_revise_is_accepted():
    raw = {"verdict": "REVISE", "problems": [{
        "field": "bull_thesis", "claim_text": "Valuation is reasonable.",
        "cited_evidence_id": "valuation", "problem_type": "BENCHMARK_MISSING_BUT_VALUE_JUDGMENT_MADE",
        "explanation": "exact_observations only states PE=215.37 with benchmark_available=false; "
                       "no basis for 'reasonable'.",
    }]}
    normalized, errors = validate_stage3_output(raw)
    assert errors == []
    assert normalized["verdict"] == "REVISE"


# ---------------------------------------------------------------------------
# Evidence-id leakage across the system/case boundary (regression coverage
# for the original decision-collapse-prevention rules, re-verified here in
# the three-stage context)
# ---------------------------------------------------------------------------


def test_system_unavailable_cannot_appear_in_counter_evidence():
    pkt = make_pkt()
    stage1_obs = [obs("valuation", ["PE=20"], direction="SUPPORTIVE")]
    s2 = base_stage2(decision="WATCH", counter_evidence_ids=["eps_revisions"])
    s2["entry"] = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
    s2["failure_exit"] = {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("not CONTRADICTORY/MIXED per Stage1" in e for e in errors)


# ---------------------------------------------------------------------------
# NEW regression tests required by the external review
# ---------------------------------------------------------------------------


def test_stage1_wrong_case_id_is_rejected():
    """Reproduces the reported bug: a Stage1 response echoing a different
    case_id than the one requested used to pass validation with zero
    errors. It must now be rejected outright."""
    stage1_raw = {
        "case_id": 999,  # wrong -- caller expects 9001
        "evidence_observations": [
            obs("valuation", ["PE = 20"], direction="SUPPORTIVE", benchmark_available=True)
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert errors, "wrong case_id must be rejected, not silently accepted"
    assert any("does not match expected_case_id" in e for e in errors)
    assert normalized == {}


def test_stage1_case_id_wrong_type_is_rejected():
    """case_id='9001' (string) must not be accepted as equal to 9001 (int)."""
    stage1_raw = {
        "case_id": "9001",
        "evidence_observations": [obs("valuation", ["PE = 20"], direction="SUPPORTIVE")],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"])
    assert errors
    assert any("does not match expected_case_id" in e for e in errors)


def test_stage1_fabricated_number_is_rejected_by_grounding_check():
    """Reproduces the reported gap: Stage1 could previously invent a
    number in exact_observations and pass, because only 'is this a
    non-empty list of strings' was checked. A number with no match in the
    raw packet data for the SAME family must now be rejected."""
    pkt = make_pkt()
    pkt["evidence"]["valuation"] = {"pe": 25.4, "pb": 2.1}  # raw ground truth
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 999.9"], direction="CONTRADICTORY",  # fabricated, not 25.4
                limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"],
                benchmark_available=False)
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"], pkt["evidence"])
    assert any("not found in this family" in e for e in errors), errors


def test_stage1_genuine_number_passes_grounding_check():
    pkt = make_pkt()
    pkt["evidence"]["valuation"] = {"pe": 25.4, "pb": 2.1}
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 25.4", "PB = 2.1"], direction="NOT_INTERPRETABLE",
                limitations=["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"],
                benchmark_available=False)
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"], pkt["evidence"])
    assert errors == [], errors


def test_stage1_grounding_ignores_lookback_digits_embedded_in_field_names():
    pkt = make_pkt()
    pkt["evidence"]["price_volume_structure"] = {"ret5_pct": 1.25, "ret20_pct": -2.5}
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("price_volume_structure", ["ret5_pct = 1.25", "ret20_pct = -2.5"], direction="MIXED")
        ],
    }
    _, errors = validate_stage1_output(
        stage1_raw, 9001, ["price_volume_structure"], pkt["evidence"]
    )
    assert errors == [], errors


def test_stage1_cannot_invent_valuation_benchmark_availability():
    pkt = make_pkt()
    pkt["evidence"]["valuation"] = {"pe": 25.4, "pb": 2.1}
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("valuation", ["PE = 25.4"], direction="SUPPORTIVE", benchmark_available=True)
        ],
    }
    _, errors = validate_stage1_output(stage1_raw, 9001, ["valuation"], pkt["evidence"])
    assert any("contradicts the raw valuation packet" in e for e in errors), errors


def test_stage1_grounding_rejects_cross_family_number_borrowing():
    """A number that is genuinely correct for family A but is claimed
    under family B (i.e. borrowed from the wrong family) must still be
    rejected, because the grounding check compares only against the SAME
    family's raw data named by evidence_id."""
    pkt = make_pkt()
    pkt["evidence"]["valuation"] = {"pe": 25.4}
    pkt["evidence"]["revenue"] = {"yoy_pct": 8.0}
    stage1_raw = {
        "case_id": 9001,
        "evidence_observations": [
            obs("revenue", ["revenue growth of 25.4%"], direction="SUPPORTIVE"),  # 25.4 belongs to valuation, not revenue
        ],
    }
    normalized, errors = validate_stage1_output(stage1_raw, 9001, ["revenue"], pkt["evidence"])
    assert any("not found in this family" in e for e in errors), errors


def test_stage2_malformed_evidence_id_field_types_do_not_crash():
    """Reproduces the reported crash: primary_evidence_ids=123 used to
    raise an uncaught TypeError inside dict.fromkeys(). It must now
    produce a normal contract error instead."""
    pkt = make_pkt()
    stage1_obs = [obs("valuation", ["PE=20"], direction="SUPPORTIVE")]

    for bad_value in (123, {"not": "a list"}, "a bare string"):
        s2 = base_stage2(primary_evidence_ids=bad_value)
        normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
        assert errors, f"bad_value={bad_value!r} should have produced errors, not silently passed"
        assert any("primary_evidence_ids" in e for e in errors), errors


def test_stage2_output_not_a_dict_does_not_crash():
    pkt = make_pkt()
    stage1_obs = [obs("valuation", ["PE=20"], direction="SUPPORTIVE")]
    for bad_raw in (None, "a string", 123, ["a", "list"]):
        normalized, errors, warnings = validate_stage2_output(bad_raw, stage1_obs, pkt["evidence"])
        assert errors
        assert normalized == {}


def test_stage3_malformed_field_value_is_rejected():
    raw = {"verdict": "REVISE", "problems": [{
        "field": "not_a_real_field", "claim_text": "x", "cited_evidence_id": None,
        "problem_type": "OVERSTATES_DIRECTION", "explanation": "y",
    }]}
    normalized, errors = validate_stage3_output(raw, ["valuation"])
    assert any("not in" in e and "field" in e for e in errors)


def test_stage3_integer_claim_text_is_rejected():
    raw = {"verdict": "REVISE", "problems": [{
        "field": "bull_thesis", "claim_text": 12345, "cited_evidence_id": None,
        "problem_type": "OVERSTATES_DIRECTION", "explanation": "y",
    }]}
    normalized, errors = validate_stage3_output(raw, ["valuation"])
    assert any("claim_text must be a non-empty string" in e for e in errors)


def test_stage3_list_type_cited_evidence_id_is_rejected():
    raw = {"verdict": "REVISE", "problems": [{
        "field": "bull_thesis", "claim_text": "x", "cited_evidence_id": ["valuation", "revenue"],
        "problem_type": "OVERSTATES_DIRECTION", "explanation": "y",
    }]}
    normalized, errors = validate_stage3_output(raw, ["valuation"])
    assert any("cited_evidence_id must be null or a string" in e for e in errors)


def test_stage3_cited_evidence_id_not_in_stage1_ids_is_rejected():
    raw = {"verdict": "REVISE", "problems": [{
        "field": "bull_thesis", "claim_text": "x", "cited_evidence_id": "not_a_real_evidence_id",
        "problem_type": "OVERSTATES_DIRECTION", "explanation": "y",
    }]}
    normalized, errors = validate_stage3_output(raw, ["valuation", "revenue"])
    assert any("is not one of Stage1's evidence_ids" in e for e in errors)


def test_stage3_all_malformed_fields_at_once_are_all_reported():
    """The exact combination the review reproduced: unknown field, integer
    claim_text, list-type cited_evidence_id, all in one problem object."""
    raw = {"verdict": "REVISE", "problems": [{
        "field": "not_a_field", "claim_text": 42, "cited_evidence_id": ["a", "b"],
        "problem_type": "NOT_A_REAL_TYPE", "explanation": "",
    }]}
    normalized, errors = validate_stage3_output(raw, ["valuation"])
    assert len(errors) >= 4, f"expected at least 4 separate violations, got: {errors}"


def test_actionable_entry_price_must_be_grounded_in_price_evidence():
    """Reproduces review point 14: a CANDIDATE's entry/failure_exit prices
    must be plausible relative to the actual numbers Stage1 observed for
    price_volume_structure, not arbitrary invented figures."""
    pkt = make_pkt()
    stage1_obs = [
        obs("price_volume_structure", ["current price = 100.0", "ret20 = +5.0%"], direction="SUPPORTIVE"),
    ]
    s2 = base_stage2(
        decision="CANDIDATE",
        primary_evidence_ids=["price_volume_structure"],
        entry={"status": "NOW", "ideal_low": 5000.0, "ideal_high": 5100.0},  # absurd vs current price ~100
        failure_exit={"exit_price": 4800.0, "reason": "x", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    )
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert any("not within a plausible range" in e for e in errors), errors


def test_actionable_entry_price_within_plausible_range_passes():
    pkt = make_pkt()
    stage1_obs = [
        obs("price_volume_structure", ["current price = 100.0", "ret20 = +5.0%"], direction="SUPPORTIVE"),
    ]
    s2 = base_stage2(
        decision="CANDIDATE",
        primary_evidence_ids=["price_volume_structure"],
        entry={"status": "NOW", "ideal_low": 95.0, "ideal_high": 105.0},
        failure_exit={"exit_price": 88.0, "reason": "structure break", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    )
    normalized, errors, warnings = validate_stage2_output(s2, stage1_obs, pkt["evidence"])
    assert errors == [], errors
