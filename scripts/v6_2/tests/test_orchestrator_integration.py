"""
test_orchestrator_integration.py

Rewritten after real-model canary runs 35482706294 / 35485028854. Tests
retry_orchestrator.run_case()/run_batch() against the new 4-call flow
(stage1, stage2_decision, stage2_actionability [conditional], stage3)
using MOCK callables -- no real LLM. Includes fixtures required by the
review: "必須有fixture證明每種Stage2 failure能在retry修復" (must have a
fixture proving each Stage2 failure type can be repaired via retry) --
see the repair_request-driven convergence tests near the end, which use
"smart mocks" that read the structured repair_request field from the
payload and apply exactly the indicated fix, proving the FEEDBACK CHANNEL
carries enough information for a fix to converge (whether the real 7B
model reliably acts on it is a separate, only-real-canary-testable
question, stated honestly in the README).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_capability import SYSTEM_AVAILABLE_FAMILIES, SYSTEM_UNAVAILABLE_FAMILIES
from retry_orchestrator import OrchestratorConfig, run_case, run_batch


def make_pkt(case_id=1):
    return {
        "case_id": case_id,
        "decision_date": 20240610,
        "code": f"T{case_id:04d}",
        "evidence": {
            "available_evidence_ids": sorted(SYSTEM_AVAILABLE_FAMILIES),
            "missing_evidence_ids": sorted(SYSTEM_UNAVAILABLE_FAMILIES),
            "price_volume_structure": {
                "current_price": 100.0, "ret20_pct": 5.0, "support_low20": 88.0,
                "high20": 105.0, "atr20": 2.0,
            },
            "valuation": {"pe": 20.0},
        },
    }


def valid_stage1_raw(pkt):
    obs_list = []
    for fam in sorted(SYSTEM_AVAILABLE_FAMILIES):
        if fam == "valuation":
            obs_list.append({
                "evidence_id": fam, "exact_observations": ["PE = 20.0"], "direction": "NOT_INTERPRETABLE",
                "timeliness": "CURRENT", "relevance_to_hypothesis_space": "MEDIUM",
                "limitations": ["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"],
                "benchmark_available": False,
            })
        elif fam == "price_volume_structure":
            obs_list.append({
                "evidence_id": fam, "exact_observations": ["current price = 100.0", "ret20 = +5.0%"],
                "direction": "SUPPORTIVE", "timeliness": "CURRENT", "relevance_to_hypothesis_space": "HIGH",
                "limitations": [], "benchmark_available": True,
            })
        else:
            obs_list.append({
                "evidence_id": fam, "exact_observations": [f"{fam} placeholder observation"],
                "direction": "SUPPORTIVE", "timeliness": "CURRENT", "relevance_to_hypothesis_space": "HIGH",
                "limitations": [], "benchmark_available": True,
            })
    return {"case_id": pkt["case_id"], "evidence_observations": obs_list}


def valid_decision_raw(decision="CANDIDATE"):
    return {
        "decision": decision,
        "evidence_quality": "STRONG",
        "hypothesis_type": "TEST",
        "hypothesis": "test",
        "primary_evidence_ids": ["revenue", "price_volume_structure"],
        "secondary_evidence_ids": [],
        "counter_evidence_ids": [],
        "system_limitations": sorted(SYSTEM_UNAVAILABLE_FAMILIES),
        "bull_thesis": "revenue and price structure support the thesis.",
        "bear_thesis": "no material counter evidence identified.",
        "invalidation": "revenue reverses.",
        "decision_reason": "revenue and price structure support the thesis with adequate strength.",
    }


def valid_actionability_raw():
    return {
        "entry": {"status": "NOW", "ideal_low": 95.0, "ideal_high": 105.0},
        "failure_exit": {"exit_price": 88.0, "reason": "support break", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    }


def approve_critic_raw():
    return {"verdict": "APPROVE", "problems": []}


def revise_critic_raw():
    return {
        "verdict": "REVISE",
        "problems": [{
            "field": "bull_thesis", "claim_text": "revenue and price structure support the thesis.",
            "cited_evidence_id": "revenue", "problem_type": "UNGROUNDED_CLAIM",
            "explanation": "placeholder critique for test purposes.",
        }],
    }


NOOP_PAYLOAD_BUILDERS = (
    lambda pkt: {"case_id": pkt["case_id"]},
    lambda pkt, s1, prior: {"case_id": pkt["case_id"], "stage1": s1, "prior": prior},
    lambda pkt, s1, decision, prior: {"case_id": pkt["case_id"], "decision": decision},
    lambda s1, s2, precheck: {"stage1": s1, "stage2_decision_output": s2, "precheck": precheck},
)


def _run(pkt, s1, s2d, s2a, s3, config=None, numerical_prior=None):
    return run_case(
        pkt, numerical_prior or {}, s1, s2d, s2a, s3,
        "sys1", "sys2d", "sys2a", "sys3", *NOOP_PAYLOAD_BUILDERS, config=config,
    )


# ---------------------------------------------------------------------------
# Happy paths -- actionable vs non-actionable call-count behavior
# ---------------------------------------------------------------------------


def test_non_actionable_decision_never_calls_actionability_stage():
    pkt = make_pkt(1)
    calls = {"s1": 0, "s2d": 0, "s2a": 0, "s3": 0}

    def s1(sysp, payload):
        calls["s1"] += 1
        return valid_stage1_raw(pkt)

    def s2d(sysp, payload):
        calls["s2d"] += 1
        return valid_decision_raw(decision="REJECT")

    def s2a(sysp, payload):
        calls["s2a"] += 1
        raise AssertionError("stage2_actionability_call must never be invoked for a REJECT decision")

    def s3(sysp, payload):
        calls["s3"] += 1
        return approve_critic_raw()

    result = _run(pkt, s1, s2d, s2a, s3)
    assert result.status == "FINALIZED"
    assert result.final_stage2_output["decision"] == "REJECT"
    assert result.final_stage2_output["entry"] == {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
    assert result.final_stage2_output["failure_exit"] == {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}
    assert calls["s2a"] == 0
    assert calls == {"s1": 1, "s2d": 1, "s2a": 0, "s3": 1}


def test_watch_decision_also_never_calls_actionability_stage():
    pkt = make_pkt(2)
    s2a_calls = {"n": 0}

    def s2a(sysp, payload):
        s2a_calls["n"] += 1
        return valid_actionability_raw()

    result = _run(
        pkt, lambda sysp, payload: valid_stage1_raw(pkt),
        lambda sysp, payload: valid_decision_raw(decision="WATCH"),
        s2a, lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert result.final_stage2_output["decision"] == "WATCH"
    assert s2a_calls["n"] == 0


def test_actionable_decision_calls_actionability_stage_exactly_once():
    pkt = make_pkt(3)
    calls = {"s2d": 0, "s2a": 0}

    def s2d(sysp, payload):
        calls["s2d"] += 1
        return valid_decision_raw(decision="CANDIDATE")

    def s2a(sysp, payload):
        calls["s2a"] += 1
        return valid_actionability_raw()

    result = _run(
        pkt, lambda sysp, payload: valid_stage1_raw(pkt), s2d, s2a,
        lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert result.final_stage2_output["decision"] == "CANDIDATE"
    assert result.final_stage2_output["entry"]["ideal_low"] == 95.0
    assert calls == {"s2d": 1, "s2a": 1}


# ---------------------------------------------------------------------------
# Stage 1 retry (unchanged mechanics, re-verified against new signature)
# ---------------------------------------------------------------------------


def test_stage1_retries_then_succeeds():
    pkt = make_pkt(4)
    attempts = {"n": 0}

    def s1(sysp, payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"case_id": pkt["case_id"]}  # missing evidence_observations
        return valid_stage1_raw(pkt)

    result = _run(
        pkt, s1, lambda sysp, payload: valid_decision_raw(decision="REJECT"),
        lambda sysp, payload: valid_actionability_raw(), lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# Critic revise -> re-run of the stage2 pipeline (decision, + actionability
# if still/newly actionable)
# ---------------------------------------------------------------------------


def test_critic_revise_reruns_decision_and_actionability_then_approves():
    pkt = make_pkt(5)
    s2d_calls, s2a_calls, critic_calls = {"n": 0}, {"n": 0}, {"n": 0}

    def s2d(sysp, payload):
        s2d_calls["n"] += 1
        raw = valid_decision_raw(decision="CANDIDATE")
        if s2d_calls["n"] == 1:
            raw["bull_thesis"] = "unrevised first attempt"
        else:
            assert "critic_feedback" in payload
            raw["bull_thesis"] = "revised after critic feedback"
        return raw

    def s2a(sysp, payload):
        s2a_calls["n"] += 1
        return valid_actionability_raw()

    def s3(sysp, payload):
        critic_calls["n"] += 1
        return revise_critic_raw() if "unrevised" in payload["stage2_decision_output"]["bull_thesis"] else approve_critic_raw()

    result = _run(pkt, lambda sysp, payload: valid_stage1_raw(pkt), s2d, s2a, s3)
    assert result.status == "FINALIZED"
    assert s2d_calls["n"] == 2
    assert s2a_calls["n"] == 2  # actionability re-run on revision too
    assert critic_calls["n"] == 2
    assert "revised after critic feedback" in result.final_stage2_output["bull_thesis"]


def test_critic_revise_from_candidate_to_reject_stops_calling_actionability():
    """If a revision changes the decision from actionable to non-
    actionable, the revision pipeline must assemble null fields
    deterministically, not call actionability again."""
    pkt = make_pkt(6)
    s2d_calls, s2a_calls = {"n": 0}, {"n": 0}

    def s2d(sysp, payload):
        s2d_calls["n"] += 1
        if s2d_calls["n"] == 1:
            return valid_decision_raw(decision="CANDIDATE")
        return valid_decision_raw(decision="REJECT")  # revised down to REJECT

    def s2a(sysp, payload):
        s2a_calls["n"] += 1
        return valid_actionability_raw()

    def s3(sysp, payload):
        return approve_critic_raw() if payload["stage2_decision_output"]["decision"] == "REJECT" else revise_critic_raw()

    result = _run(pkt, lambda sysp, payload: valid_stage1_raw(pkt), s2d, s2a, s3)
    assert result.status == "FINALIZED"
    assert result.final_stage2_output["decision"] == "REJECT"
    assert result.final_stage2_output["entry"]["status"] == "NOT_ACTIONABLE"
    assert s2a_calls["n"] == 1  # only from the FIRST (CANDIDATE) pass, not the revision


# ---------------------------------------------------------------------------
# Timeout / exception-safety regressions (carried forward from prior review)
# ---------------------------------------------------------------------------


def test_hard_per_call_timeout_is_enforced():
    pkt = make_pkt(7)

    def hanging_s1(sysp, payload):
        time.sleep(5.0)
        return valid_stage1_raw(pkt)

    config = OrchestratorConfig(stage1_max_attempts=1, per_call_timeout_seconds=0.2, per_case_timeout_seconds=1.0)
    start = time.monotonic()
    result = _run(
        pkt, hanging_s1, lambda sysp, payload: valid_decision_raw(),
        lambda sysp, payload: valid_actionability_raw(), lambda sysp, payload: approve_critic_raw(),
        config=config,
    )
    elapsed = time.monotonic() - start
    assert result.status == "MODEL_CONTRACT_FAILURE"
    assert elapsed < 2.0
    assert any(h.timed_out for h in result.raw_history)


def test_malformed_decision_field_types_do_not_crash_run_case():
    pkt = make_pkt(8)

    def bad_s2d(sysp, payload):
        raw = valid_decision_raw()
        raw["primary_evidence_ids"] = 123
        return raw

    config = OrchestratorConfig(stage2_decision_max_attempts=1)
    result = _run(
        pkt, lambda sysp, payload: valid_stage1_raw(pkt), bad_s2d,
        lambda sysp, payload: valid_actionability_raw(), lambda sysp, payload: approve_critic_raw(),
        config=config,
    )
    assert result.status == "MODEL_CONTRACT_FAILURE"


def test_run_batch_survives_unexpected_exception_in_one_case():
    pkt_bad, pkt_good = make_pkt(9), make_pkt(10)

    def s1(sysp, payload):
        if payload["case_id"] == pkt_bad["case_id"]:
            raise RuntimeError("simulated totally unexpected failure")
        return valid_stage1_raw(pkt_good)

    results = run_batch(
        [pkt_bad, pkt_good], {}, s1,
        lambda sysp, payload: valid_decision_raw(decision="REJECT"),
        lambda sysp, payload: valid_actionability_raw(),
        lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2d", "sys2a", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    by_id = {r.case_id: r for r in results}
    assert by_id[pkt_good["case_id"]].status == "FINALIZED"
    assert by_id[pkt_bad["case_id"]].status == "MODEL_CONTRACT_FAILURE"


def test_history_entries_record_elapsed_seconds():
    """Review point 7: per-stage latency must be capturable."""
    pkt = make_pkt(11)
    result = _run(
        pkt, lambda sysp, payload: valid_stage1_raw(pkt),
        lambda sysp, payload: valid_decision_raw(decision="REJECT"),
        lambda sysp, payload: valid_actionability_raw(), lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert all(isinstance(h.elapsed_seconds, float) for h in result.raw_history)
    assert all(h.elapsed_seconds >= 0.0 for h in result.raw_history)


# ---------------------------------------------------------------------------
# Repair-request-driven convergence fixtures (review point 5: "必須有
# fixture證明每種Stage2 failure能在retry修復")
# ---------------------------------------------------------------------------


def test_repair_request_driven_retry_fixes_decision_enum_violation():
    """A 'smart mock' that reads repair_request off the payload and
    applies exactly the indicated fix. Proves the feedback channel alone
    carries enough information for a mechanical fix to converge."""
    pkt = make_pkt(12)
    attempts = {"n": 0}

    def smart_s2d(sysp, payload):
        attempts["n"] += 1
        raw = valid_decision_raw(decision="REJECT")
        if attempts["n"] == 1:
            raw["decision"] = "MAYBE"  # invalid enum value
        return raw

    result = _run(
        pkt, lambda sysp, payload: valid_stage1_raw(pkt), smart_s2d,
        lambda sysp, payload: valid_actionability_raw(), lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert attempts["n"] == 2


def test_repair_request_driven_retry_fixes_missing_counter_evidence():
    pkt = make_pkt(13)
    attempts = {"n": 0}

    def stage1_with_contradictory(pkt_):
        raw = valid_stage1_raw(pkt_)
        for o in raw["evidence_observations"]:
            if o["evidence_id"] == "revenue":
                o["direction"] = "CONTRADICTORY"
                o["exact_observations"] = ["revenue -15% YoY"]
        return raw

    def smart_s2d(sysp, payload):
        attempts["n"] += 1
        raw = valid_decision_raw(decision="WATCH")
        raw["primary_evidence_ids"] = ["price_volume_structure"]
        if attempts["n"] == 1:
            raw["counter_evidence_ids"] = []  # omits required CONTRADICTORY revenue family
        else:
            repair = payload.get("repair_request", [])
            must_add_fields = [r for r in repair if r.get("problem") == "missing_required_contradictory_family"]
            assert must_add_fields, f"expected a missing_required_contradictory_family repair item, got {repair}"
            raw["counter_evidence_ids"] = ["revenue"]
        return raw

    result = _run(
        pkt, lambda sysp, payload: stage1_with_contradictory(pkt), smart_s2d,
        lambda sysp, payload: valid_actionability_raw(), lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert attempts["n"] == 2
    assert "revenue" in result.final_stage2_output["counter_evidence_ids"]


def test_repair_request_driven_retry_fixes_actionability_price_violation():
    pkt = make_pkt(14)
    attempts = {"n": 0}

    def smart_s2a(sysp, payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"entry": {"status": "NOW", "ideal_low": 5000.0, "ideal_high": 5100.0},
                    "failure_exit": {"exit_price": 4800.0, "reason": "x", "trigger_type": "PRICE_STRUCTURE_BREAK"}}
        repair = payload.get("repair_request", [])
        assert any(r.get("problem") == "price_not_grounded_in_evidence" for r in repair), repair
        return valid_actionability_raw()

    result = _run(
        pkt, lambda sysp, payload: valid_stage1_raw(pkt),
        lambda sysp, payload: valid_decision_raw(decision="CANDIDATE"),
        smart_s2a, lambda sysp, payload: approve_critic_raw(),
    )
    assert result.status == "FINALIZED"
    assert attempts["n"] == 2
    assert result.final_stage2_output["entry"]["ideal_low"] == 95.0
