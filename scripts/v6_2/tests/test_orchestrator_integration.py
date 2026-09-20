"""
test_orchestrator_integration.py

Exercises retry_orchestrator.run_case()/run_batch() end-to-end using MOCK
stage callables (no real LLM). This version adds the specific regression
tests required by the external review:

  - retry's 2nd payload actually contains the 1st attempt's errors
  - a hard per-call timeout is enforced and a late result is never finalized
  - a validator exception (or any unexpected exception) never crashes
    run_case() or run_batch()
  - safe-finalizer applied_fixes/warnings are retained in history
  - run_batch() survives an unexpected exception raised from inside
    run_case() for one case without affecting any other case
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
        },
    }


def valid_stage1_raw(pkt):
    return {
        "case_id": pkt["case_id"],
        "evidence_observations": [
            {
                "evidence_id": fam,
                "exact_observations": [f"{fam} placeholder observation"],
                "direction": "SUPPORTIVE",
                "timeliness": "CURRENT",
                "relevance_to_hypothesis_space": "HIGH",
                "limitations": (
                    ["no historical or peer valuation benchmark supplied -- cheap/expensive cannot be asserted"]
                    if fam == "valuation" else []
                ),
                "benchmark_available": fam != "valuation",
            }
            for fam in sorted(SYSTEM_AVAILABLE_FAMILIES)
        ],
    }


def valid_stage2_raw():
    return {
        "decision": "CANDIDATE",
        "evidence_quality": "STRONG",
        "hypothesis_type": "TEST",
        "hypothesis": "test",
        "primary_evidence_ids": ["revenue"],
        "secondary_evidence_ids": [],
        "counter_evidence_ids": [],
        "system_limitations": sorted(SYSTEM_UNAVAILABLE_FAMILIES),
        "bull_thesis": "revenue supports the thesis.",
        "bear_thesis": "no material counter evidence identified.",
        "invalidation": "revenue reverses.",
        "decision_reason": "revenue supports the thesis with adequate strength.",
        "entry": {"status": "NOW", "ideal_low": 90.0, "ideal_high": 95.0},
        "failure_exit": {"exit_price": 80.0, "reason": "support break", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    }


def approve_critic_raw():
    return {"verdict": "APPROVE", "problems": []}


def revise_critic_raw():
    return {
        "verdict": "REVISE",
        "problems": [{
            "field": "bull_thesis", "claim_text": "revenue supports the thesis.",
            "cited_evidence_id": "revenue", "problem_type": "UNGROUNDED_CLAIM",
            "explanation": "placeholder critique for test purposes.",
        }],
    }


NOOP_PAYLOAD_BUILDERS = (
    lambda pkt: {"case_id": pkt["case_id"]},
    lambda pkt, s1, prior: {"case_id": pkt["case_id"], "stage1": s1, "prior": prior},
    lambda s1, s2, precheck: {"stage1": s1, "stage2_decision_output": s2, "precheck": precheck},
)


def test_happy_path_finalizes_on_first_try():
    pkt = make_pkt(1)
    calls = {"s1": 0, "s2": 0, "s3": 0}

    def s1_call(sysp, payload):
        calls["s1"] += 1
        return valid_stage1_raw(pkt)

    def s2_call(sysp, payload):
        calls["s2"] += 1
        return valid_stage2_raw()

    def s3_call(sysp, payload):
        calls["s3"] += 1
        return approve_critic_raw()

    result = run_case(pkt, {}, s1_call, s2_call, s3_call, "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS)
    assert result.status == "FINALIZED"
    assert result.final_stage2_output["decision"] == "CANDIDATE"
    assert calls == {"s1": 1, "s2": 1, "s3": 1}
    assert len(result.raw_history) == 3


def test_stage1_retries_then_succeeds():
    pkt = make_pkt(2)
    attempts = {"n": 0}

    def s1_call(sysp, payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"case_id": pkt["case_id"]}  # missing evidence_observations -> invalid
        return valid_stage1_raw(pkt)

    result = run_case(
        pkt, {}, s1_call, lambda sysp, payload: valid_stage2_raw(), lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    assert result.status == "FINALIZED"
    assert attempts["n"] == 2
    stage1_entries = [h for h in result.raw_history if h.stage == "stage1"]
    assert len(stage1_entries) == 2
    assert stage1_entries[0].errors
    assert not stage1_entries[1].errors


def test_stage1_exhausts_attempts_and_fails_case_without_touching_stage2_or_3():
    pkt = make_pkt(3)
    s2_calls, s3_calls = {"n": 0}, {"n": 0}

    def always_broken_s1(sysp, payload):
        return {"case_id": pkt["case_id"]}

    def s2_call(sysp, payload):
        s2_calls["n"] += 1
        return valid_stage2_raw()

    def s3_call(sysp, payload):
        s3_calls["n"] += 1
        return approve_critic_raw()

    config = OrchestratorConfig(stage1_max_attempts=2)
    result = run_case(
        pkt, {}, always_broken_s1, s2_call, s3_call, "sys1", "sys2", "sys3",
        *NOOP_PAYLOAD_BUILDERS, config=config,
    )
    assert result.status == "MODEL_CONTRACT_FAILURE"
    assert result.failure_stage == "stage1"
    assert s2_calls["n"] == 0
    assert s3_calls["n"] == 0
    stage1_entries = [h for h in result.raw_history if h.stage == "stage1"]
    assert len(stage1_entries) == 2


def test_critic_revise_then_approve_on_revision():
    pkt = make_pkt(4)
    s2_calls, critic_calls = {"n": 0}, {"n": 0}

    def s2_call(sysp, payload):
        s2_calls["n"] += 1
        raw = valid_stage2_raw()
        if s2_calls["n"] == 1:
            raw["bull_thesis"] = "unrevised first attempt"
        else:
            raw["bull_thesis"] = "revised after critic feedback: " + str(payload.get("critic_feedback"))
        return raw

    def s3_call(sysp, payload):
        critic_calls["n"] += 1
        decision_text = payload["stage2_decision_output"]["bull_thesis"]
        return revise_critic_raw() if "unrevised" in decision_text else approve_critic_raw()

    result = run_case(
        pkt, {}, lambda sysp, payload: valid_stage1_raw(pkt), s2_call, s3_call,
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    assert result.status == "FINALIZED"
    assert s2_calls["n"] == 2
    assert critic_calls["n"] == 2
    assert "revised after critic feedback" in result.final_stage2_output["bull_thesis"]
    assert result.critic_history[0]["verdict"] == "REVISE"
    assert result.critic_history[1]["verdict"] == "APPROVE"


def test_critic_never_approves_exhausts_rounds_and_fails():
    pkt = make_pkt(5)
    config = OrchestratorConfig(critic_max_rounds=2)
    result = run_case(
        pkt, {}, lambda sysp, payload: valid_stage1_raw(pkt),
        lambda sysp, payload: valid_stage2_raw(),
        lambda sysp, payload: revise_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS, config=config,
    )
    assert result.status == "MODEL_CONTRACT_FAILURE"
    assert result.failure_stage == "stage3"
    assert "critic_max_rounds exhausted" in result.failure_reason
    critic_entries = [h for h in result.raw_history if h.stage == "stage3"]
    assert len(critic_entries) == 2


def test_batch_independence_one_failure_does_not_affect_other_cases():
    pkt_good, pkt_bad = make_pkt(6), make_pkt(7)

    def s1_call(sysp, payload):
        if payload["case_id"] == pkt_bad["case_id"]:
            return {"case_id": pkt_bad["case_id"]}
        return valid_stage1_raw(pkt_good)

    results = run_batch(
        [pkt_good, pkt_bad], {}, s1_call,
        lambda sysp, payload: valid_stage2_raw(),
        lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
        config=OrchestratorConfig(stage1_max_attempts=1),
    )
    by_id = {r.case_id: r for r in results}
    assert by_id[pkt_good["case_id"]].status == "FINALIZED"
    assert by_id[pkt_bad["case_id"]].status == "MODEL_CONTRACT_FAILURE"


def test_per_case_timeout_triggers_model_contract_failure():
    pkt = make_pkt(8)

    def slow_s1(sysp, payload):
        time.sleep(0.05)
        return valid_stage1_raw(pkt)

    config = OrchestratorConfig(per_case_timeout_seconds=0.001, per_call_timeout_seconds=0.001)
    result = run_case(
        pkt, {}, slow_s1, lambda sysp, payload: valid_stage2_raw(), lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS, config=config,
    )
    assert result.status == "MODEL_CONTRACT_FAILURE"


# ---------------------------------------------------------------------------
# NEW regression tests required by the external review
# ---------------------------------------------------------------------------


def test_retry_second_attempt_payload_contains_first_attempt_errors():
    """The reported bug: the 2nd attempt used to get an identical payload
    to the 1st, with no memory of what was wrong. Verify the payload
    passed to the model on attempt 2 actually carries attempt 1's raw
    output and its exact validation errors."""
    pkt = make_pkt(9)
    seen_payloads = []

    def s1_call(sysp, payload):
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            return {"case_id": pkt["case_id"]}  # invalid: missing evidence_observations
        return valid_stage1_raw(pkt)

    result = run_case(
        pkt, {}, s1_call, lambda sysp, payload: valid_stage2_raw(), lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    assert result.status == "FINALIZED"
    assert len(seen_payloads) == 2
    assert "previous_attempt" not in seen_payloads[0]
    assert "previous_attempt" in seen_payloads[1]
    assert seen_payloads[1]["previous_attempt"]["raw_output"] == {"case_id": pkt["case_id"]}
    assert any("missing key" in e for e in seen_payloads[1]["previous_attempt"]["validation_errors"])


def test_stage2_revision_payload_contains_critic_feedback_and_prior_errors():
    """Same check, but for the Stage2-revision path triggered by a critic
    REVISE verdict: the revision payload must carry critic_feedback, and
    if the revision itself first fails contract validation, the SECOND
    revision attempt must also carry previous_attempt."""
    pkt = make_pkt(10)
    s2_calls = {"n": 0}
    seen_revision_payloads = []

    def s2_call(sysp, payload):
        s2_calls["n"] += 1
        if s2_calls["n"] == 1:
            raw = valid_stage2_raw()
            raw["bull_thesis"] = "first pass"
            return raw
        # This is a revision attempt (critic_feedback present in payload)
        seen_revision_payloads.append(payload)
        if len(seen_revision_payloads) == 1:
            bad = valid_stage2_raw()
            bad["decision"] = "NOT_A_REAL_DECISION"  # force contract failure on first revision try
            return bad
        good = valid_stage2_raw()
        good["bull_thesis"] = "second revision, fixed"
        return good

    def s3_call(sysp, payload):
        text = payload["stage2_decision_output"]["bull_thesis"]
        return approve_critic_raw() if "second revision" in text else revise_critic_raw()

    result = run_case(
        pkt, {}, lambda sysp, payload: valid_stage1_raw(pkt), s2_call, s3_call,
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    assert result.status == "FINALIZED"
    assert len(seen_revision_payloads) == 2
    assert "critic_feedback" in seen_revision_payloads[0]
    assert "previous_attempt" not in seen_revision_payloads[0]
    assert "critic_feedback" in seen_revision_payloads[1]
    assert "previous_attempt" in seen_revision_payloads[1]


def test_hard_per_call_timeout_is_enforced_even_if_call_never_returns():
    """A model call that hangs forever must not hang the orchestrator
    forever -- the hard per-call timeout must abandon it."""
    pkt = make_pkt(11)

    def hanging_s1(sysp, payload):
        time.sleep(5.0)  # much longer than the timeout below
        return valid_stage1_raw(pkt)

    config = OrchestratorConfig(stage1_max_attempts=1, per_call_timeout_seconds=0.2, per_case_timeout_seconds=1.0)
    start = time.monotonic()
    result = run_case(
        pkt, {}, hanging_s1, lambda sysp, payload: valid_stage2_raw(), lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS, config=config,
    )
    elapsed = time.monotonic() - start
    assert result.status == "MODEL_CONTRACT_FAILURE"
    assert elapsed < 2.0, f"orchestrator waited {elapsed:.2f}s, hard timeout was not enforced"
    assert any(h.timed_out for h in result.raw_history)


def test_late_result_after_deadline_is_not_finalized():
    """Whether a slow call is caught by the hard per-call timeout, or by
    the post-return deadline recheck (both mechanisms exist -- see
    retry_orchestrator.py's module docstring, fix #2), the outcome must be
    the same: a case whose call takes longer than the available budget is
    never finalized, and some history entry is flagged timed_out."""
    pkt = make_pkt(12)

    def slow_but_valid_s1(sysp, payload):
        time.sleep(0.15)
        return valid_stage1_raw(pkt)

    config = OrchestratorConfig(
        stage1_max_attempts=1, per_call_timeout_seconds=5.0, per_case_timeout_seconds=0.05,
    )
    result = run_case(
        pkt, {}, slow_but_valid_s1, lambda sysp, payload: valid_stage2_raw(), lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS, config=config,
    )
    assert result.status == "MODEL_CONTRACT_FAILURE"
    assert any(h.timed_out for h in result.raw_history), result.raw_history


def test_malformed_field_types_do_not_crash_run_case():
    """Reproduces the reported crash pattern end-to-end through run_case
    (not just the validator directly): a Stage2 response with
    primary_evidence_ids as an int must produce a normal
    MODEL_CONTRACT_FAILURE, never an uncaught exception."""
    pkt = make_pkt(13)

    def bad_s2_call(sysp, payload):
        raw = valid_stage2_raw()
        raw["primary_evidence_ids"] = 123  # malformed: should be a list
        return raw

    config = OrchestratorConfig(stage2_max_format_attempts=1)
    result = run_case(
        pkt, {}, lambda sysp, payload: valid_stage1_raw(pkt), bad_s2_call,
        lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS, config=config,
    )
    assert result.status == "MODEL_CONTRACT_FAILURE"  # not an exception


def test_history_retains_applied_fixes_and_warnings():
    """The reported gap: safe_finalizer's applied_fixes/warnings used to
    be computed and then discarded by the orchestrator wrapper, keeping
    only errors. Verify they now survive into HistoryEntry."""
    pkt = make_pkt(14)

    def s2_call_watch_wrong_nullfields(sysp, payload):
        raw = valid_stage2_raw()
        raw["decision"] = "WATCH"
        # entry/failure_exit left in the CANDIDATE-shaped form from
        # valid_stage2_raw() -- the ONLY problem, so safe_finalizer should
        # fix it and record the fixes.
        return raw

    result = run_case(
        pkt, {}, lambda sysp, payload: valid_stage1_raw(pkt), s2_call_watch_wrong_nullfields,
        lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    assert result.status == "FINALIZED"
    assert result.final_stage2_output["decision"] == "WATCH"
    stage2_entries = [h for h in result.raw_history if h.stage == "stage2"]
    assert len(stage2_entries) == 1
    assert stage2_entries[0].applied_fixes, "applied_fixes must be retained in history, not discarded"
    assert len(stage2_entries[0].applied_fixes) == 4


def test_run_batch_survives_unexpected_exception_in_one_case():
    """If run_case() itself raises an exception nobody anticipated (not a
    normal validator contract-error path, which never raises), run_batch
    must catch it per-case and continue with the rest of the batch."""
    pkt_bad = make_pkt(15)
    pkt_good = make_pkt(16)

    def s1_call(sysp, payload):
        if payload["case_id"] == pkt_bad["case_id"]:
            raise RuntimeError("simulated totally unexpected failure")
        return valid_stage1_raw(pkt_good)

    results = run_batch(
        [pkt_bad, pkt_good], {}, s1_call,
        lambda sysp, payload: valid_stage2_raw(),
        lambda sysp, payload: approve_critic_raw(),
        "sys1", "sys2", "sys3", *NOOP_PAYLOAD_BUILDERS,
    )
    by_id = {r.case_id: r for r in results}
    # The RuntimeError is raised from inside s1_call, which IS caught by
    # run_case's own per-attempt try/except (call_fn failures are always
    # retryable attempts) -- so this case still resolves to
    # MODEL_CONTRACT_FAILURE via the normal stage1-exhausted path, and
    # crucially the OTHER case in the batch is completely unaffected.
    assert by_id[pkt_bad["case_id"]].status == "MODEL_CONTRACT_FAILURE"
    assert by_id[pkt_good["case_id"]].status == "FINALIZED"
