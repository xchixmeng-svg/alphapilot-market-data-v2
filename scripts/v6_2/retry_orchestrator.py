"""
retry_orchestrator.py

Runs the Stage1 -> Stage2 -> Stage3(critic) -> [revise Stage2] pipeline for
one case.

FIXED AFTER EXTERNAL REVIEW (this version):

  1. Retries now actually feed the model concrete feedback. Every payload
     builder receives (previous_raw_output, previous_errors) in addition
     to its normal arguments. On attempt 1 both are None/[]; on attempt 2+
     they carry exactly what the model returned last time and exactly
     which contract rules it violated. Verified by
     test_retry_second_attempt_payload_contains_first_attempt_errors.

  2. Timeout is now enforced two ways, not one:
       a) each model call is wrapped in a hard per-call timeout via
          concurrent.futures (the call is abandoned at the code level if
          it doesn't return in time -- note Python cannot forcibly kill a
          native thread, so the underlying HTTP call should ALSO set its
          own request timeout in your model client for a true network-
          level abort; this is documented, not hidden);
       b) after every call returns, the deadline is re-checked BEFORE the
          result is accepted/validated. A result that arrives after the
          case deadline has passed is discarded and treated as a timeout,
          even if it would otherwise have been a valid APPROVE. Verified
          by test_late_result_after_deadline_is_not_finalized.

  3. Every validate_fn call is wrapped in an additional orchestrator-level
     try/except as defense-in-depth (the validators in
     deterministic_validators.py are already internally exception-safe as
     of this version, but the orchestrator does not rely on that alone).
     A single malformed Stage2 field (e.g. primary_evidence_ids=123) can
     no longer raise an uncaught exception anywhere in this file. Verified
     by test_malformed_field_types_do_not_crash_run_case.

  4. run_batch() now wraps each run_case() call in its own try/except, so
     an unexpected bug in ANY single case can never abort the rest of the
     batch. Verified by test_run_batch_survives_unexpected_exception_in_
     one_case.

  5. History now retains the FULL FinalizeResult for Stage 2 attempts
     (normalized output, applied_fixes, remaining_errors, remaining_
     warnings), not just errors. Verified by
     test_history_retains_applied_fixes_and_warnings.

  6. Default timeouts recalibrated against the REPORTED real Qwen2.5-7B
     single-call latency (~149s observed). See OrchestratorConfig for the
     derivation. The previous 240s per-case default was never realistic
     for a 3-stage pipeline and has been replaced.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Callable

from deterministic_validators import validate_stage1_output, validate_stage3_output
from evidence_capability import derive_case_evidence_sets
from safe_finalizer import FinalizeResult, attempt_safe_finalize

StageCallable = Callable[[str, dict], dict]
# payload_factory(previous_raw, previous_errors) -> user_payload dict
PayloadFactory = Callable[[dict | None, list[str]], dict]

_executor = ThreadPoolExecutor(max_workers=8)


@dataclass
class OrchestratorConfig:
    stage1_max_attempts: int = 2
    stage2_max_format_attempts: int = 2
    critic_max_rounds: int = 2

    # Derivation: the reported real single Qwen2.5-7B call latency is
    # ~149s. A single model call must be allowed to run at least that
    # long without being killed, so per_call_timeout_seconds is set with
    # headroom above the observed figure, not below it.
    per_call_timeout_seconds: float = 180.0

    # Worst case call count for one case: stage1(<=2) + stage2 initial
    # (<=2) + critic round1(1) + stage2 revision(<=2) + critic round2(1)
    # = <=8 calls. At up to per_call_timeout_seconds each, that is up to
    # 8 * 180s = 1440s. per_case_timeout_seconds is set with headroom
    # above that worst case, not equal to or below it (the previous 240s
    # default could never have completed even one full round in practice
    # and is retired).
    per_case_timeout_seconds: float = 1500.0


@dataclass
class HistoryEntry:
    stage: str  # "stage1" | "stage2" | "stage3" | "stage2_revision"
    attempt: int
    raw_output: dict | None
    errors: list[str]
    warnings: list[str] = field(default_factory=list)
    applied_fixes: list[str] = field(default_factory=list)
    exception: str | None = None
    timed_out: bool = False


@dataclass
class CaseResult:
    case_id: int
    status: str  # "FINALIZED" | "MODEL_CONTRACT_FAILURE"
    final_stage2_output: dict | None
    stage1_output: dict | None
    critic_history: list[dict]
    raw_history: list[HistoryEntry]
    elapsed_seconds: float
    failure_stage: str | None = None
    failure_reason: str | None = None


def _call_with_hard_timeout(call_fn: StageCallable, system_prompt: str, payload: dict, timeout_s: float):
    """
    Runs call_fn in a worker thread and enforces `timeout_s` at the code
    level via Future.result(timeout=...). Raises FutureTimeoutError if the
    call does not return in time (the underlying thread may continue
    running in the background -- Python offers no safe way to forcibly
    kill it; your actual model-serving client should set its own request-
    level timeout, e.g. `requests.post(..., timeout=timeout_s)`, so the
    network call itself is aborted rather than merely abandoned here).
    """
    future = _executor.submit(call_fn, system_prompt, payload)
    return future.result(timeout=timeout_s)


def _run_stage_with_retries(
    stage_name: str,
    call_fn: StageCallable,
    system_prompt: str,
    payload_factory: PayloadFactory,
    validate_fn: Callable[[dict], tuple],
    max_attempts: int,
    history: list[HistoryEntry],
    deadline: float,
    config: OrchestratorConfig,
    has_warnings: bool = False,
) -> tuple[dict | None, list[str]]:
    """
    Generic retry loop. Returns (normalized_output_or_None, last_errors).
    Appends exactly one HistoryEntry per attempt. On attempt 2+, the
    payload_factory receives the previous raw output and previous errors
    so the model gets concrete feedback, not a repeat of the same request.
    """
    last_errors: list[str] = ["not attempted"]
    previous_raw: dict | None = None
    previous_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=None,
                errors=["per-case wall-clock budget exceeded before this attempt"],
                timed_out=True,
            ))
            return None, ["per-case wall-clock budget exceeded"]

        payload = payload_factory(previous_raw, previous_errors)
        call_timeout = min(config.per_call_timeout_seconds, remaining)

        try:
            raw = _call_with_hard_timeout(call_fn, system_prompt, payload, call_timeout)
        except FutureTimeoutError:
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=None,
                errors=[f"model call exceeded {call_timeout:.0f}s hard timeout"], timed_out=True,
            ))
            last_errors = ["model call timed out"]
            previous_raw, previous_errors = None, last_errors
            continue
        except Exception as e:  # noqa: BLE001 -- any call failure is a retryable attempt, never a crash
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=None,
                errors=[], exception=f"{type(e).__name__}: {e}",
            ))
            last_errors = [f"{type(e).__name__}: {e}"]
            previous_raw, previous_errors = None, last_errors
            continue

        # Re-check the deadline AFTER the call returns. A result that
        # arrives late is discarded, never validated/accepted/finalized.
        if time.monotonic() > deadline:
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=raw,
                errors=["result arrived after per-case deadline; discarded"], timed_out=True,
            ))
            return None, ["result arrived after per-case deadline"]

        try:
            if has_warnings:
                result = validate_fn(raw)
                normalized, errors, warnings, applied_fixes = result
            else:
                normalized, errors = validate_fn(raw)
                warnings, applied_fixes = [], []
        except Exception as e:  # noqa: BLE001 -- defense in depth; validators are also self-protecting
            errors = [f"validator crashed unexpectedly: {type(e).__name__}: {e}"]
            normalized, warnings, applied_fixes = {}, [], []

        history.append(HistoryEntry(
            stage=stage_name, attempt=attempt, raw_output=raw,
            errors=errors, warnings=warnings, applied_fixes=applied_fixes,
        ))

        if not errors:
            return normalized, []

        last_errors = errors
        previous_raw, previous_errors = raw, errors

    return None, last_errors


def _stage2_validate_with_finalizer(raw, stage1_observations, pkt_evidence) -> tuple[dict, list[str], list[str], list[str]]:
    """Adapter so the generic retry loop's has_warnings=True path also
    gets applied_fixes, keeping the full FinalizeResult in history."""
    result: FinalizeResult = attempt_safe_finalize(raw, stage1_observations, pkt_evidence)
    return result.normalized, result.remaining_errors, result.remaining_warnings, result.applied_fixes


def run_case(
    pkt: dict,
    numerical_prior: dict,
    stage1_call: StageCallable,
    stage2_call: StageCallable,
    stage3_call: StageCallable,
    stage1_system_prompt: str,
    stage2_system_prompt: str,
    stage3_system_prompt: str,
    build_stage1_payload: Callable[[dict], dict],
    build_stage2_payload: Callable[[dict, dict, dict], dict],
    build_stage3_payload: Callable[[dict, dict, dict], dict],
    config: OrchestratorConfig | None = None,
) -> CaseResult:
    config = config or OrchestratorConfig()
    start = time.monotonic()
    deadline = start + config.per_case_timeout_seconds
    history: list[HistoryEntry] = []
    case_id = pkt["case_id"]

    try:
        evidence_sets = derive_case_evidence_sets(pkt["evidence"])
    except Exception as e:  # noqa: BLE001
        return CaseResult(
            case_id=case_id, status="MODEL_CONTRACT_FAILURE",
            final_stage2_output=None, stage1_output=None, critic_history=[],
            raw_history=[HistoryEntry(stage="setup", attempt=0, raw_output=None,
                                       errors=[f"evidence set derivation failed: {e}"])],
            elapsed_seconds=time.monotonic() - start,
            failure_stage="setup", failure_reason=str(e),
        )
    case_available_ids = evidence_sets["case_available_evidence_ids"]

    # ---- Stage 1 -----------------------------------------------------
    def stage1_payload_factory(prev_raw, prev_errors):
        base = build_stage1_payload(pkt)
        if prev_raw is not None or prev_errors:
            base["previous_attempt"] = {"raw_output": prev_raw, "validation_errors": prev_errors}
            base["instruction"] = (
                base.get("instruction", "")
                + " The previous attempt failed contract validation with the errors listed in "
                  "previous_attempt.validation_errors. Fix exactly those problems."
            )
        return base

    stage1_output, stage1_errors = _run_stage_with_retries(
        "stage1", stage1_call, stage1_system_prompt, stage1_payload_factory,
        lambda raw: validate_stage1_output(raw, case_id, case_available_ids, pkt["evidence"]),
        config.stage1_max_attempts, history, deadline, config,
    )
    if stage1_output is None:
        return CaseResult(
            case_id=case_id, status="MODEL_CONTRACT_FAILURE",
            final_stage2_output=None, stage1_output=None, critic_history=[],
            raw_history=history, elapsed_seconds=time.monotonic() - start,
            failure_stage="stage1", failure_reason="; ".join(stage1_errors),
        )

    # ---- Stage 2 (format-only retry loop) -----------------------------
    def stage2_payload_factory(prev_raw, prev_errors):
        base = build_stage2_payload(pkt, stage1_output, numerical_prior)
        if prev_raw is not None or prev_errors:
            base["previous_attempt"] = {"raw_output": prev_raw, "validation_errors": prev_errors}
            base["instruction"] = (
                base.get("instruction", "")
                + " The previous attempt failed contract validation with the errors listed in "
                  "previous_attempt.validation_errors. Fix exactly those problems using only "
                  "evidence_observations already supplied; do not introduce new claims."
            )
        return base

    stage2_output, stage2_errors = _run_stage_with_retries(
        "stage2", stage2_call, stage2_system_prompt, stage2_payload_factory,
        lambda raw: _stage2_validate_with_finalizer(raw, stage1_output["evidence_observations"], pkt["evidence"]),
        config.stage2_max_format_attempts, history, deadline, config, has_warnings=True,
    )
    if stage2_output is None:
        return CaseResult(
            case_id=case_id, status="MODEL_CONTRACT_FAILURE",
            final_stage2_output=None, stage1_output=stage1_output, critic_history=[],
            raw_history=history, elapsed_seconds=time.monotonic() - start,
            failure_stage="stage2", failure_reason="; ".join(stage2_errors),
        )

    # ---- Stage 3 critic loop --------------------------------------------
    critic_history: list[dict] = []
    current_stage2 = stage2_output
    stage1_evidence_ids = [o["evidence_id"] for o in stage1_output["evidence_observations"]]

    for critic_round in range(1, config.critic_max_rounds + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason="per-case wall-clock budget exceeded",
            )

        det_precheck_summary = {"errors": [], "note": "structural checks already passed prior to critic"}
        payload = build_stage3_payload(stage1_output, current_stage2, det_precheck_summary)
        call_timeout = min(config.per_call_timeout_seconds, remaining)

        try:
            raw_critic = _call_with_hard_timeout(stage3_call, stage3_system_prompt, payload, call_timeout)
        except FutureTimeoutError:
            history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=None,
                                         errors=[f"model call exceeded {call_timeout:.0f}s hard timeout"], timed_out=True))
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason="model call timed out",
            )
        except Exception as e:  # noqa: BLE001
            history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=None,
                                         errors=[], exception=f"{type(e).__name__}: {e}"))
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason=f"{type(e).__name__}: {e}",
            )

        # Late-result discard: a critic APPROVE that arrives after the
        # deadline must never finalize the case.
        if time.monotonic() > deadline:
            history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=raw_critic,
                                         errors=["result arrived after per-case deadline; discarded"], timed_out=True))
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason="result arrived after per-case deadline",
            )

        try:
            normalized_critic, critic_errors = validate_stage3_output(raw_critic, stage1_evidence_ids)
        except Exception as e:  # noqa: BLE001 -- defense in depth
            normalized_critic, critic_errors = {}, [f"validator crashed unexpectedly: {type(e).__name__}: {e}"]

        history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=raw_critic, errors=critic_errors))

        if critic_errors:
            critic_history.append({"round": critic_round, "raw": raw_critic, "contract_errors": critic_errors})
            continue

        critic_history.append({"round": critic_round, "verdict": normalized_critic["verdict"], "problems": normalized_critic["problems"]})

        if normalized_critic["verdict"] == "APPROVE":
            return CaseResult(
                case_id=case_id, status="FINALIZED",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
            )

        if critic_round >= config.critic_max_rounds:
            break

        problems_for_revision = normalized_critic["problems"]

        def stage2_revision_payload_factory(prev_raw, prev_errors, _problems=problems_for_revision):
            base = build_stage2_payload(pkt, stage1_output, numerical_prior)
            base["critic_feedback"] = _problems
            instr = (
                " An independent critic returned verdict=REVISE with the problems listed in "
                "critic_feedback. Address every listed problem using only evidence_observations "
                "already supplied, without introducing new unsupported claims."
            )
            if prev_raw is not None or prev_errors:
                base["previous_attempt"] = {"raw_output": prev_raw, "validation_errors": prev_errors}
                instr += (
                    " Your previous revision attempt ALSO failed contract validation with the "
                    "errors in previous_attempt.validation_errors -- fix those too."
                )
            base["instruction"] = base.get("instruction", "") + instr
            return base

        revised_stage2, revised_errors = _run_stage_with_retries(
            "stage2_revision", stage2_call, stage2_system_prompt, stage2_revision_payload_factory,
            lambda raw: _stage2_validate_with_finalizer(raw, stage1_output["evidence_observations"], pkt["evidence"]),
            config.stage2_max_format_attempts, history, deadline, config, has_warnings=True,
        )
        if revised_stage2 is None:
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage2_revision", failure_reason="; ".join(revised_errors),
            )
        current_stage2 = revised_stage2

    return CaseResult(
        case_id=case_id, status="MODEL_CONTRACT_FAILURE",
        final_stage2_output=current_stage2, stage1_output=stage1_output,
        critic_history=critic_history, raw_history=history,
        elapsed_seconds=time.monotonic() - start,
        failure_stage="stage3", failure_reason="critic_max_rounds exhausted without APPROVE",
    )


def run_batch(
    packets: list[dict],
    numerical_priors: dict,
    stage1_call: StageCallable,
    stage2_call: StageCallable,
    stage3_call: StageCallable,
    stage1_system_prompt: str,
    stage2_system_prompt: str,
    stage3_system_prompt: str,
    build_stage1_payload: Callable[[dict], dict],
    build_stage2_payload: Callable[[dict, dict, dict], dict],
    build_stage3_payload: Callable[[dict, dict, dict], dict],
    config: OrchestratorConfig | None = None,
) -> list[CaseResult]:
    """
    Runs run_case() independently for every packet. Each call is wrapped
    in its own try/except: an unexpected bug in ANY single case (not just
    a validator's normal contract-error path, which never raises, but a
    genuinely unforeseen exception anywhere in run_case's own logic) is
    caught here and converted into a MODEL_CONTRACT_FAILURE for that case
    only. It can never abort or skip any other case in the batch.
    """
    results = []
    for pkt in packets:
        prior = numerical_priors.get(pkt["case_id"], {})
        try:
            result = run_case(
                pkt, prior, stage1_call, stage2_call, stage3_call,
                stage1_system_prompt, stage2_system_prompt, stage3_system_prompt,
                build_stage1_payload, build_stage2_payload, build_stage3_payload,
                config,
            )
        except Exception as e:  # noqa: BLE001 -- outermost per-case isolation
            result = CaseResult(
                case_id=pkt.get("case_id", -1), status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=None, stage1_output=None, critic_history=[],
                raw_history=[HistoryEntry(stage="run_case", attempt=0, raw_output=None,
                                           errors=[], exception=f"{type(e).__name__}: {e}")],
                elapsed_seconds=0.0, failure_stage="run_case",
                failure_reason=f"unexpected exception escaped run_case: {type(e).__name__}: {e}",
            )
        results.append(result)
    return results
