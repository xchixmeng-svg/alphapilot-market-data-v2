"""
retry_orchestrator.py

REWRITTEN after real-model canary runs 35482706294 / 35485028854.

STRUCTURAL FIX (review point 1): Stage 2 is now driven through TWO
possible model calls per attempt, never one call that has to get both
"what's the decision" and "what's the actionable price, if any" right at
once:

    stage2_decision_call   -- always called. Produces decision, thesis,
        evidence roles. Never entry/failure_exit.
    stage2_actionability_call -- called ONLY if the decision that came
        back is CANDIDATE/HIGH_CONVICTION. Produces entry/failure_exit
        ONLY, with an enum that cannot express NOT_ACTIONABLE/UNAVAILABLE.

For REJECT/WATCH, entry/failure_exit is assembled deterministically by
safe_finalizer.assemble_non_actionable_stage2() with NO model call at
all -- see that module's docstring for why this is assembly, not
invention. This also means the common case (REJECT/WATCH, which the
original decision-collapse investigation found to be ~94% of cases) now
costs FEWER model calls than the old single-call design, not more.

STRUCTURED REPAIR REQUESTS (review point 5): every retry payload now
carries BOTH the raw previous errors (as before) AND a compact structured
`repair_request` list from deterministic_validators.errors_to_repair_
request(), so the model gets a short, mechanical "what exactly to change"
alongside the full text.

PER-STAGE LATENCY (review point 7): HistoryEntry now has an
elapsed_seconds field, populated around every individual model call, so
summarize_canary_results.py can report real per-stage latency/call-count
breakdowns, not just per-case totals.

TIMEOUT BUDGET RECALIBRATED (review point 7): the real 2-round canary
measured a 316.47s mean per-case elapsed time with ZERO finalized cases
-- i.e. the old budget was being burned entirely on doomed retries, not
on legitimate model latency. Worst-case call count under the new design
is higher on paper (stage1 <=2, stage2a <=2, stage2b <=2, critic 1,
revision stage2a <=2, revision stage2b <=2, critic 1 = <=12 calls), so
per_case_timeout_seconds is raised accordingly (see OrchestratorConfig),
but the STRUCTURAL fixes above mean the common REJECT/WATCH case no
longer needs stage2b or usually needs a full revision round, so real
elapsed time for most cases should fall, not rise.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Callable

from deterministic_validators import (
    ACTIONABLE_DECISIONS,
    NON_ACTIONABLE_DECISIONS,
    errors_to_repair_request,
    validate_stage1_output,
    validate_stage2_actionability_output,
    validate_stage2_decision_output,
    validate_stage3_output,
)
from evidence_capability import derive_case_evidence_sets
from safe_finalizer import assemble_actionable_stage2, assemble_non_actionable_stage2

StageCallable = Callable[[str, dict], dict]
PayloadFactory = Callable[[dict | None, list[str]], dict]

_executor = ThreadPoolExecutor(max_workers=8)


@dataclass
class OrchestratorConfig:
    stage1_max_attempts: int = 2
    stage2_decision_max_attempts: int = 2
    stage2_actionability_max_attempts: int = 2
    critic_max_rounds: int = 2

    # Derivation: reported real single Qwen2.5-7B call latency ~149s.
    per_call_timeout_seconds: float = 180.0

    # Worst-case call count for one case under the split-Stage2 design:
    #   stage1(<=2) + stage2a(<=2) + stage2b(<=2, only if actionable)
    #   + critic round1(1) + revision stage2a(<=2) + revision stage2b(<=2)
    #   + critic round2(1) = <=12 calls. At <=180s each: <=2160s. Budget
    # is set with headroom above that worst case.
    per_case_timeout_seconds: float = 2400.0


@dataclass
class HistoryEntry:
    stage: str  # "stage1" | "stage2_decision" | "stage2_actionability" | "stage3" | "*_revision"
    attempt: int
    raw_output: dict | None
    errors: list[str]
    warnings: list[str] = field(default_factory=list)
    applied_fixes: list[str] = field(default_factory=list)
    exception: str | None = None
    timed_out: bool = False
    elapsed_seconds: float = 0.0


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
    last_errors: list[str] = ["not attempted"]
    previous_raw: dict | None = None
    previous_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=None,
                errors=["per-case wall-clock budget exceeded before this attempt"], timed_out=True,
            ))
            return None, ["per-case wall-clock budget exceeded"]

        payload = payload_factory(previous_raw, previous_errors)
        call_timeout = min(config.per_call_timeout_seconds, remaining)
        call_start = time.monotonic()

        try:
            raw = _call_with_hard_timeout(call_fn, system_prompt, payload, call_timeout)
        except FutureTimeoutError:
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=None,
                errors=[f"model call exceeded {call_timeout:.0f}s hard timeout"], timed_out=True,
                elapsed_seconds=time.monotonic() - call_start,
            ))
            last_errors = ["model call timed out"]
            previous_raw, previous_errors = None, last_errors
            continue
        except Exception as e:  # noqa: BLE001
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=None, errors=[],
                exception=f"{type(e).__name__}: {e}", elapsed_seconds=time.monotonic() - call_start,
            ))
            last_errors = [f"{type(e).__name__}: {e}"]
            previous_raw, previous_errors = None, last_errors
            continue

        call_elapsed = time.monotonic() - call_start

        if time.monotonic() > deadline:
            history.append(HistoryEntry(
                stage=stage_name, attempt=attempt, raw_output=raw,
                errors=["result arrived after per-case deadline; discarded"], timed_out=True,
                elapsed_seconds=call_elapsed,
            ))
            return None, ["result arrived after per-case deadline"]

        try:
            if has_warnings:
                normalized, errors, warnings = validate_fn(raw)
            else:
                normalized, errors = validate_fn(raw)
                warnings = []
        except Exception as e:  # noqa: BLE001
            errors = [f"validator crashed unexpectedly: {type(e).__name__}: {e}"]
            normalized, warnings = {}, []

        history.append(HistoryEntry(
            stage=stage_name, attempt=attempt, raw_output=raw, errors=errors, warnings=warnings,
            elapsed_seconds=call_elapsed,
        ))

        if not errors:
            return normalized, []

        last_errors = errors
        previous_raw, previous_errors = raw, errors

    return None, last_errors


def _with_repair_request(base_payload: dict, previous_raw: dict | None, previous_errors: list[str], extra_instruction: str) -> dict:
    if previous_raw is not None or previous_errors:
        base_payload["previous_attempt"] = {"raw_output": previous_raw, "validation_errors": previous_errors}
        base_payload["repair_request"] = errors_to_repair_request(previous_errors)
        base_payload["instruction"] = base_payload.get("instruction", "") + " " + extra_instruction
    return base_payload


def _run_stage2_pipeline(
    pkt: dict,
    stage1_output: dict,
    numerical_prior: dict,
    stage2_decision_call: StageCallable,
    stage2_actionability_call: StageCallable,
    stage2_decision_system_prompt: str,
    stage2_actionability_system_prompt: str,
    build_stage2_decision_payload: Callable[[dict, dict, dict], dict],
    build_stage2_actionability_payload: Callable[[dict, dict, dict, dict], dict],
    history: list[HistoryEntry],
    deadline: float,
    config: OrchestratorConfig,
    critic_feedback: list[dict] | None = None,
    stage_prefix: str = "stage2",
) -> tuple[dict | None, list[str]]:
    """
    Runs the decision call, then (if actionable) the actionability call,
    then deterministically assembles the final Stage2 dict. Used for both
    the initial pass and for a critic-triggered revision pass (via
    critic_feedback + stage_prefix="stage2_revision").
    """

    def decision_payload_factory(prev_raw, prev_errors):
        base = build_stage2_decision_payload(pkt, stage1_output, numerical_prior)
        if critic_feedback:
            base["critic_feedback"] = critic_feedback
            base["instruction"] = (
                base.get("instruction", "")
                + " An independent critic returned verdict=REVISE with the problems in critic_feedback. "
                  "Address every listed problem using only evidence_observations already supplied."
            )
        return _with_repair_request(
            base, prev_raw, prev_errors,
            "The previous attempt failed contract validation. repair_request lists exactly what to change, "
            "field by field -- apply every item in it. Do not repeat the same invalid value.",
        )

    decision_output, decision_errors = _run_stage_with_retries(
        f"{stage_prefix}_decision", stage2_decision_call, stage2_decision_system_prompt,
        decision_payload_factory,
        lambda raw: validate_stage2_decision_output(raw, stage1_output["evidence_observations"], pkt["evidence"]),
        config.stage2_decision_max_attempts, history, deadline, config, has_warnings=True,
    )
    if decision_output is None:
        return None, decision_errors

    decision = decision_output["decision"]

    if decision in NON_ACTIONABLE_DECISIONS:
        assembled = assemble_non_actionable_stage2(decision_output)
        return assembled.combined, []

    def actionability_payload_factory(prev_raw, prev_errors):
        base = build_stage2_actionability_payload(pkt, stage1_output, decision_output, numerical_prior)
        return _with_repair_request(
            base, prev_raw, prev_errors,
            "The previous attempt failed contract validation. repair_request lists exactly what to change. "
            "entry.status may NOT be NOT_ACTIONABLE and failure_exit.trigger_type may NOT be UNAVAILABLE in "
            "this call -- this call is only made because the decision is actionable.",
        )

    actionability_output, actionability_errors = _run_stage_with_retries(
        f"{stage_prefix}_actionability", stage2_actionability_call, stage2_actionability_system_prompt,
        actionability_payload_factory,
        lambda raw: validate_stage2_actionability_output(
            raw, stage1_output["evidence_observations"], pkt["evidence"]
        ),
        config.stage2_actionability_max_attempts, history, deadline, config, has_warnings=False,
    )
    if actionability_output is None:
        return None, actionability_errors

    assembled = assemble_actionable_stage2(decision_output, actionability_output)
    return assembled.combined, []


def run_case(
    pkt: dict,
    numerical_prior: dict,
    stage1_call: StageCallable,
    stage2_decision_call: StageCallable,
    stage2_actionability_call: StageCallable,
    stage3_call: StageCallable,
    stage1_system_prompt: str,
    stage2_decision_system_prompt: str,
    stage2_actionability_system_prompt: str,
    stage3_system_prompt: str,
    build_stage1_payload: Callable[[dict], dict],
    build_stage2_decision_payload: Callable[[dict, dict, dict], dict],
    build_stage2_actionability_payload: Callable[[dict, dict, dict, dict], dict],
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
            elapsed_seconds=time.monotonic() - start, failure_stage="setup", failure_reason=str(e),
        )
    case_available_ids = evidence_sets["case_available_evidence_ids"]

    # ---- Stage 1 -----------------------------------------------------
    def stage1_payload_factory(prev_raw, prev_errors):
        base = build_stage1_payload(pkt)
        return _with_repair_request(
            base, prev_raw, prev_errors,
            "Fix precisely the errors listed in previous_attempt.validation_errors / repair_request.",
        )

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

    # ---- Stage 2 (decision, then conditionally actionability) --------
    current_stage2, stage2_errors = _run_stage2_pipeline(
        pkt, stage1_output, numerical_prior, stage2_decision_call, stage2_actionability_call,
        stage2_decision_system_prompt, stage2_actionability_system_prompt,
        build_stage2_decision_payload, build_stage2_actionability_payload,
        history, deadline, config,
    )
    if current_stage2 is None:
        return CaseResult(
            case_id=case_id, status="MODEL_CONTRACT_FAILURE",
            final_stage2_output=None, stage1_output=stage1_output, critic_history=[],
            raw_history=history, elapsed_seconds=time.monotonic() - start,
            failure_stage="stage2", failure_reason="; ".join(stage2_errors),
        )

    # ---- Stage 3 critic loop --------------------------------------------
    critic_history: list[dict] = []
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
        call_start = time.monotonic()

        try:
            raw_critic = _call_with_hard_timeout(stage3_call, stage3_system_prompt, payload, call_timeout)
        except FutureTimeoutError:
            history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=None,
                                         errors=[f"model call exceeded {call_timeout:.0f}s hard timeout"],
                                         timed_out=True, elapsed_seconds=time.monotonic() - call_start))
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason="model call timed out",
            )
        except Exception as e:  # noqa: BLE001
            history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=None, errors=[],
                                         exception=f"{type(e).__name__}: {e}", elapsed_seconds=time.monotonic() - call_start))
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason=f"{type(e).__name__}: {e}",
            )

        call_elapsed = time.monotonic() - call_start

        if time.monotonic() > deadline:
            history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=raw_critic,
                                         errors=["result arrived after per-case deadline; discarded"],
                                         timed_out=True, elapsed_seconds=call_elapsed))
            return CaseResult(
                case_id=case_id, status="MODEL_CONTRACT_FAILURE",
                final_stage2_output=current_stage2, stage1_output=stage1_output,
                critic_history=critic_history, raw_history=history,
                elapsed_seconds=time.monotonic() - start,
                failure_stage="stage3", failure_reason="result arrived after per-case deadline",
            )

        try:
            normalized_critic, critic_errors = validate_stage3_output(raw_critic, stage1_evidence_ids)
        except Exception as e:  # noqa: BLE001
            normalized_critic, critic_errors = {}, [f"validator crashed unexpectedly: {type(e).__name__}: {e}"]

        history.append(HistoryEntry(stage="stage3", attempt=critic_round, raw_output=raw_critic,
                                     errors=critic_errors, elapsed_seconds=call_elapsed))

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

        revised_stage2, revised_errors = _run_stage2_pipeline(
            pkt, stage1_output, numerical_prior, stage2_decision_call, stage2_actionability_call,
            stage2_decision_system_prompt, stage2_actionability_system_prompt,
            build_stage2_decision_payload, build_stage2_actionability_payload,
            history, deadline, config,
            critic_feedback=normalized_critic["problems"], stage_prefix="stage2_revision",
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
    stage2_decision_call: StageCallable,
    stage2_actionability_call: StageCallable,
    stage3_call: StageCallable,
    stage1_system_prompt: str,
    stage2_decision_system_prompt: str,
    stage2_actionability_system_prompt: str,
    stage3_system_prompt: str,
    build_stage1_payload: Callable[[dict], dict],
    build_stage2_decision_payload: Callable[[dict, dict, dict], dict],
    build_stage2_actionability_payload: Callable[[dict, dict, dict, dict], dict],
    build_stage3_payload: Callable[[dict, dict, dict], dict],
    config: OrchestratorConfig | None = None,
) -> list[CaseResult]:
    results = []
    for pkt in packets:
        prior = numerical_priors.get(pkt["case_id"], {})
        try:
            result = run_case(
                pkt, prior, stage1_call, stage2_decision_call, stage2_actionability_call, stage3_call,
                stage1_system_prompt, stage2_decision_system_prompt, stage2_actionability_system_prompt,
                stage3_system_prompt, build_stage1_payload, build_stage2_decision_payload,
                build_stage2_actionability_payload, build_stage3_payload, config,
            )
        except Exception as e:  # noqa: BLE001
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
