"""
safe_finalizer.py

REWRITTEN after real-model canary runs 35482706294 / 35485028854 showed
the reported "decision=CANDIDATE + entry.status=NOT_ACTIONABLE + null
prices" failure. The old version of this module tried to *repair* a
single combined Stage2 response after the fact. The new architecture
(deterministic_validators.py's validate_stage2_decision_output /
validate_stage2_actionability_output split) makes that repair
unnecessary for the REJECT/WATCH case, and impossible-by-construction for
the CANDIDATE/HIGH_CONVICTION case:

  - For REJECT/WATCH: entry/failure_exit is not asked of the model at
    all. It is ALWAYS exactly the same fixed value (NOT_ACTIONABLE / null
    / null / UNAVAILABLE) regardless of the case, so assembling it
    deterministically is not "guessing" or "inventing" anything -- there
    is only one value the contract permits, and it does not depend on any
    case-specific fact the model would need to supply.

  - For CANDIDATE/HIGH_CONVICTION: entry/failure_exit comes from a
    SEPARATE model call (validate_stage2_actionability_output) whose
    schema/enum structurally excludes NOT_ACTIONABLE/UNAVAILABLE, so the
    combination the review reported cannot recur by construction. This
    module still does zero repair on that response -- if it fails
    validation, retry_orchestrator.py retries the actionability call
    itself with the same errors, this module never touches the content.

This module's remaining job is pure, audit-logged assembly: combine a
validated decision-output dict with either the canonical null block or a
validated actionability-output dict into the final Stage2 shape the rest
of the pipeline (Stage 3 critic, output files) expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deterministic_validators import ACTIONABLE_DECISIONS, NON_ACTIONABLE_DECISIONS

CANONICAL_NON_ACTIONABLE_ENTRY = {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None}
CANONICAL_NON_ACTIONABLE_FAILURE_EXIT = {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"}


@dataclass
class AssembledStage2:
    combined: dict
    assembly_note: str
    applied_fixes: list[str] = field(default_factory=list)


def assemble_non_actionable_stage2(decision_output: dict) -> AssembledStage2:
    """decision_output must already be a VALIDATED (zero-error) output of
    validate_stage2_decision_output(), with decision in NON_ACTIONABLE_DECISIONS."""
    if decision_output.get("decision") not in NON_ACTIONABLE_DECISIONS:
        raise ValueError(
            f"assemble_non_actionable_stage2 called with decision={decision_output.get('decision')!r}, "
            f"which is not in NON_ACTIONABLE_DECISIONS. This is a caller bug, not a model-output problem."
        )
    combined = dict(decision_output)
    combined["entry"] = dict(CANONICAL_NON_ACTIONABLE_ENTRY)
    combined["failure_exit"] = dict(CANONICAL_NON_ACTIONABLE_FAILURE_EXIT)
    return AssembledStage2(
        combined=combined,
        assembly_note=(
            "entry/failure_exit deterministically set to the canonical null form for "
            f"decision={combined['decision']}; no model call was made for these fields."
        ),
        applied_fixes=["entry/failure_exit set to canonical null form (deterministic, not model-supplied)"],
    )


def assemble_actionable_stage2(decision_output: dict, actionability_output: dict) -> AssembledStage2:
    """Both arguments must already be VALIDATED (zero-error) outputs of
    their respective validators, with decision_output['decision'] in
    ACTIONABLE_DECISIONS."""
    if decision_output.get("decision") not in ACTIONABLE_DECISIONS:
        raise ValueError(
            f"assemble_actionable_stage2 called with decision={decision_output.get('decision')!r}, "
            f"which is not in ACTIONABLE_DECISIONS. This is a caller bug, not a model-output problem."
        )
    combined = dict(decision_output)
    combined["entry"] = dict(actionability_output["entry"])
    combined["failure_exit"] = dict(actionability_output["failure_exit"])
    return AssembledStage2(
        combined=combined,
        assembly_note="entry/failure_exit taken verbatim from the validated Stage2b actionability response.",
        applied_fixes=[],
    )
