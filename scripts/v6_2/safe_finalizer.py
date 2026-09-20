"""
safe_finalizer.py

Implements the boundary from design section C: what deterministic code is
allowed to silently fix, and what it must never touch.

ALLOWED (format-only, cannot change the decision's meaning):
  - enum case normalization (already done in validate_stage2_output's
    normalization step, e.g. 'reject' -> 'REJECT')
  - whitespace trimming on string fields
  - de-duplicating evidence-id lists while preserving order
  - forcing entry.{status,ideal_low,ideal_high} and failure_exit.{exit_price,
    reason,trigger_type} to their canonical null/UNAVAILABLE form for
    REJECT/WATCH -- ONLY IF the only violation(s) detected are exactly
    those null-field-contract violations, and the model's own `decision`
    field is left completely untouched.

NEVER ALLOWED (would change the substance of the decision):
  - changing `decision` itself (e.g. WATCH -> CANDIDATE)
  - inventing or altering failure_exit.exit_price, ideal_low/ideal_high
  - inventing counter_evidence_ids, primary_evidence_ids, or
    secondary_evidence_ids that the model did not supply
  - rewriting bull_thesis / bear_thesis / decision_reason / hypothesis /
    invalidation text
  - resolving a Stage 3 REVISE verdict by editing the flagged claim itself
    -- only Stage 2 (the model) may revise its own thesis text; the
    finalizer never touches content

If finalize_watch_reject_nullfields() is applied, the change is always
logged in the returned FinalizeResult.applied_fixes list so downstream
audit can see exactly what deterministic code changed and why -- silent
repair is never acceptable even for allowed fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deterministic_validators import (
    ACTIONABLE_DECISIONS,
    NON_ACTIONABLE_DECISIONS,
    validate_stage2_output,
)


@dataclass
class FinalizeResult:
    normalized: dict
    applied_fixes: list[str] = field(default_factory=list)
    remaining_errors: list[str] = field(default_factory=list)
    remaining_warnings: list[str] = field(default_factory=list)

    @property
    def is_finalizable(self) -> bool:
        return len(self.remaining_errors) == 0


def _only_nullfield_violations(errors: list[str]) -> bool:
    """
    True if every error string in `errors` is one of the specific, narrow
    null-field-contract violations that finalize_watch_reject_nullfields()
    is permitted to fix. Any other error (evidence-id leakage, value-
    judgment-without-benchmark, decision/evidence_quality enum problems,
    empty text fields, etc.) means this function returns False and the
    finalizer must NOT touch anything -- the case goes back for a real
    model revision instead.
    """
    allowed_substrings = (
        "requires entry.status == 'NOT_ACTIONABLE'",
        "requires entry.ideal_low/ideal_high == null",
        "requires failure_exit.exit_price == null",
        "requires failure_exit.trigger_type == 'UNAVAILABLE'",
    )
    return all(any(sub in e for sub in allowed_substrings) for e in errors)


def finalize_watch_reject_nullfields(
    normalized_stage2: dict, errors: list[str]
) -> FinalizeResult:
    """
    If `errors` contains ONLY null-field-contract violations for a
    REJECT/WATCH decision, force entry/failure_exit into canonical null
    form and return a result with zero remaining errors. Otherwise, return
    the input unchanged with all errors preserved -- this function refuses
    to guess at anything beyond the exact allowed fix.
    """
    decision = normalized_stage2.get("decision")
    if decision not in NON_ACTIONABLE_DECISIONS:
        # This fix only ever applies to REJECT/WATCH; for any other
        # decision, or an unrecognized one, do nothing.
        return FinalizeResult(normalized=normalized_stage2, remaining_errors=list(errors))

    if not errors:
        return FinalizeResult(normalized=normalized_stage2)

    if not _only_nullfield_violations(errors):
        return FinalizeResult(normalized=normalized_stage2, remaining_errors=list(errors))

    fixed = dict(normalized_stage2)
    applied = []

    if fixed.get("entry", {}).get("status") != "NOT_ACTIONABLE":
        fixed["entry"] = {**fixed.get("entry", {}), "status": "NOT_ACTIONABLE"}
        applied.append("entry.status forced to NOT_ACTIONABLE")
    if fixed.get("entry", {}).get("ideal_low") is not None or fixed.get("entry", {}).get("ideal_high") is not None:
        fixed["entry"] = {**fixed["entry"], "ideal_low": None, "ideal_high": None}
        applied.append("entry.ideal_low/ideal_high forced to null")
    if fixed.get("failure_exit", {}).get("exit_price") is not None:
        fixed["failure_exit"] = {**fixed.get("failure_exit", {}), "exit_price": None}
        applied.append("failure_exit.exit_price forced to null")
    if fixed.get("failure_exit", {}).get("trigger_type") != "UNAVAILABLE":
        fixed["failure_exit"] = {**fixed["failure_exit"], "trigger_type": "UNAVAILABLE"}
        applied.append("failure_exit.trigger_type forced to UNAVAILABLE")

    return FinalizeResult(normalized=fixed, applied_fixes=applied)


def attempt_safe_finalize(
    raw_stage2: dict, stage1_observations: list[dict], pkt_evidence: dict
) -> FinalizeResult:
    """
    Full entry point: validate, and if the ONLY problems are null-field
    contract violations on a REJECT/WATCH case, apply the safe fix.
    Otherwise return the validation result unchanged (including any
    warnings) so the orchestrator knows a real model revision is needed.
    """
    normalized, errors, warnings = validate_stage2_output(raw_stage2, stage1_observations, pkt_evidence)
    if not errors:
        return FinalizeResult(normalized=normalized, remaining_warnings=warnings)

    result = finalize_watch_reject_nullfields(normalized, errors)
    result.remaining_warnings = warnings
    return result
