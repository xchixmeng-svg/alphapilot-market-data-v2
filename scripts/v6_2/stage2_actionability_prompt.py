"""
stage2_actionability_prompt.py

Stage 2b: ACTIONABILITY-ONLY. Called ONLY after Stage 2a's decision comes
back CANDIDATE or HIGH_CONVICTION. Asks for entry/failure_exit ONLY --
the decision, thesis, and evidence roles are already fixed by Stage 2a
and are supplied here as read-only context, not something this call can
change.

The schema for this call structurally excludes entry.status=NOT_ACTIONABLE
and failure_exit.trigger_type=UNAVAILABLE -- those values are not valid
options in this call's enum at all, because this call only ever happens
when the decision is already known to be actionable. This is the other
half of the fix for the reported CANDIDATE+NOT_ACTIONABLE+null-price
failure: it is now impossible for this specific call to produce that
combination, because half of it (the non-actionable values) does not
exist in its vocabulary.
"""

from __future__ import annotations

STAGE2_ACTIONABILITY_SYSTEM = """You are the AlphaPilot V6.2 Stage 2 actionability layer (actionability-only sub-stage).

A separate decision layer has already determined this case's decision is CANDIDATE or HIGH_CONVICTION, based on the thesis and evidence supplied to you below as read-only context. Your ONLY job is to state the actionable entry zone and failure-exit price. You are NOT re-deciding whether this stock is a candidate -- that has already been settled; do not second-guess it here.

Because the decision is already known to be actionable, entry.status can ONLY be one of NOW / WAIT_FOR_PULLBACK / WAIT_FOR_CONFIRMATION -- there is no NOT_ACTIONABLE option in this call, because that would contradict the decision already made. Likewise failure_exit.trigger_type can ONLY be one of PRICE_STRUCTURE_BREAK / THESIS_INVALIDATION / CATALYST_FAILURE / VALUATION_EXPECTATION_BREAK / MULTI_EVIDENCE_FAILURE -- there is no UNAVAILABLE option here.

entry.ideal_low and entry.ideal_high MUST be real numbers you derive from the actual numbers present in the price_volume_structure evidence_observation's exact_observations (supplied below) -- never invented, never a placeholder, never null. failure_exit.exit_price MUST likewise be a real, evidence-grounded price derived from those same numbers (never a generic fixed percentage, never null). A deterministic check will reject any price that is not within a plausible range of the numbers Stage 1 actually observed for price_volume_structure.

failure_exit.reason must be a non-empty, specific statement of what price/structure event would invalidate the thesis -- not a generic phrase.

If your input contains a `previous_attempt` field, it holds your own prior response and the exact contract errors it violated. A `repair_request` field, if present, is a compact machine-generated list of exactly what to change -- apply every item in it.

Return exactly one JSON object with exactly two top-level keys, `entry` and `failure_exit`, matching the supplied schema, and no markdown. Do not include decision, thesis, or any other field from the decision layer -- this call only ever answers with entry/failure_exit."""


def build_stage2_actionability_user_payload(pkt: dict, stage1_output: dict, decision_output: dict, numerical_prior: dict) -> dict:
    return {
        "case_id": pkt["case_id"],
        "decision": decision_output["decision"],
        "evidence_quality": decision_output["evidence_quality"],
        "hypothesis": decision_output["hypothesis"],
        "bull_thesis": decision_output["bull_thesis"],
        "primary_evidence_ids": decision_output["primary_evidence_ids"],
        "evidence_observations": stage1_output["evidence_observations"],
        "numerical_prior": numerical_prior,
        "instruction": (
            "State entry.ideal_low/ideal_high and failure_exit.exit_price/reason/trigger_type, derived "
            "strictly from the price_volume_structure evidence_observation's exact_observations above. "
            "Do not include decision, thesis, or evidence-id fields -- this call only answers with entry "
            "and failure_exit."
        ),
    }
