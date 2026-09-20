"""
evidence_capability.py

Single source of truth for the V6.2 Evidence Capability Registry, and the
derivation logic that separates three distinct evidence-availability
concepts that were previously conflated into one "missing_evidence_ids"
list:

    1. system_unavailable_evidence_ids
       Evidence families the V6.2 data pipeline does not collect for ANY
       stock, at any decision date (currently: eps_revisions,
       analyst_consensus, industry_pricing, inventory_supply_demand,
       broad_news_semantics). This is a permanent, case-independent
       background limitation. It must never function as counter-evidence
       and must never by itself justify WATCH or REJECT.

    2. case_available_evidence_ids
       The subset of SYSTEM_AVAILABLE_FAMILIES that happens to have real
       data for this specific stock/decision-date (e.g. institutional_flow
       is system-supported, but 2 of the 64 replay cases have no
       institutional data for that specific date).

    3. case_missing_but_system_supported_evidence_ids
       SYSTEM_AVAILABLE_FAMILIES minus case_available_evidence_ids. This
       IS meaningful case-specific evidence (its absence can legitimately
       inform WATCH/REJECT), unlike system_unavailable_evidence_ids.

This module does not touch 0A-0F, R10, or any numerical model. It is pure
evidence-bookkeeping logic consumed by the prompt-construction step and by
normalize_output()/validate() in normalize_and_validate.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Capability Registry (authoritative; matches the 10-family Evidence
# Capability Registry currently deployed in production).
# ---------------------------------------------------------------------------

SYSTEM_AVAILABLE_FAMILIES: frozenset[str] = frozenset(
    {
        "price_volume_structure",
        "valuation",
        "revenue",
        "institutional_flow",
        "mops_material_information",
    }
)

SYSTEM_UNAVAILABLE_FAMILIES: frozenset[str] = frozenset(
    {
        "eps_revisions",
        "analyst_consensus",
        "industry_pricing",
        "inventory_supply_demand",
        "broad_news_semantics",
    }
)

ALL_REGISTRY_FAMILIES: frozenset[str] = SYSTEM_AVAILABLE_FAMILIES | SYSTEM_UNAVAILABLE_FAMILIES


class EvidenceRegistryError(RuntimeError):
    """Raised when a case packet's evidence-availability fields are
    inconsistent with the authoritative Capability Registry. This is a
    data-integrity condition and must never be silently repaired."""


def derive_case_evidence_sets(pkt_evidence: dict) -> dict[str, list[str]]:
    """
    Given a case packet's `evidence` block (with `available_evidence_ids`
    and `missing_evidence_ids` as provided today), derive the three
    distinct evidence sets described in the module docstring.

    This performs a cross-check: case_missing_but_system_supported can be
    derived two independent ways (from the registry, and from the packet's
    own raw `missing_evidence_ids` minus the known system-unavailable
    families). If the two derivations disagree, the packet's evidence
    bookkeeping is internally inconsistent with the registry and this
    function raises EvidenceRegistryError rather than silently picking one
    side.

    Returns a dict with sorted lists (sorted for deterministic output /
    reproducibility):
        system_unavailable_evidence_ids
        case_available_evidence_ids
        case_missing_but_system_supported_evidence_ids
    """
    raw_available = set(pkt_evidence.get("available_evidence_ids") or [])
    raw_missing = set(pkt_evidence.get("missing_evidence_ids") or [])

    unknown_available = raw_available - ALL_REGISTRY_FAMILIES
    if unknown_available:
        raise EvidenceRegistryError(
            f"available_evidence_ids contains families not in the Capability "
            f"Registry: {sorted(unknown_available)}"
        )
    unknown_missing = raw_missing - ALL_REGISTRY_FAMILIES
    if unknown_missing:
        raise EvidenceRegistryError(
            f"missing_evidence_ids contains families not in the Capability "
            f"Registry: {sorted(unknown_missing)}"
        )

    case_available_from_available_list = raw_available & SYSTEM_AVAILABLE_FAMILIES
    case_missing_but_supported_from_registry = SYSTEM_AVAILABLE_FAMILIES - raw_available
    case_missing_but_supported_from_raw_missing = raw_missing - SYSTEM_UNAVAILABLE_FAMILIES

    if case_missing_but_supported_from_registry != case_missing_but_supported_from_raw_missing:
        raise EvidenceRegistryError(
            "Inconsistent evidence bookkeeping for this case: "
            f"SYSTEM_AVAILABLE_FAMILIES - available_evidence_ids = "
            f"{sorted(case_missing_but_supported_from_registry)}, but "
            f"missing_evidence_ids - SYSTEM_UNAVAILABLE_FAMILIES = "
            f"{sorted(case_missing_but_supported_from_raw_missing)}. "
            "These must be identical; the packet's available/missing lists "
            "disagree with the Capability Registry."
        )

    # A family should never appear in both available and missing for the
    # same case -- that would be a direct contradiction in the source data.
    contradictory = raw_available & raw_missing
    if contradictory:
        raise EvidenceRegistryError(
            f"Evidence family listed as BOTH available and missing for the "
            f"same case: {sorted(contradictory)}"
        )

    return {
        "system_unavailable_evidence_ids": sorted(SYSTEM_UNAVAILABLE_FAMILIES),
        "case_available_evidence_ids": sorted(case_available_from_available_list),
        "case_missing_but_system_supported_evidence_ids": sorted(
            case_missing_but_supported_from_registry
        ),
    }
