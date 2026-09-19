#!/usr/bin/env python3
"""Evidence-capability bookkeeping for the V6.2 decision layer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvidenceRegistryError(RuntimeError):
    pass


def _registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "research" / "V6_2_EVIDENCE_CAPABILITY_REGISTRY.json"


def load_capability_registry(path: Path | None = None) -> dict[str, Any]:
    source = path or _registry_path()
    obj = json.loads(source.read_text(encoding="utf-8"))
    families = obj.get("families")
    if not isinstance(families, dict) or not families:
        raise EvidenceRegistryError(f"invalid evidence registry: {source}")
    bad = sorted(k for k, v in families.items() if v.get("status") not in {"AVAILABLE", "PARTIAL", "UNAVAILABLE"})
    if bad:
        raise EvidenceRegistryError(f"unknown registry status for: {bad}")
    return obj


def registry_partitions(path: Path | None = None) -> tuple[frozenset[str], frozenset[str]]:
    families = load_capability_registry(path)["families"]
    supported = frozenset(k for k, v in families.items() if v["status"] in {"AVAILABLE", "PARTIAL"})
    unavailable = frozenset(k for k, v in families.items() if v["status"] == "UNAVAILABLE")
    if supported & unavailable or not supported or not unavailable:
        raise EvidenceRegistryError("registry does not form a valid supported/unavailable partition")
    return supported, unavailable


def _string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise EvidenceRegistryError(f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise EvidenceRegistryError(f"{field} contains duplicate values")
    return set(value)


def derive_case_evidence_sets(pkt_evidence: dict[str, Any]) -> dict[str, list[str]]:
    """Derive the three evidence sets while accepting the deployed packet convention.

    Deployed prepared packets list all system-unavailable families in
    ``missing_evidence_ids`` but do not necessarily repeat case-specific gaps
    there.  Case-specific gaps are therefore derived authoritatively as the
    registry-supported families absent from ``available_evidence_ids``.
    Future packets may additionally list those gaps in ``missing_evidence_ids``.
    """
    if not isinstance(pkt_evidence, dict):
        raise EvidenceRegistryError("evidence must be an object")
    raw_available = _string_set(pkt_evidence.get("available_evidence_ids"), "available_evidence_ids")
    raw_missing = _string_set(pkt_evidence.get("missing_evidence_ids"), "missing_evidence_ids")
    supported, unavailable = registry_partitions()
    all_families = supported | unavailable

    unknown = (raw_available | raw_missing) - all_families
    if unknown:
        raise EvidenceRegistryError(f"packet contains evidence families absent from registry: {sorted(unknown)}")
    overlap = raw_available & raw_missing
    if overlap:
        raise EvidenceRegistryError(f"evidence family listed as both available and missing: {sorted(overlap)}")
    wrongly_available = raw_available & unavailable
    if wrongly_available:
        raise EvidenceRegistryError(f"system-unavailable family listed as available: {sorted(wrongly_available)}")
    omitted_limitations = unavailable - raw_missing
    if omitted_limitations:
        raise EvidenceRegistryError(f"packet omitted system-unavailable families: {sorted(omitted_limitations)}")

    case_available = raw_available & supported
    case_missing = supported - case_available
    explicit_case_missing = raw_missing & supported
    impossible_explicit = explicit_case_missing - case_missing
    if impossible_explicit:
        raise EvidenceRegistryError(f"available supported families also marked missing: {sorted(impossible_explicit)}")

    return {
        "system_unavailable_evidence_ids": sorted(unavailable),
        "case_available_evidence_ids": sorted(case_available),
        "case_missing_but_system_supported_evidence_ids": sorted(case_missing),
    }
