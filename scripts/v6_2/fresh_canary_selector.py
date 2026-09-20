"""
fresh_canary_selector.py

Deterministic, outcome-blind selection of a canary case set from a pool of
prepared case packets. This module MUST NEVER import, open, or reference
HIDDEN_OUTCOMES.csv, HIDDEN_FUTURE_PATHS.parquet, or any hidden-outcome
concept -- selection uses ONLY fields already present in the prepared
packet (evidence + the calibrated numerical prior).

FIXED AFTER EXTERNAL REVIEW (this version):

  1. Bucket naming no longer implies a quality judgment the selector does
     not actually make. The old BUCKET_HIGH_PRIOR_STRONG_EVIDENCE name
     claimed "strong evidence" while only counting how many families were
     present -- exactly the family-count-as-quality conflation the whole
     architecture exists to avoid. Renamed to
     BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT /
     BUCKET_LOW_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT, and the
     docstrings say explicitly: this is a coverage/conflict-detection
     proxy for STRATIFIED SAMPLING PURPOSES ONLY. Real evidence_quality
     is judged exclusively inside Stage 1/2, never here.

  2. Fixed the tautology in the old institutional-flow check
     (`has_negative and (has_positive or True)` is identical to
     `has_negative`, and could never distinguish "negative-only" from
     "genuinely mixed"). Replaced with
     `_institutional_flow_direction_signal()`, which returns one of
     NEGATIVE_ONLY / MIXED / POSITIVE_ONLY / NEUTRAL_OR_UNKNOWN, an
     honest three-way (four-way) classification instead of a broken
     boolean.

  3. "資金貸與" (loan to affiliate) was removed from the blanket
     ROUTINE_MOPS_KEYWORDS auto-classification list. A related-party loan
     can be ordinary-course, or it can be a genuine liquidity/governance/
     fund-transfer risk signal -- title keywords alone cannot tell these
     apart (this exact overreach caused one of the reported real
     misclassifications). This selector now leaves that MOPS title
     UNCLASSIFIED (neither ROUTINE_MOPS_FILING nor CATALYTIC_MOPS_FILING)
     rather than guessing; Stage 1's own prompt has been updated
     separately to require amount/counterparty/rationale-based judgment,
     defaulting to NOT_INTERPRETABLE with a stated limitation when that
     detail is absent, rather than a title-keyword default.

  4. The PRIOR_HIGH_THRESHOLD/PRIOR_LOW_THRESHOLD constants are now
     documented as declared, round-number, a-priori sampling strata
     (0.60 / 0.40, chosen to divide [0,1] into thirds before ever seeing
     this or any canary pool's actual prior distribution), not values fit
     to the 64-case or any other specific sample. They remain strictly a
     SAMPLING concept, never a decision admission rule -- nothing in this
     module, or anywhere else in this delivery, ever says
     "if prior > X then CANDIDATE".

  5. DEFAULT_EXCLUDE_CASE_IDS lists the 31 case_ids already used across
     five prior canary rounds (35456479322 / 35458136917 / 35459949583 /
     35482706294 / 35485028854), so a fresh canary selection defaults to
     genuinely fresh cases without requiring the caller to remember and
     pass this list manually. Callers can still override via the
     exclude_case_ids parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence_capability import SYSTEM_AVAILABLE_FAMILIES, derive_case_evidence_sets

# ---------------------------------------------------------------------------
# Bucket definitions (the 10 dimensions requested)
# ---------------------------------------------------------------------------

BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT = "HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT"
BUCKET_HIGH_PRIOR_CONFLICTING_EVIDENCE = "HIGH_PRIOR_CONFLICTING_EVIDENCE"
BUCKET_MID_PRIOR_MIXED_EVIDENCE = "MID_PRIOR_MIXED_EVIDENCE"
BUCKET_LOW_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT = "LOW_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT"
BUCKET_LOW_PRIOR_WEAK_OR_CONFLICTING_EVIDENCE = "LOW_PRIOR_WEAK_OR_CONFLICTING_EVIDENCE"
BUCKET_CASE_SPECIFIC_EVIDENCE_MISSING = "CASE_SPECIFIC_EVIDENCE_MISSING"
BUCKET_ROUTINE_MOPS_FILING = "ROUTINE_MOPS_FILING"
BUCKET_CATALYTIC_MOPS_FILING = "CATALYTIC_MOPS_FILING"
BUCKET_NO_VALUATION_BENCHMARK = "NO_VALUATION_BENCHMARK"
BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE = "INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE"

ALL_BUCKETS = [
    BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT,
    BUCKET_HIGH_PRIOR_CONFLICTING_EVIDENCE,
    BUCKET_MID_PRIOR_MIXED_EVIDENCE,
    BUCKET_LOW_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT,
    BUCKET_LOW_PRIOR_WEAK_OR_CONFLICTING_EVIDENCE,
    BUCKET_CASE_SPECIFIC_EVIDENCE_MISSING,
    BUCKET_ROUTINE_MOPS_FILING,
    BUCKET_CATALYTIC_MOPS_FILING,
    BUCKET_NO_VALUATION_BENCHMARK,
    BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE,
]

# Declared a priori sampling strata ONLY -- see fix #4 in the module
# docstring. Never used as a decision admission rule anywhere.
PRIOR_HIGH_THRESHOLD = 0.60
PRIOR_LOW_THRESHOLD = 0.40

# Case IDs already used across canary rounds 35456479322 / 35458136917 /
# 35459949583 / 35482706294 / 35485028854, per the external review.
DEFAULT_EXCLUDE_CASE_IDS: frozenset[int] = frozenset({
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    21, 22, 24, 32, 38, 39, 40, 47, 48, 50, 53, 55, 57, 61,
})

ROUTINE_MOPS_KEYWORDS = [
    "board approved", "consolidated financial report", "quarterly report", "annual report",
    "personnel", "rotation", "shareholder meeting", "routine",
    "董事會通過", "合併財務報告", "股東會", "股東臨時會", "人事", "輪調",
    # NOTE: "資金貸與" / "loan to affiliate" / "ordinary-course loan" are
    # intentionally NOT in this list -- see fix #3 above. Related-party
    # loans are left unclassified by keyword; they require amount/
    # counterparty/rationale context this selector does not have.
]
CATALYTIC_MOPS_KEYWORDS = [
    "acquisition", "merger", "major contract", "new product", "patent", "capacity expansion",
    "price increase", "guidance revision", "joint venture", "strategic partnership",
    "收購", "併購", "重大合約", "新產品", "專利", "產能擴充", "調漲", "上修財測", "策略合作",
]


@dataclass
class CanaryCaseTag:
    case_id: int
    buckets: list[str] = field(default_factory=list)
    numerical_prior_p_hit10_h120: float | None = None


def _get_prior(pkt: dict) -> float | None:
    numref = pkt.get("numerical_reference_read_only") or {}
    v = numref.get("p_hit10_h120")
    return v if isinstance(v, (int, float)) else None


def _packet_has_valuation_benchmark_hint(pkt: dict) -> bool:
    """ADAPT-THIS: field names here are best-guesses; update to your real
    prepared-packet schema's benchmark field name(s) if different."""
    valuation = pkt.get("evidence", {}).get("valuation")
    if not isinstance(valuation, dict):
        return False
    benchmark_like_keys = {"peer_pe", "peer_pe_median", "historical_pe_band", "historical_pe_low",
                            "historical_pe_high", "sector_median_pe", "peer_pb", "historical_pb_band"}
    return bool(benchmark_like_keys & set(valuation.keys()))


def _classify_mops(pkt: dict) -> str | None:
    mops = pkt.get("evidence", {}).get("mops_material_information")
    if not isinstance(mops, dict):
        return None
    titles = mops.get("titles") or []
    joined = " ".join(str(t) for t in titles).lower()
    if any(kw.lower() in joined for kw in CATALYTIC_MOPS_KEYWORDS):
        return "CATALYTIC"
    if any(kw.lower() in joined for kw in ROUTINE_MOPS_KEYWORDS):
        return "ROUTINE"
    return None  # includes related-party loans and anything else not confidently classifiable


def _institutional_flow_direction_signal(pkt: dict) -> str:
    """
    Returns exactly one of:
      NEGATIVE_ONLY       -- every available signal is <= 0, at least one < 0
      MIXED                -- at least one signal < 0 AND at least one > 0
      POSITIVE_ONLY        -- every available signal is >= 0, at least one > 0
      NEUTRAL_OR_UNKNOWN    -- no numeric signals available, or all exactly 0
    (Fixes the prior tautology: `has_negative and (has_positive or True)`
    was identical to `has_negative` and could never actually report MIXED
    vs NEGATIVE_ONLY as distinct outcomes.)
    """
    flow = pkt.get("evidence", {}).get("institutional_flow")
    if not isinstance(flow, dict) or not flow:
        return "NEUTRAL_OR_UNKNOWN"
    values = [
        flow.get("foreign_net_ratio_prev_session"),
        flow.get("foreign_mean5"),
        flow.get("foreign_mean20"),
        flow.get("trust_net_ratio_prev_session"),
    ]
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return "NEUTRAL_OR_UNKNOWN"
    has_negative = any(v < 0 for v in numeric)
    has_positive = any(v > 0 for v in numeric)
    if has_negative and has_positive:
        return "MIXED"
    if has_negative:
        return "NEGATIVE_ONLY"
    if has_positive:
        return "POSITIVE_ONLY"
    return "NEUTRAL_OR_UNKNOWN"


def _price_vs_revenue_conflict(pkt: dict) -> bool:
    pvs = pkt.get("evidence", {}).get("price_volume_structure") or {}
    revenue = pkt.get("evidence", {}).get("revenue") or {}
    ret20 = pvs.get("ret20_pct")
    yoy = revenue.get("yoy_pct")
    if not isinstance(ret20, (int, float)) or not isinstance(yoy, (int, float)):
        return False
    return ret20 >= 15.0 and yoy <= -10.0


def tag_case(pkt: dict) -> CanaryCaseTag:
    tag = CanaryCaseTag(case_id=pkt["case_id"])
    prior = _get_prior(pkt)
    tag.numerical_prior_p_hit10_h120 = prior

    evidence_sets = derive_case_evidence_sets(pkt["evidence"])
    case_missing = set(evidence_sets["case_missing_but_system_supported_evidence_ids"])
    n_available = len(evidence_sets["case_available_evidence_ids"])
    conflict = _price_vs_revenue_conflict(pkt)
    flow_signal = _institutional_flow_direction_signal(pkt)
    inst_mixed_or_negative = flow_signal in ("NEGATIVE_ONLY", "MIXED")
    has_benchmark = _packet_has_valuation_benchmark_hint(pkt)
    mops_type = _classify_mops(pkt)

    if prior is not None:
        if prior >= PRIOR_HIGH_THRESHOLD:
            if conflict or inst_mixed_or_negative:
                tag.buckets.append(BUCKET_HIGH_PRIOR_CONFLICTING_EVIDENCE)
            elif n_available >= 4:
                tag.buckets.append(BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT)
        elif prior <= PRIOR_LOW_THRESHOLD:
            if conflict or inst_mixed_or_negative or n_available <= 2:
                tag.buckets.append(BUCKET_LOW_PRIOR_WEAK_OR_CONFLICTING_EVIDENCE)
            else:
                tag.buckets.append(BUCKET_LOW_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT)
        else:
            tag.buckets.append(BUCKET_MID_PRIOR_MIXED_EVIDENCE)

    if case_missing:
        tag.buckets.append(BUCKET_CASE_SPECIFIC_EVIDENCE_MISSING)

    if mops_type == "ROUTINE":
        tag.buckets.append(BUCKET_ROUTINE_MOPS_FILING)
    elif mops_type == "CATALYTIC":
        tag.buckets.append(BUCKET_CATALYTIC_MOPS_FILING)

    if "valuation" in evidence_sets["case_available_evidence_ids"] and not has_benchmark:
        tag.buckets.append(BUCKET_NO_VALUATION_BENCHMARK)

    if inst_mixed_or_negative:
        tag.buckets.append(BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE)

    return tag


def select_fresh_canary(
    packet_pool: list[dict],
    per_bucket_target: int = 2,
    exclude_case_ids: set[int] | None = None,
) -> dict:
    """
    Deterministically selects up to `per_bucket_target` packets per bucket,
    sorted by case_id for reproducibility. Defaults to excluding the 31
    case_ids already used in prior canary rounds (DEFAULT_EXCLUDE_CASE_IDS);
    pass exclude_case_ids explicitly to override (e.g. set() to include
    everything, or a larger set after another round is run).
    """
    if exclude_case_ids is None:
        exclude_case_ids = set(DEFAULT_EXCLUDE_CASE_IDS)

    pool_sorted = sorted(
        (p for p in packet_pool if p["case_id"] not in exclude_case_ids),
        key=lambda p: p["case_id"],
    )

    tags = {p["case_id"]: tag_case(p) for p in pool_sorted}

    bucket_coverage: dict[str, list[int]] = {b: [] for b in ALL_BUCKETS}
    for pkt in pool_sorted:
        tag = tags[pkt["case_id"]]
        for b in tag.buckets:
            if len(bucket_coverage[b]) < per_bucket_target:
                bucket_coverage[b].append(pkt["case_id"])

    selected = sorted({cid for ids in bucket_coverage.values() for cid in ids})
    uncovered = [b for b, ids in bucket_coverage.items() if not ids]

    return {
        "selected_case_ids": selected,
        "bucket_coverage": bucket_coverage,
        "tags": tags,
        "uncovered_buckets": uncovered,
        "excluded_case_ids": sorted(exclude_case_ids),
    }
