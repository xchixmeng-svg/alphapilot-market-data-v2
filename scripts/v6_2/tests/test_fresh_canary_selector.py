import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_capability import SYSTEM_AVAILABLE_FAMILIES, SYSTEM_UNAVAILABLE_FAMILIES
from fresh_canary_selector import (
    ALL_BUCKETS,
    BUCKET_CASE_SPECIFIC_EVIDENCE_MISSING,
    BUCKET_HIGH_PRIOR_CONFLICTING_EVIDENCE,
    BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT,
    BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE,
    BUCKET_LOW_PRIOR_WEAK_OR_CONFLICTING_EVIDENCE,
    BUCKET_NO_VALUATION_BENCHMARK,
    BUCKET_ROUTINE_MOPS_FILING,
    BUCKET_CATALYTIC_MOPS_FILING,
    DEFAULT_EXCLUDE_CASE_IDS,
    select_fresh_canary,
    tag_case,
    _institutional_flow_direction_signal,
    _classify_mops,
)


def base_pkt(case_id, prior=0.75, available=None):
    available = available if available is not None else list(SYSTEM_AVAILABLE_FAMILIES)
    missing = (set(SYSTEM_AVAILABLE_FAMILIES) - set(available)) | set(SYSTEM_UNAVAILABLE_FAMILIES)
    return {
        "case_id": case_id,
        "decision_date": 20240610,
        "code": f"T{case_id:04d}",
        "validation_stratum": 6 if prior >= 0.67 else (1 if prior <= 0.40 else 4),
        "evidence": {
            "available_evidence_ids": sorted(available),
            "missing_evidence_ids": sorted(missing),
            "price_volume_structure": {"ret20_pct": 5.0},
            "revenue": {"yoy_pct": 3.0},
            "valuation": {"pe": 18.0, "pb": 1.5},
            "institutional_flow": {"foreign_net_ratio_prev_session": 0.1, "foreign_mean5": 0.05},
            "mops_material_information": {"titles": ["routine board meeting notice"], "event_age_hours": 10},
        },
        "numerical_reference_read_only": {"p_hit10_h120": prior},
    }


def test_sampling_uses_prepared_validation_stratum_not_probability_thresholds():
    import fresh_canary_selector
    src = Path(fresh_canary_selector.__file__).read_text(encoding="utf-8")
    assert "PRIOR_HIGH_THRESHOLD" not in src
    assert "PRIOR_LOW_THRESHOLD" not in src
    assert 'pkt.get("validation_stratum")' in src


def test_high_prior_broad_coverage_tagging():
    pkt = base_pkt(1, prior=0.75)
    tag = tag_case(pkt)
    assert BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT in tag.buckets


def test_high_prior_conflicting_evidence_tagging():
    pkt = base_pkt(2, prior=0.75)
    pkt["evidence"]["price_volume_structure"]["ret20_pct"] = 20.0
    pkt["evidence"]["revenue"]["yoy_pct"] = -15.0
    tag = tag_case(pkt)
    assert BUCKET_HIGH_PRIOR_CONFLICTING_EVIDENCE in tag.buckets


def test_low_prior_weak_or_conflicting_evidence_tagging():
    pkt = base_pkt(3, prior=0.35, available=["price_volume_structure", "valuation"])
    tag = tag_case(pkt)
    assert BUCKET_LOW_PRIOR_WEAK_OR_CONFLICTING_EVIDENCE in tag.buckets


def test_case_specific_missing_evidence_tagging():
    pkt = base_pkt(4, available=list(SYSTEM_AVAILABLE_FAMILIES - {"institutional_flow"}))
    tag = tag_case(pkt)
    assert BUCKET_CASE_SPECIFIC_EVIDENCE_MISSING in tag.buckets


def test_routine_mops_tagging():
    pkt = base_pkt(5)
    pkt["evidence"]["mops_material_information"]["titles"] = ["board approved quarterly report"]
    tag = tag_case(pkt)
    assert BUCKET_ROUTINE_MOPS_FILING in tag.buckets


def test_catalytic_mops_tagging():
    pkt = base_pkt(6)
    pkt["evidence"]["mops_material_information"]["titles"] = ["announces major contract with new customer"]
    tag = tag_case(pkt)
    assert BUCKET_CATALYTIC_MOPS_FILING in tag.buckets


def test_no_valuation_benchmark_tagging():
    pkt = base_pkt(7)
    tag = tag_case(pkt)
    assert BUCKET_NO_VALUATION_BENCHMARK in tag.buckets


def test_valuation_benchmark_present_not_tagged():
    pkt = base_pkt(8)
    pkt["evidence"]["valuation"]["peer_pe_median"] = 20.0
    tag = tag_case(pkt)
    assert BUCKET_NO_VALUATION_BENCHMARK not in tag.buckets


def test_institutional_flow_negative_tagging():
    pkt = base_pkt(9)
    pkt["evidence"]["institutional_flow"]["foreign_net_ratio_prev_session"] = -0.5
    pkt["evidence"]["institutional_flow"]["foreign_mean5"] = -0.3
    tag = tag_case(pkt)
    assert BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE in tag.buckets


# ---------------------------------------------------------------------------
# NEW: fix #2 (tautology) -- direct unit tests on the honest 4-way signal
# ---------------------------------------------------------------------------


def test_institutional_flow_signal_negative_only_distinct_from_mixed():
    """The old code could never distinguish these two cases (the tautology
    `has_negative and (has_positive or True)` == `has_negative` regardless
    of has_positive). Now they must produce genuinely different labels."""
    pkt_neg_only = base_pkt(10)
    pkt_neg_only["evidence"]["institutional_flow"] = {
        "foreign_net_ratio_prev_session": -0.5, "foreign_mean5": -0.3, "foreign_mean20": -0.1,
    }
    assert _institutional_flow_direction_signal(pkt_neg_only) == "NEGATIVE_ONLY"

    pkt_mixed = base_pkt(11)
    pkt_mixed["evidence"]["institutional_flow"] = {
        "foreign_net_ratio_prev_session": -0.5, "foreign_mean5": 0.3, "foreign_mean20": -0.1,
    }
    assert _institutional_flow_direction_signal(pkt_mixed) == "MIXED"

    pkt_pos_only = base_pkt(12)
    pkt_pos_only["evidence"]["institutional_flow"] = {
        "foreign_net_ratio_prev_session": 0.5, "foreign_mean5": 0.3,
    }
    assert _institutional_flow_direction_signal(pkt_pos_only) == "POSITIVE_ONLY"

    pkt_none = base_pkt(13)
    pkt_none["evidence"]["institutional_flow"] = {}
    assert _institutional_flow_direction_signal(pkt_none) == "NEUTRAL_OR_UNKNOWN"

    # Both NEGATIVE_ONLY and MIXED still qualify for the same bucket (the
    # bucket itself is intentionally "mixed or negative"), but they are no
    # longer computed via a tautological expression -- verify both tag
    # correctly, distinctly, via the real signal.
    assert BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE in tag_case(pkt_neg_only).buckets
    assert BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE in tag_case(pkt_mixed).buckets
    assert BUCKET_INSTITUTIONAL_FLOW_MIXED_OR_NEGATIVE not in tag_case(pkt_pos_only).buckets


# ---------------------------------------------------------------------------
# NEW: fix #3 -- related-party loan is no longer blanket-classified ROUTINE
# ---------------------------------------------------------------------------


def test_related_party_loan_is_not_auto_classified_as_routine():
    pkt = base_pkt(14)
    pkt["evidence"]["mops_material_information"]["titles"] = ["subsidiary ordinary-course loan to affiliate"]
    assert _classify_mops(pkt) is None  # neither ROUTINE nor CATALYTIC -- left unclassified
    tag = tag_case(pkt)
    assert BUCKET_ROUTINE_MOPS_FILING not in tag.buckets
    assert BUCKET_CATALYTIC_MOPS_FILING not in tag.buckets


def test_zh_related_party_loan_is_not_auto_classified_as_routine():
    pkt = base_pkt(15)
    pkt["evidence"]["mops_material_information"]["titles"] = ["資金貸與關係企業"]
    assert _classify_mops(pkt) is None


# ---------------------------------------------------------------------------
# NEW: fix #5 -- default exclusion of prior canary case_ids
# ---------------------------------------------------------------------------


def test_default_exclude_case_ids_matches_prior_canary_rounds():
    assert DEFAULT_EXCLUDE_CASE_IDS == frozenset({5, 7, 14, 15, 16, 24, 32, 38, 39, 48, 55, 57})


def test_select_fresh_canary_defaults_to_excluding_prior_case_ids():
    pool = [base_pkt(cid) for cid in [5, 7, 999, 1000]]
    result = select_fresh_canary(pool, per_bucket_target=2)
    assert 5 not in result["selected_case_ids"]
    assert 7 not in result["selected_case_ids"]
    assert set(result["excluded_case_ids"]) == set(DEFAULT_EXCLUDE_CASE_IDS)


def test_select_fresh_canary_explicit_exclusion_overrides_default():
    pool = [base_pkt(cid) for cid in [5, 999]]
    result = select_fresh_canary(pool, per_bucket_target=2, exclude_case_ids=set())
    # explicit empty set overrides the default -- case 5 is now eligible
    assert 5 in result["tags"] or 5 not in result["excluded_case_ids"]
    assert result["excluded_case_ids"] == []


# ---------------------------------------------------------------------------
# Selection / determinism / outcome-blindness
# ---------------------------------------------------------------------------


def test_select_fresh_canary_covers_multiple_buckets_deterministically():
    pool = []
    pool.append(base_pkt(101, prior=0.75))
    p2 = base_pkt(102, prior=0.75)
    p2["evidence"]["price_volume_structure"]["ret20_pct"] = 20.0
    p2["evidence"]["revenue"]["yoy_pct"] = -15.0
    pool.append(p2)
    pool.append(base_pkt(103, prior=0.35, available=["price_volume_structure", "valuation"]))
    pool.append(base_pkt(104, available=list(SYSTEM_AVAILABLE_FAMILIES - {"institutional_flow"})))
    p5 = base_pkt(105)
    p5["evidence"]["mops_material_information"]["titles"] = ["announces major contract"]
    pool.append(p5)

    result = select_fresh_canary(pool, per_bucket_target=1, exclude_case_ids=set())
    assert result["selected_case_ids"] == sorted(set(result["selected_case_ids"]))
    assert len(result["bucket_coverage"][BUCKET_HIGH_PRIOR_BROAD_COVERAGE_NO_DETECTED_CONFLICT]) >= 1
    assert 101 in result["selected_case_ids"] or 102 in result["selected_case_ids"]

    result2 = select_fresh_canary(pool, per_bucket_target=1, exclude_case_ids=set())
    assert (
        result["selected_case_ids"] == result2["selected_case_ids"]
        and result["bucket_coverage"] == result2["bucket_coverage"]
    )


def test_select_fresh_canary_respects_exclusion_set():
    pool = [base_pkt(201, prior=0.75), base_pkt(202, prior=0.75)]
    result = select_fresh_canary(pool, per_bucket_target=2, exclude_case_ids={201})
    assert 201 not in result["selected_case_ids"]
    assert 201 not in result["tags"]


def test_all_buckets_enumerated_are_reachable_in_principle():
    assert len(ALL_BUCKETS) == 10
    assert len(set(ALL_BUCKETS)) == 10


def test_module_never_reads_any_outcome_files():
    """Regression guard: this module must never be modified to read hidden
    outcomes for case selection. It has no legitimate reason to read any
    file at all."""
    import fresh_canary_selector
    src = Path(fresh_canary_selector.__file__).read_text(encoding="utf-8")
    forbidden_calls = ["read_csv", "read_parquet", "read_json", "open("]
    for term in forbidden_calls:
        assert term not in src, f"fresh_canary_selector.py must never call {term!r} -- it is dict-in, dict-out only"
