import copy
import unittest
from datetime import datetime

from scripts.v6_2_v4_2_contract import (
    CONTRACT_VERSION,
    ContractError,
    LAUNCH_WINDOWS,
    OUTCOME_HORIZONS,
    validate_pair,
)


def packet():
    return {
        "case_id": 4,
        "decision_date": 20240102,
        "code": "2465",
        "evidence": {"available_evidence_ids": ["revenue"], "revenue": {"yoy_pct": 101.25}},
        "numerical_reference_read_only": {"p_hit10_h20": 0.2},
    }


def sidecar():
    return {
        "contract_version": CONTRACT_VERSION,
        "case_id": 4,
        "decision_date": 20240102,
        "code": "2465",
        "causal_context_read_only": {
            "market_context": {"values": {"market_breadth": 0.55}},
            "industry_index_context": {"values": {}, "source_columns": [f"industry_ret1_{i:02d}" for i in range(37)], "relationship_to_company": "CONTEXT_ONLY_NOT_COMPANY_CLASSIFICATION", "company_industry_mapping": None},
            "macro_context": {"values": {}},
            "event_timeliness_context": {"values": {"event_last_age_hours": 5}},
            "revenue_context": {"values": {"rev_yoy_pct": 101.25}},
        },
        "launch_observation_windows": LAUNCH_WINDOWS,
        "profit_outcome_horizons": OUTCOME_HORIZONS,
        "h120_is_holding_period": False,
        "outcomes_opened": False,
        "selection_recomputed": False,
        "frozen_layers_modified": False,
    }


class ContractTests(unittest.TestCase):
    def test_valid_exact_pair(self):
        validate_pair(packet(), sidecar())

    def assert_rejected(self, mutate):
        p, s = packet(), sidecar()
        mutate(p, s)
        with self.assertRaises(ContractError):
            validate_pair(p, s)

    def test_exact_key_mismatch(self):
        self.assert_rejected(lambda p, s: s.update(code="9999"))

    def test_post_2024_date(self):
        self.assert_rejected(lambda p, s: s.update(decision_date=20250102))

    def test_opened_outcomes(self):
        self.assert_rejected(lambda p, s: s.update(outcomes_opened=True))

    def test_selection_recomputed(self):
        self.assert_rejected(lambda p, s: s.update(selection_recomputed=True))

    def test_frozen_layer_modified(self):
        self.assert_rejected(lambda p, s: s.update(frozen_layers_modified=True))

    def test_raw_pe_leak(self):
        self.assert_rejected(lambda p, s: p["evidence"].update(valuation={"pe": 10.2}))

    def test_future_label_leak(self):
        self.assert_rejected(lambda p, s: s["causal_context_read_only"]["revenue_context"].update(y_hit10_h120=1))

    def test_industry_impersonation(self):
        self.assert_rejected(lambda p, s: s["causal_context_read_only"]["industry_index_context"].update(company_industry="Semiconductor"))

    def test_missing_industry_indices(self):
        self.assert_rejected(lambda p, s: s["causal_context_read_only"]["industry_index_context"].update(source_columns=[]))

    def test_horizon_drift(self):
        self.assert_rejected(lambda p, s: s.update(profit_outcome_horizons=[120]))

    def test_timestamp_date_normalizes(self):
        s = sidecar()
        s["decision_date"] = datetime(2024, 1, 2)
        validate_pair(packet(), s)


if __name__ == "__main__":
    unittest.main()
