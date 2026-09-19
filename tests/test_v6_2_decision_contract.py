from __future__ import annotations

import copy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evidence_capability import EvidenceRegistryError, derive_case_evidence_sets, registry_partitions
from v6_2_decision_contract import ContractError, build_user_payload, load_response_schema, normalize_output, validate
from run_v6_2_ai_replay64_shard import call_with_retry


def packet(*, institutional: bool = True) -> dict:
    available = ["price_volume_structure", "valuation", "revenue", "mops_material_information"]
    if institutional:
        available.append("institutional_flow")
    return {
        "case_id": 1,
        "decision_date": 20241230,
        "code": "0050",
        "evidence": {
            "available_evidence_ids": available,
            # This is the deployed prepared-artifact convention: permanent
            # limitations only; case-specific gaps need not be repeated here.
            "missing_evidence_ids": [
                "eps_revisions", "analyst_consensus", "industry_pricing",
                "inventory_supply_demand", "broad_news_semantics",
            ],
            "price_volume_structure": {"current_price": 100.0},
            "valuation": {}, "revenue": {}, "institutional_flow": {},
            "mops_material_information": {},
        },
        "numerical_reference_read_only": {"p_hit10_h120": 0.7},
    }


def candidate_response() -> dict:
    return {
        "decision": "CANDIDATE",
        "evidence_quality": "STRONG",
        "hypothesis_type": "REVENUE_MOMENTUM",
        "hypothesis": "Revenue and price structure support a rerating thesis.",
        "primary_evidence_ids": ["price_volume_structure", "revenue"],
        "secondary_evidence_ids": ["valuation"],
        "counter_evidence_ids": [],
        "system_limitations": [
            "analyst_consensus", "broad_news_semantics", "eps_revisions",
            "industry_pricing", "inventory_supply_demand",
        ],
        "bull_thesis": "Revenue and price-volume structure are aligned.",
        "bear_thesis": "Valuation could limit upside.",
        "invalidation": "Price-volume structure breaks below support.",
        "decision_reason": "Revenue and price-volume structure support an actionable setup.",
        "entry": {"status": "NOW", "ideal_low": 98.0, "ideal_high": 102.0},
        "failure_exit": {"exit_price": 94.0, "reason": "Price structure breaks.", "trigger_type": "PRICE_STRUCTURE_BREAK"},
    }


class EvidenceTests(unittest.TestCase):
    def test_registry_partition(self):
        supported, unavailable = registry_partitions()
        self.assertIn("institutional_flow", supported)
        self.assertIn("eps_revisions", unavailable)
        self.assertFalse(supported & unavailable)

    def test_deployed_legacy_missing_convention(self):
        sets = derive_case_evidence_sets(packet(institutional=False)["evidence"])
        self.assertIn("institutional_flow", sets["case_missing_but_system_supported_evidence_ids"])
        self.assertNotIn("institutional_flow", sets["system_unavailable_evidence_ids"])

    def test_future_explicit_case_missing_is_accepted(self):
        pkt = packet(institutional=False)
        pkt["evidence"]["missing_evidence_ids"].append("institutional_flow")
        sets = derive_case_evidence_sets(pkt["evidence"])
        self.assertIn("institutional_flow", sets["case_missing_but_system_supported_evidence_ids"])

    def test_omitted_system_limitation_rejected(self):
        pkt = packet()
        pkt["evidence"]["missing_evidence_ids"].remove("eps_revisions")
        with self.assertRaises(EvidenceRegistryError):
            derive_case_evidence_sets(pkt["evidence"])

    def test_unknown_family_rejected(self):
        pkt = packet()
        pkt["evidence"]["available_evidence_ids"].append("invented")
        with self.assertRaises(EvidenceRegistryError):
            derive_case_evidence_sets(pkt["evidence"])


class PayloadAndSchemaTests(unittest.TestCase):
    def test_payload_removes_ambiguous_legacy_labels(self):
        payload = build_user_payload(packet(institutional=False))
        evidence = payload["case"]["evidence"]
        self.assertNotIn("available_evidence_ids", evidence)
        self.assertNotIn("missing_evidence_ids", evidence)
        self.assertIn("institutional_flow", evidence["evidence_availability"]["case_missing_but_system_supported_evidence_ids"])

    def test_schema_is_runtime_source_for_required_keys(self):
        schema = load_response_schema()
        self.assertEqual(set(schema["required"]), set(candidate_response()))
        self.assertFalse(schema["additionalProperties"])

    def test_payload_does_not_mutate_packet(self):
        pkt = packet()
        original = copy.deepcopy(pkt)
        build_user_payload(pkt)
        self.assertEqual(pkt, original)


class ContractTests(unittest.TestCase):
    def normalized(self, response=None, pkt=None):
        pkt = pkt or packet()
        out = normalize_output(pkt, response or candidate_response())
        return pkt, out

    def test_candidate_valid(self):
        pkt, out = self.normalized()
        self.assertEqual(validate(pkt, out), [])

    def test_enum_and_id_representation_normalized(self):
        response = candidate_response()
        response["decision"] = "candidate"
        response["entry"]["status"] = "now"
        response["primary_evidence_ids"] = ["Price Volume Structure", "Revenue"]
        pkt, out = self.normalized(response)
        self.assertEqual(out["decision"], "CANDIDATE")
        self.assertEqual(out["primary_evidence_ids"], ["price_volume_structure", "revenue"])
        self.assertEqual(validate(pkt, out), [])

    def test_extra_top_level_key_rejected(self):
        response = candidate_response()
        response["probability"] = 0.9
        with self.assertRaises(ContractError):
            normalize_output(packet(), response)

    def test_system_limitations_must_echo_exactly(self):
        response = candidate_response()
        response["system_limitations"].remove("eps_revisions")
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "system_limitations"):
            validate(pkt, out)

    def test_case_missing_family_cannot_be_cited(self):
        pkt = packet(institutional=False)
        response = candidate_response()
        response["secondary_evidence_ids"] = ["institutional_flow"]
        _, out = self.normalized(response, pkt)
        with self.assertRaisesRegex(ContractError, "unavailable/unknown"):
            validate(pkt, out)

    def test_system_family_cannot_be_counter_evidence(self):
        response = candidate_response()
        response["counter_evidence_ids"] = ["eps_revisions"]
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "unavailable/unknown"):
            validate(pkt, out)

    def test_reject_based_only_on_system_limits_is_hard_error(self):
        response = candidate_response()
        response.update({
            "decision": "REJECT",
            "counter_evidence_ids": [],
            "decision_reason": "EPS revisions and analyst consensus are unavailable.",
            "bear_thesis": "No analyst consensus is available.",
            "entry": {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None},
            "failure_exit": {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"},
        })
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "system limitations"):
            validate(pkt, out)

    def test_valid_reject_uses_available_counter_evidence(self):
        response = candidate_response()
        response.update({
            "decision": "REJECT",
            "counter_evidence_ids": ["price_volume_structure"],
            "decision_reason": "Price-volume structure actively contradicts the thesis.",
            "entry": {"status": "NOT_ACTIONABLE", "ideal_low": None, "ideal_high": None},
            "failure_exit": {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"},
        })
        pkt, out = self.normalized(response)
        self.assertEqual(validate(pkt, out), [])

    def test_non_actionable_entry_bounds_must_be_null(self):
        response = candidate_response()
        response.update({
            "decision": "WATCH",
            "entry": {"status": "NOT_ACTIONABLE", "ideal_low": 90.0, "ideal_high": None},
            "failure_exit": {"exit_price": None, "reason": None, "trigger_type": "UNAVAILABLE"},
        })
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "null entry bounds"):
            validate(pkt, out)

    def test_entry_distance_guard_preserved(self):
        response = candidate_response()
        response["entry"] = {"status": "NOW", "ideal_low": 140.0, "ideal_high": 145.0}
        response["failure_exit"]["exit_price"] = 130.0
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "implausibly far"):
            validate(pkt, out)

    def test_failure_exit_below_entry_high_guard_preserved(self):
        response = candidate_response()
        response["failure_exit"]["exit_price"] = 103.0
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "failure-exit price"):
            validate(pkt, out)

    def test_empty_primary_evidence_rejected(self):
        response = candidate_response()
        response["primary_evidence_ids"] = []
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "empty primary"):
            validate(pkt, out)

    def test_duplicate_evidence_rejected(self):
        response = candidate_response()
        response["primary_evidence_ids"] = ["revenue", "revenue"]
        pkt, out = self.normalized(response)
        with self.assertRaisesRegex(ContractError, "duplicate"):
            validate(pkt, out)

    def test_retry_receives_first_contract_error(self):
        calls = []

        def fake_call(_pkt, repair=None):
            calls.append(repair)
            if len(calls) == 1:
                raise ContractError("first validation failure")
            return candidate_response(), 0.1, []

        with patch("run_v6_2_ai_replay64_shard.call", side_effect=fake_call):
            result, elapsed, warnings = call_with_retry(packet())
        self.assertEqual(result["decision"], "CANDIDATE")
        self.assertEqual(elapsed, 0.1)
        self.assertEqual(warnings, [])
        self.assertIsNone(calls[0])
        self.assertIn("first validation failure", calls[1])


if __name__ == "__main__":
    unittest.main()
