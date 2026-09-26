import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_v6_2_v4_2_stage_c import (  # noqa: E402
    action_schema,
    decision_schema,
    render_decision_narratives,
)
from v6_2_v4_2_contract import validate_decision_semantics  # noqa: E402


def packet():
    return {
        "case_id": 4,
        "decision_date": 20240102,
        "code": "2465",
        "evidence": {
            "available_evidence_ids": ["revenue", "price_volume_structure", "institutional_flow"],
            "price_volume_structure": {"current_price": 100.0},
        },
        "numerical_reference_read_only": {"p_hit10_h20": 0.2},
    }


class StageCCompactContractTests(unittest.TestCase):
    def test_decision_schema_is_compact_and_has_no_free_text_narratives(self):
        props = decision_schema(packet())["properties"]
        self.assertIn("hypothesis_code", props)
        self.assertIn("decision_reason_code", props)
        for forbidden in (
            "hypothesis",
            "bull_thesis",
            "bear_thesis",
            "invalidation",
            "context_assessment",
            "decision_reason",
        ):
            self.assertNotIn(forbidden, props)

    def test_action_schema_has_no_model_generated_reason_prose(self):
        failure_props = action_schema(packet())["properties"]["failure_exit"]["properties"]
        self.assertNotIn("reason", failure_props)
        self.assertIn("trigger_type", failure_props)

    def test_rendered_narratives_are_grounded_codes_and_ids_only(self):
        compact = {
            "decision": "WATCH",
            "hypothesis_code": "REVENUE_ACCELERATION",
            "primary_evidence_ids": ["revenue"],
            "secondary_evidence_ids": ["price_volume_structure"],
            "bull_evidence_ids": ["revenue"],
            "bear_evidence_ids": ["price_volume_structure"],
            "invalidation_code": "REVENUE_MOMENTUM_BREAK",
            "invalidation_evidence_ids": ["revenue"],
            "timing_code": "DEVELOPING",
            "market_context_stance": "NEUTRAL",
            "macro_context_stance": "NEUTRAL",
            "industry_index_context_stance": "SUPPORTIVE",
            "event_timeliness_context_stance": "NEUTRAL",
            "revenue_context_stance": "SUPPORTIVE",
            "decision_reason_code": "CONSTRUCTIVE_BUT_NOT_ACTIONABLE",
        }
        narratives = render_decision_narratives(compact)
        validate_decision_semantics(narratives)
        self.assertIn("REVENUE_ACCELERATION", narratives["hypothesis"])
        self.assertNotIn("analyst", " ".join(narratives.values()).lower())


if __name__ == "__main__":
    unittest.main()
