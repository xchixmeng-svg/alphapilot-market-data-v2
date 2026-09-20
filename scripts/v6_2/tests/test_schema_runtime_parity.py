"""
test_schema_runtime_parity.py

Compares the hand-written required-key-sets/enums in
deterministic_validators.py against the corresponding JSON Schema files,
for the (now four) schemas: STAGE1, STAGE2_DECISION, STAGE2_ACTIONABILITY,
STAGE3_CRITIC.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deterministic_validators as dv

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def _load(name):
    with open(SCHEMA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def test_stage1_top_level_keys_match_schema():
    schema = _load("STAGE1_EVIDENCE_DIRECTION_SCHEMA.json")
    assert set(schema["required"]) == set(schema["properties"].keys()) == {"case_id", "evidence_observations"}


def test_stage1_observation_keys_match_schema():
    schema = _load("STAGE1_EVIDENCE_DIRECTION_SCHEMA.json")
    obs_def = schema["definitions"]["evidence_observation"]
    schema_keys = set(obs_def["properties"].keys())
    assert schema_keys == set(obs_def["required"])
    runtime_keys = {
        "evidence_id", "exact_observations", "direction", "timeliness",
        "relevance_to_hypothesis_space", "limitations", "benchmark_available",
    }
    assert schema_keys == runtime_keys


def test_stage1_direction_enum_matches_schema():
    schema = _load("STAGE1_EVIDENCE_DIRECTION_SCHEMA.json")
    schema_enum = set(schema["definitions"]["evidence_observation"]["properties"]["direction"]["enum"])
    assert schema_enum == dv.DIRECTION_VALUES


def test_stage1_timeliness_enum_matches_schema():
    schema = _load("STAGE1_EVIDENCE_DIRECTION_SCHEMA.json")
    schema_enum = set(schema["definitions"]["evidence_observation"]["properties"]["timeliness"]["enum"])
    assert schema_enum == dv.TIMELINESS_VALUES


def test_stage1_relevance_enum_matches_schema():
    schema = _load("STAGE1_EVIDENCE_DIRECTION_SCHEMA.json")
    schema_enum = set(schema["definitions"]["evidence_observation"]["properties"]["relevance_to_hypothesis_space"]["enum"])
    assert schema_enum == dv.RELEVANCE_VALUES


def test_stage2_decision_top_level_keys_match_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    schema_keys = set(schema["properties"].keys())
    assert schema_keys == set(schema["required"])
    runtime_keys = {
        "decision", "evidence_quality", "hypothesis_type", "hypothesis",
        "primary_evidence_ids", "secondary_evidence_ids", "counter_evidence_ids",
        "system_limitations", "bull_thesis", "bear_thesis", "invalidation", "decision_reason",
    }
    assert schema_keys == runtime_keys


def test_stage2_decision_schema_has_no_entry_or_failure_exit():
    """Structural regression guard: the whole point of the split is that
    this schema must never re-acquire entry/failure_exit fields."""
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    assert "entry" not in schema["properties"]
    assert "failure_exit" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_stage2_decision_decision_enum_matches_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    assert set(schema["properties"]["decision"]["enum"]) == dv.DECISION_VALUES


def test_stage2_decision_evidence_quality_enum_matches_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    assert set(schema["properties"]["evidence_quality"]["enum"]) == dv.EVIDENCE_QUALITY_VALUES


def test_stage2_actionability_top_level_keys_match_schema():
    schema = _load("STAGE2_ACTIONABILITY_SCHEMA.json")
    schema_keys = set(schema["properties"].keys())
    assert schema_keys == set(schema["required"]) == {"entry", "failure_exit"}
    assert schema["additionalProperties"] is False


def test_stage2_actionability_entry_status_enum_excludes_not_actionable():
    schema = _load("STAGE2_ACTIONABILITY_SCHEMA.json")
    entry_schema = schema["properties"]["entry"]
    schema_enum = set(entry_schema["properties"]["status"]["enum"])
    assert schema_enum == dv.ENTRY_STATUS_ACTIONABLE_VALUES
    assert "NOT_ACTIONABLE" not in schema_enum


def test_stage2_actionability_trigger_type_enum_excludes_unavailable():
    schema = _load("STAGE2_ACTIONABILITY_SCHEMA.json")
    fe_schema = schema["properties"]["failure_exit"]
    schema_enum = set(fe_schema["properties"]["trigger_type"]["enum"])
    assert schema_enum == dv.FAILURE_TRIGGER_ACTIONABLE_VALUES
    assert "UNAVAILABLE" not in schema_enum


def test_stage2_actionability_prices_are_required_non_null_numbers():
    schema = _load("STAGE2_ACTIONABILITY_SCHEMA.json")
    entry_schema = schema["properties"]["entry"]
    assert entry_schema["properties"]["ideal_low"]["type"] == "number"  # not ["number","null"]
    assert entry_schema["properties"]["ideal_high"]["type"] == "number"
    fe_schema = schema["properties"]["failure_exit"]
    assert fe_schema["properties"]["exit_price"]["type"] == "number"


def test_stage3_top_level_keys_match_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    schema_keys = set(schema["properties"].keys())
    assert schema_keys == set(schema["required"]) == {"verdict", "problems"}


def test_stage3_verdict_enum_matches_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    assert set(schema["properties"]["verdict"]["enum"]) == dv.CRITIC_VERDICT_VALUES


def test_stage3_problem_item_keys_match_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    problem_schema = schema["properties"]["problems"]["items"]
    schema_keys = set(problem_schema["properties"].keys())
    assert schema_keys == set(problem_schema["required"]) == dv.CRITIC_PROBLEM_REQUIRED_KEYS


def test_stage3_problem_field_enum_matches_schema_including_entry_or_failure_exit():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    problem_schema = schema["properties"]["problems"]["items"]
    schema_enum = set(problem_schema["properties"]["field"]["enum"])
    assert schema_enum == dv.CRITIC_PROBLEM_FIELD_VALUES
    assert "entry_or_failure_exit" in schema_enum


def test_stage3_problem_type_enum_matches_schema_including_actionability_price():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    problem_schema = schema["properties"]["problems"]["items"]
    schema_enum = set(problem_schema["properties"]["problem_type"]["enum"])
    assert schema_enum == dv.CRITIC_PROBLEM_TYPES
    assert "ACTIONABILITY_PRICE_NOT_GROUNDED" in schema_enum


def test_stage3_schema_forbids_additional_properties():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    assert schema["additionalProperties"] is False
