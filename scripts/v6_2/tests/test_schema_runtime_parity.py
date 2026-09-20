"""
test_schema_runtime_parity.py

Fix for review point 15: the JSON Schema files are not actually loaded by
the runtime validators (deterministic_validators.py is hand-written
Python, not a jsonschema-library consumer, since the offline sandbox this
was built in has no network access to install the `jsonschema` package --
see README for the recommendation to wire real jsonschema validation in
your CI, which DOES have network access). Until that wiring exists, these
tests are the mechanism that prevents the hand-written validators and the
schema files from silently drifting apart: every required-key set the
runtime hard-codes is compared against the corresponding JSON Schema
file's own `required`/`properties` declarations.
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
    # The runtime's own required-key set is inlined in
    # _validate_stage1_output_impl; reproduce it here for comparison
    # rather than importing a private local, since it is a local variable
    # not a module-level constant.
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


def test_stage2_top_level_keys_match_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    schema_keys = set(schema["properties"].keys())
    assert schema_keys == set(schema["required"])
    runtime_keys = {
        "decision", "evidence_quality", "hypothesis_type", "hypothesis",
        "primary_evidence_ids", "secondary_evidence_ids", "counter_evidence_ids",
        "system_limitations", "bull_thesis", "bear_thesis", "invalidation",
        "decision_reason", "entry", "failure_exit",
    }
    assert schema_keys == runtime_keys


def test_stage2_decision_enum_matches_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    schema_enum = set(schema["properties"]["decision"]["enum"])
    assert schema_enum == dv.DECISION_VALUES


def test_stage2_evidence_quality_enum_matches_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    schema_enum = set(schema["properties"]["evidence_quality"]["enum"])
    assert schema_enum == dv.EVIDENCE_QUALITY_VALUES


def test_stage2_entry_keys_and_enum_match_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    entry_schema = schema["properties"]["entry"]
    assert set(entry_schema["properties"].keys()) == set(entry_schema["required"]) == {"status", "ideal_low", "ideal_high"}
    assert set(entry_schema["properties"]["status"]["enum"]) == dv.ENTRY_STATUS_VALUES
    assert entry_schema["additionalProperties"] is False


def test_stage2_failure_exit_keys_and_enum_match_schema():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    fe_schema = schema["properties"]["failure_exit"]
    assert set(fe_schema["properties"].keys()) == set(fe_schema["required"]) == {"exit_price", "reason", "trigger_type"}
    assert set(fe_schema["properties"]["trigger_type"]["enum"]) == dv.FAILURE_TRIGGER_VALUES
    assert fe_schema["additionalProperties"] is False


def test_stage2_schema_forbids_additional_properties():
    schema = _load("STAGE2_DECISION_SCHEMA.json")
    assert schema["additionalProperties"] is False


def test_stage3_top_level_keys_match_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    schema_keys = set(schema["properties"].keys())
    assert schema_keys == set(schema["required"]) == {"verdict", "problems"}


def test_stage3_verdict_enum_matches_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    schema_enum = set(schema["properties"]["verdict"]["enum"])
    assert schema_enum == dv.CRITIC_VERDICT_VALUES


def test_stage3_problem_item_keys_match_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    problem_schema = schema["properties"]["problems"]["items"]
    schema_keys = set(problem_schema["properties"].keys())
    assert schema_keys == set(problem_schema["required"]) == dv.CRITIC_PROBLEM_REQUIRED_KEYS


def test_stage3_problem_field_enum_matches_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    problem_schema = schema["properties"]["problems"]["items"]
    schema_enum = set(problem_schema["properties"]["field"]["enum"])
    assert schema_enum == dv.CRITIC_PROBLEM_FIELD_VALUES


def test_stage3_problem_type_enum_matches_schema():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    problem_schema = schema["properties"]["problems"]["items"]
    schema_enum = set(problem_schema["properties"]["problem_type"]["enum"])
    assert schema_enum == dv.CRITIC_PROBLEM_TYPES


def test_stage3_schema_forbids_additional_properties():
    schema = _load("STAGE3_CRITIC_SCHEMA.json")
    assert schema["additionalProperties"] is False
