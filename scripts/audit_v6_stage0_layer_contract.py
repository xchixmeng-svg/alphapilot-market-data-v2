#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "research" / "V6_STAGE0_LAYER_CONTRACT.json"
PREREG_PATH = ROOT / "research" / "AI_MARKET_REASONING_FORECAST_V6_CPU_QUANTILE_PREREGISTRATION.md"
REGISTRY_PATH = ROOT / "research" / "V6_CPU_QUANTILE_HYPOTHESIS_REGISTRY.json"

EXPECTED_LAYERS = [
    "0A_MARKET",
    "0B_INDUSTRY",
    "0C_COMPANY",
    "0D_MACRO",
    "0E_EVENT_TIME",
    "0F_FINAL_ASSEMBLY",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_contract_only():
    contract = load_json(CONTRACT_PATH)
    registry = load_json(REGISTRY_PATH)
    prereg = PREREG_PATH.read_text(encoding="utf-8")

    assert contract["version"] == "V6-CPU-QUANTILE-STAGE0-LAYER-CONTRACT-v1"
    assert contract["branch"] == registry["branch"]
    assert [x["layer_id"] for x in contract["layers"]] == EXPECTED_LAYERS
    assert all(x["required"] for x in contract["layers"])

    p = contract["principles"]
    assert p["layers_are_data_engineering_only"] is True
    assert p["single_model_after_final_assembly"] is True
    assert p["separate_layer_models_allowed"] is False
    assert p["manual_layer_weights_allowed"] is False
    assert p["silent_zero_fill_allowed"] is False
    assert p["future_join_allowed"] is False

    fb = contract["fallback_equivalence_gate"]
    assert fb["must_pass_before_layer_freeze"] is True
    assert fb["minimum_overlap_observations"] >= 60
    assert fb["minimum_distinct_calendar_years"] >= 3
    assert fb["absolute_value_tolerance"] <= 1e-6
    assert fb["allowed_value_mismatch_rate"] == 0.0
    assert "publication_or_availability_lag" in fb["required_checks"]

    gate = contract["stage0f_cross_layer_gate"]
    assert gate["all_required_layers_must_be_frozen_pass"] is True
    assert gate["daily_required_layer_eligible_date_coverage"] == 1.0
    assert gate["periodic_join_rule"] == "available_at <= decision_time"
    assert gate["future_join_violations_allowed"] == 0
    assert gate["silent_zero_fill_violations_allowed"] == 0
    assert gate["duplicate_decision_key_violations_allowed"] == 0

    ab = contract["layer_ablation_audit"]
    assert ab["diagnostic_only"] is True
    assert ab["may_remove_or_reweight_layer_under_same_lock_after_oos"] is False
    assert ab["schema_change_requires_successor_preregistration"] is True

    sr = registry["stage0_layer_contract"]
    assert sr["required_layers"] == EXPECTED_LAYERS
    assert sr["single_model_after_assembly"] is True
    assert sr["layer_weighting_or_voting_allowed"] is False
    assert sr["fallback_requires_equivalence_pass_before_freeze"] is True
    assert sr["cross_layer_alignment_required_before_oos"] is True
    assert sr["ablation_may_change_locked_schema_after_oos"] is False

    for needle in (
        "Fallback-source equivalence gate",
        "Stage 0F cross-layer alignment gate",
        "Layer Ablation Audit is also diagnostic only",
    ):
        assert needle in prereg, needle

    print("STAGE0 LAYER CONTRACT STATIC AUDIT PASS")


def validate_manifest(path: Path, layer_id: str, required_fields: list[str]):
    obj = load_json(path)
    missing = [k for k in required_fields if k not in obj]
    assert not missing, f"{layer_id}: missing manifest fields {missing}"
    assert obj["layer_id"] == layer_id
    assert obj["status"] == "FROZEN_PASS", f"{layer_id}: status={obj['status']}"
    assert int(obj["row_count"]) > 0
    assert obj["date_start"] <= obj["date_end"]
    assert isinstance(obj["datasets"], list) and obj["datasets"]
    assert isinstance(obj["source_lineage"], list) and obj["source_lineage"]
    assert isinstance(obj["file_sha256"], dict) and obj["file_sha256"]
    for name, digest in obj["file_sha256"].items():
        assert SHA256_RE.fullmatch(str(digest)), f"{layer_id}: invalid sha256 for {name}"

    for source in obj.get("fallback_sources", []):
        assert source.get("equivalence_status") == "PASS", (
            f"{layer_id}: fallback {source.get('name')} not equivalence PASS"
        )
    return obj


def validate_full_manifests():
    contract = load_json(CONTRACT_PATH)
    required_fields = contract["layer_manifest_required_fields"]
    manifests = {}
    for layer in contract["layers"]:
        p = ROOT / layer["manifest_path"]
        assert p.exists(), f"missing required layer manifest: {p}"
        manifests[layer["layer_id"]] = validate_manifest(p, layer["layer_id"], required_fields)

    final = manifests["0F_FINAL_ASSEMBLY"]
    audit = final.get("cross_layer_audit") or {}
    gate = contract["stage0f_cross_layer_gate"]
    for k in gate["must_report"]:
        assert k in audit, f"0F_FINAL_ASSEMBLY missing cross_layer_audit.{k}"

    assert int(audit["future_join_violations"]) == 0
    assert int(audit["silent_zero_fill_violations"]) == 0
    assert int(audit["duplicate_decision_key_violations"]) == 0

    coverage = audit["per_layer_eligible_date_coverage"]
    for layer_id in ("0A_MARKET", "0B_INDUSTRY"):
        assert float(coverage[layer_id]) >= gate["daily_required_layer_eligible_date_coverage"], (
            f"{layer_id}: eligible-date coverage below gate"
        )

    input_hashes = audit["input_manifest_sha256"]
    for layer_id in EXPECTED_LAYERS[:-1]:
        layer_path = ROOT / next(x["manifest_path"] for x in contract["layers"] if x["layer_id"] == layer_id)
        assert input_hashes[layer_id] == sha256(layer_path), f"{layer_id}: assembly manifest hash mismatch"

    print("STAGE0 FULL LAYER/MANIFEST + CROSS-LAYER PIT AUDIT PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-only", action="store_true")
    args = ap.parse_args()
    validate_contract_only()
    if not args.contract_only:
        validate_full_manifests()


if __name__ == "__main__":
    main()
