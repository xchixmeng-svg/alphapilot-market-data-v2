#!/usr/bin/env python3
"""Pair consumed packets with exact causal-context sidecars for Stage C."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from v6_2_v4_2_contract import ContractError, index_exact, load_jsonl, norm_code, norm_date, validate_pair


def redact_raw_multiples(packet: dict) -> dict:
    out = copy.deepcopy(packet)
    evidence = out.get("evidence", {})
    valuation = evidence.get("valuation")
    if isinstance(valuation, dict):
        valuation.pop("pe", None)
        valuation.pop("pb", None)
        if not any(v is not None for v in valuation.values()):
            evidence.pop("valuation", None)
            ids = evidence.get("available_evidence_ids", [])
            evidence["available_evidence_ids"] = [x for x in ids if x != "valuation"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, nargs="+", required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    packets = []
    for path in args.cases:
        packets.extend(load_jsonl(path))
    sidecars = index_exact(load_jsonl(args.sidecar))
    upgraded = []
    for original in packets:
        packet = redact_raw_multiples(original)
        key = (int(packet["case_id"]), norm_date(packet["decision_date"]), norm_code(packet["code"]))
        if key not in sidecars:
            raise ContractError(f"missing exact sidecar: {key}")
        sidecar = sidecars[key]
        validate_pair(packet, sidecar)
        packet["causal_context_read_only"] = sidecar["causal_context_read_only"]
        packet["transport_contract"] = {k: v for k, v in sidecar.items() if k != "causal_context_read_only"}
        upgraded.append(packet)
    if len(upgraded) != len(sidecars):
        raise ContractError("orphan sidecar or packet count mismatch")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in upgraded), encoding="utf-8")
    print(json.dumps({"status": "PASS", "packets": len(upgraded), "stage_c_received_context": True}))


if __name__ == "__main__":
    main()
