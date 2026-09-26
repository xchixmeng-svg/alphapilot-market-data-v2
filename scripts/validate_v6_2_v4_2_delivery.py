#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v6_2_v4_2_contract import ContractError, load_jsonl, validate_packet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packets", type=Path)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    rows = load_jsonl(args.packets)
    if not rows:
        raise ContractError("no upgraded packets")
    for row in rows:
        validate_packet(row)
        if not isinstance(row.get("causal_context_read_only"), dict):
            raise ContractError("Stage C did not receive causal_context_read_only")
        tc = row.get("transport_contract", {})
        for flag in ("outcomes_opened", "selection_recomputed", "frozen_layers_modified"):
            if tc.get(flag) is not False:
                raise ContractError(f"transport flag failed: {flag}")
    result = {
        "status": "PASS",
        "packets": len(rows),
        "stage_c_received_causal_context_read_only": True,
        "raw_pe_pb_absent": True,
        "future_outcomes_absent": True,
        "contract_only": True,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
