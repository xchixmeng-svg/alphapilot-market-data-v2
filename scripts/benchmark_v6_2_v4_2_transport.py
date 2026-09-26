#!/usr/bin/env python3
"""Benchmark transport coverage without outcomes or prediction-value claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v6_2_v4_2_contract import ContractError, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = load_jsonl(args.results)
    keys = {(int(x["case_id"]), int(x["decision_date"]), str(x["code"]).zfill(4)) for x in rows}
    if len(rows) != args.expected or len(keys) != args.expected:
        raise ContractError("runner result count or exact-key uniqueness failed")
    if not all(x.get("stage_c_received_causal_context_read_only") is True for x in rows):
        raise ContractError("Stage C context receipt not proven")
    result = {
        "status": "PASS",
        "rows": len(rows),
        "exact_keys_unique": True,
        "stage_c_context_receipt_rate": 1.0,
        "outcomes_opened": False,
        "prediction_value_claimed": False,
        "scope": "transport benchmark only",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
