#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from v6_2_decision_contract import (
    SYSTEM,
    build_user_payload,
    normalize_output,
    validate,
)

MODEL = os.environ.get("V62_LOCAL_MODEL", "qwen2.5:7b")
OUT = Path("out")
OUT.mkdir(exist_ok=True)


def call(pkt: dict, repair: str | None = None) -> tuple[dict, float, list[str]]:
    user = build_user_payload(pkt)
    if repair:
        user["previous_contract_error"] = repair
        user["repair_instruction"] = (
            "Preserve the substantive analysis but repair every reported contract error. "
            "Use exact enum values and only evidence IDs from the allowed arrays. Do not "
            "invent evidence or alter immutable numerical forecasts."
        )
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 950},
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        raw = json.loads(response.read().decode())
    obj = json.loads(raw["message"]["content"].strip())
    obj = normalize_output(pkt, obj)
    warnings = validate(pkt, obj)
    return obj, time.time() - started, warnings


def call_with_retry(pkt: dict, attempts: int = 3) -> tuple[dict, float, list[str]]:
    error: str | None = None
    for _attempt in range(1, attempts + 1):
        try:
            return call(pkt, error)
        except Exception as exc:
            error = repr(exc)
    raise RuntimeError(error)


def main(shard: int) -> None:
    cases = [
        json.loads(line)
        for line in Path(f"prepared/cases_shard_{shard}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows: list[dict] = []
    errors: list[dict] = []
    warnings: list[dict] = []
    for pkt in cases:
        obj = None
        elapsed = None
        case_warnings: list[str] = []
        err = None
        try:
            obj, elapsed, case_warnings = call_with_retry(pkt)
        except Exception as exc:
            err = repr(exc)
        identity = {"case_id": pkt["case_id"], "date": pkt["decision_date"], "code": pkt["code"]}
        if obj is None:
            errors.append({**identity, "error": err})
            print("ERROR", errors[-1], flush=True)
            continue
        if case_warnings:
            warnings.append({**identity, "warnings": case_warnings})
        rows.append(
            {
                **identity,
                "elapsed_seconds": elapsed,
                "p_hit10_h120": pkt["numerical_reference_read_only"]["p_hit10_h120"],
                **obj,
            }
        )
        print(
            json.dumps(
                {"case_id": pkt["case_id"], "code": pkt["code"], "decision": obj["decision"], "seconds": round(elapsed, 2)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    (OUT / f"responses_shard_{shard}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (OUT / f"errors_shard_{shard}.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
    (OUT / f"contract_warnings_shard_{shard}.json").write_text(json.dumps(warnings, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "shard": shard,
        "requested": len(cases),
        "completed": len(rows),
        "errors": len(errors),
        "contract_warning_cases": len(warnings),
        "model": MODEL,
        "api_cost_usd": 0,
    }
    (OUT / f"summary_shard_{shard}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY=" + json.dumps(summary), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    main(parser.parse_args().shard)
