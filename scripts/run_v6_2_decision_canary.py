#!/usr/bin/env python3
"""Eight-case, outcome-blind canary for the V6.2 decision contract."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evidence_capability import derive_case_evidence_sets
from run_v6_2_ai_replay64_shard import MODEL, call_with_retry

PREPARED = Path("prepared")
OUT = Path("canary_out")
OUT.mkdir(exist_ok=True)


def load_cases() -> list[dict]:
    summary = json.loads((PREPARED / "PREP_SUMMARY.json").read_text())
    if summary.get("2025_opened") is not False:
        raise RuntimeError("prepared artifact does not affirm 2025_opened=false")
    rows = [
        json.loads(line)
        for path in sorted(PREPARED.glob("cases_shard_*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 64 or len({r["case_id"] for r in rows}) != 64:
        raise RuntimeError("expected the immutable 64-case prepared artifact")
    if max(int(r["decision_date"]) for r in rows) > 20241231:
        raise RuntimeError("2025 packet leak")
    return rows


def select_cases(rows: list[dict]) -> list[dict]:
    enriched = []
    for pkt in rows:
        gaps = derive_case_evidence_sets(pkt["evidence"])["case_missing_but_system_supported_evidence_ids"]
        enriched.append((pkt, gaps))
    gapped = sorted((x for x in enriched if x[1]), key=lambda x: x[0]["case_id"])
    if not gapped or len(gapped) >= 8:
        raise RuntimeError(f"unexpected case-specific-gap packet count: {len(gapped)}")
    gapped_ids = {x[0]["case_id"] for x in gapped}
    full = sorted(
        (x for x in enriched if x[0]["case_id"] not in gapped_ids),
        key=lambda x: (-float(x[0]["numerical_reference_read_only"]["p_hit10_h120"]), x[0]["case_id"]),
    )[: 8 - len(gapped)]
    return [x[0] for x in gapped + full]


def main() -> None:
    selected = select_cases(load_cases())
    responses = []
    errors = []
    warnings = []
    manifest = []
    for pkt in selected:
        gaps = derive_case_evidence_sets(pkt["evidence"])["case_missing_but_system_supported_evidence_ids"]
        manifest.append(
            {
                "case_id": pkt["case_id"],
                "decision_date": pkt["decision_date"],
                "code": pkt["code"],
                "p_hit10_h120": pkt["numerical_reference_read_only"]["p_hit10_h120"],
                "case_missing_but_system_supported_evidence_ids": gaps,
            }
        )
        try:
            obj, elapsed, case_warnings = call_with_retry(pkt)
            responses.append(
                {
                    "case_id": pkt["case_id"],
                    "decision_date": pkt["decision_date"],
                    "code": pkt["code"],
                    "elapsed_seconds": elapsed,
                    "p_hit10_h120": pkt["numerical_reference_read_only"]["p_hit10_h120"],
                    "case_missing_but_system_supported_evidence_ids": gaps,
                    **obj,
                }
            )
            if case_warnings:
                warnings.append({"case_id": pkt["case_id"], "warnings": case_warnings})
            print(json.dumps({"case_id": pkt["case_id"], "decision": obj["decision"]}), flush=True)
        except Exception as exc:
            errors.append({"case_id": pkt["case_id"], "error": repr(exc)})
            print(json.dumps(errors[-1]), flush=True)

    decisions = Counter(x["decision"] for x in responses)
    qualities = Counter(x["evidence_quality"] for x in responses)
    reject_count = decisions.get("REJECT", 0)
    non_actionable = reject_count + decisions.get("WATCH", 0)
    summary = {
        "status": "PASS" if not errors and len(responses) == 8 else "FAIL",
        "model": MODEL,
        "selection_is_outcome_blind": True,
        "selection": "all case-specific-gap packets plus highest p_hit10_h120 packets among remaining cases, eight total",
        "requested": 8,
        "completed": len(responses),
        "errors": len(errors),
        "decision_counts": dict(sorted(decisions.items())),
        "evidence_quality_counts": dict(sorted(qualities.items())),
        "contract_warning_cases": len(warnings),
        "diagnostic_reject_collapse_flag": reject_count >= 6,
        "diagnostic_non_actionable_collapse_flag": non_actionable >= 7,
        "2025_opened": False,
    }
    (OUT / "CANARY_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "CANARY_RESPONSES.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n")
    (OUT / "CANARY_ERRORS.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
    (OUT / "CANARY_WARNINGS.json").write_text(json.dumps(warnings, ensure_ascii=False, indent=2) + "\n")
    (OUT / "CANARY_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("CANARY_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
