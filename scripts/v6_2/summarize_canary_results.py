#!/usr/bin/env python3
"""
summarize_canary_results.py

Reads the output of run_fresh_canary.py and prints a human-readable
summary. This is intentionally NOT an auto-pass/fail gate (per design
section F: success criteria require human judgment, not a hard-coded
formula) -- it prints numbers for a person to read and decide.

Usage:
    python3 summarize_canary_results.py --input-dir ./canary_output
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def summarize(input_dir: Path) -> None:
    results_path = input_dir / "canary_case_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    n = len(results)
    print("=" * 72)
    print(f"V6.2 Fresh Canary Summary -- {n} case(s)")
    print("=" * 72)

    status_counts = Counter(r["status"] for r in results)
    print("\n-- Completion status --")
    for status, count in status_counts.items():
        print(f"  {status}: {count} ({_pct(count, n)})")

    finalized = [r for r in results if r["status"] == "FINALIZED"]
    print(f"\nContract-valid completion rate: {_pct(len(finalized), n)}")
    print(f"MODEL_CONTRACT_FAILURE rate: {_pct(status_counts.get('MODEL_CONTRACT_FAILURE', 0), n)}")

    print("\n-- Decision distribution (finalized cases only) --")
    decision_counts = Counter(r["final_stage2_output"]["decision"] for r in finalized if r.get("final_stage2_output"))
    for d, count in decision_counts.items():
        print(f"  {d}: {count} ({_pct(count, len(finalized))})")
    if len(decision_counts) <= 1 and finalized:
        print("  *** WARNING: decision has NO real distribution (all one value). ***")
        print("  *** This alone does not prove a problem, but it is the exact ***")
        print("  *** pattern (reflexive REJECT, then reflexive CANDIDATE) that ***")
        print("  *** motivated this redesign. Investigate before trusting this run. ***")

    print("\n-- Evidence quality distribution (finalized cases only) --")
    eq_counts = Counter(r["final_stage2_output"]["evidence_quality"] for r in finalized if r.get("final_stage2_output"))
    for eq, count in eq_counts.items():
        print(f"  {eq}: {count} ({_pct(count, len(finalized))})")
    if len(eq_counts) <= 1 and finalized:
        print("  *** WARNING: evidence_quality has NO real distribution either. ***")

    print("\n-- Repair / retry activity --")
    total_attempts_by_stage = Counter()
    total_errors_by_stage = Counter()
    for r in results:
        for h in r["raw_history"]:
            total_attempts_by_stage[h["stage"]] += 1
            if h["errors"]:
                total_errors_by_stage[h["stage"]] += 1
    for stage in sorted(total_attempts_by_stage):
        attempts = total_attempts_by_stage[stage]
        errs = total_errors_by_stage[stage]
        print(f"  {stage}: {attempts} attempt(s) across all cases, {errs} had contract errors ({_pct(errs, attempts)})")

    print("\n-- Latency --")
    elapsed = sorted(r["elapsed_seconds"] for r in results)
    if elapsed:
        mean = sum(elapsed) / len(elapsed)
        p95_idx = max(0, int(len(elapsed) * 0.95) - 1)
        print(f"  mean elapsed: {mean:.1f}s")
        print(f"  p95 elapsed:  {elapsed[p95_idx]:.1f}s")
        print(f"  max elapsed:  {elapsed[-1]:.1f}s")

    print("\n-- system_limitations neutrality check --")
    suspicious = []
    for r in results:
        if r["status"] == "MODEL_CONTRACT_FAILURE" and r.get("failure_reason"):
            reason = r["failure_reason"]
            if "system_limitations" in reason or "system_unavailable" in reason:
                suspicious.append((r["case_id"], reason))
    if suspicious:
        print(f"  *** {len(suspicious)} MODEL_CONTRACT_FAILURE case(s) mention system_limitations/")
        print(f"      system_unavailable in their failure_reason -- review these specifically: ***")
        for cid, reason in suspicious:
            print(f"      case_id={cid}: {reason}")
    else:
        print("  No MODEL_CONTRACT_FAILURE case's failure_reason mentions system_limitations/system_unavailable.")

    print("\n" + "=" * 72)
    print("This is a descriptive summary, not an automated pass/fail gate.")
    print("A human should read the decision/evidence_quality distributions,")
    print("spot-check a sample of finalized cases' Stage 1 direction calls")
    print("against the underlying packet numbers, and review any")
    print("MODEL_CONTRACT_FAILURE cases' raw_history before trusting this run.")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    args = parser.parse_args()
    summarize(Path(args.input_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
