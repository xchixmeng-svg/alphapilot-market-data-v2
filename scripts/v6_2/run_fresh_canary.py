#!/usr/bin/env python3
"""
run_fresh_canary.py

CLI entry point that wires a REAL Ollama-served qwen2.5:7b model into the
three-stage pipeline (evidence_capability.py / deterministic_validators.py
/ safe_finalizer.py / retry_orchestrator.py / fresh_canary_selector.py)
and runs an outcome-blind fresh canary.

IMPORTANT, STATED HONESTLY: this file has never been executed against a
real Ollama server in the environment it was written in (no model access,
no network access there). The Ollama HTTP client below (`call_ollama`) is
written against Ollama's documented /api/chat request/response shape, and
its JSON-parsing logic is covered by an offline unit test using a
mocked HTTP response (tests/test_run_fresh_canary.py) -- but the network
call itself is unverified. Test it against your real
http://127.0.0.1:11434/api/chat + qwen2.5:7b setup before trusting it in
CI, and expect to need small fixes if Ollama's actual response shape
differs from what is assumed here.

Data sources read (LOCAL FILES ONLY, no network for data):
    data/prepared/cases_shard_*.jsonl
    data/prepared/PREP_SUMMARY.json
This script NEVER reads HIDDEN_OUTCOMES.csv, HIDDEN_FUTURE_PATHS.parquet,
or any 2025 data. The 2025/case-count/date guard below is FAIL-CLOSED:
if PREP_SUMMARY.json is missing, malformed, or does not explicitly
declare 2025_opened=false, this script exits with a non-zero status and
runs nothing (fixed after external review -- the previous version only
printed a warning and continued).

Usage:
    python3 run_fresh_canary.py \\
        --max-cases 12 \\
        --exclude-case-ids "5,7,14,15,16,24,32,38,39,48,55,57" \\
        --per-case-timeout-seconds 1500 \\
        --output-dir ./canary_output \\
        [--prepared-dir ./data/prepared] \\
        [--ollama-url http://127.0.0.1:11434/api/chat] \\
        [--model qwen2.5:7b]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_capability import derive_case_evidence_sets  # noqa: E402
from fresh_canary_selector import select_fresh_canary  # noqa: E402
from retry_orchestrator import CaseResult, HistoryEntry, OrchestratorConfig, run_batch  # noqa: E402
from stage1_extractor_prompt import STAGE1_SYSTEM, build_stage1_user_payload  # noqa: E402
from stage2_decision_prompt import STAGE2_SYSTEM as STAGE2_DECISION_SYSTEM, build_stage2_decision_user_payload  # noqa: E402
from stage2_actionability_prompt import STAGE2_ACTIONABILITY_SYSTEM, build_stage2_actionability_user_payload  # noqa: E402
from stage3_critic_prompt import STAGE3_SYSTEM, build_stage3_user_payload  # noqa: E402

HARD_MAX_CASES = 20  # hard safety cap, independent of whatever --max-cases is passed


class GuardFailure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Fail-closed data-integrity guard (fixed after external review: the
# previous version only warned and continued when PREP_SUMMARY.json was
# missing; this version refuses to run anything).
# ---------------------------------------------------------------------------


def load_and_guard_prep_summary(prepared_dir: Path) -> dict:
    path = prepared_dir / "PREP_SUMMARY.json"
    if not path.exists():
        raise GuardFailure(f"PREP_SUMMARY.json not found at {path}. Refusing to run -- this guard is fail-closed.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        raise GuardFailure(f"PREP_SUMMARY.json is not valid JSON: {e}")

    if summary.get("2025_opened") is not False:
        raise GuardFailure(
            f"PREP_SUMMARY.json declares 2025_opened={summary.get('2025_opened')!r}, not explicitly False. "
            f"Refusing to run any canary against this data."
        )
    return summary


def load_packets(prepared_dir: Path) -> list[dict]:
    shard_paths = sorted(prepared_dir.glob("cases_shard_*.jsonl"))
    if not shard_paths:
        raise GuardFailure(f"No cases_shard_*.jsonl files found under {prepared_dir}")

    packets = []
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    packets.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise GuardFailure(f"{path.name}:{line_no}: invalid JSON ({e})")

    case_ids = [p.get("case_id") for p in packets]
    if len(case_ids) != len(set(case_ids)):
        dupes = [c for c in case_ids if case_ids.count(c) > 1]
        raise GuardFailure(f"Duplicate case_id values in prepared packets: {sorted(set(dupes))}")

    for p in packets:
        date = p.get("decision_date")
        if not isinstance(date, int) or date > 20241231:
            raise GuardFailure(f"case_id={p.get('case_id')}: decision_date={date!r} is missing or beyond 20241231.")

    return packets


def validate_immutable_packet_set(summary: dict, packets: list[dict]) -> None:
    """Require the exact immutable 64-case prepared-artifact contract."""
    if summary.get("cases") != 64:
        raise GuardFailure(f"PREP_SUMMARY.json must declare cases=64; got {summary.get('cases')!r}.")
    if len(packets) != 64:
        raise GuardFailure(f"Immutable prepared artifact must contain exactly 64 packets; got {len(packets)}.")


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------


def call_ollama(system_prompt: str, user_payload: dict, *, model: str, url: str, timeout_s: float, json_schema: dict | None = None) -> dict:
    """
    POSTs to Ollama's /api/chat endpoint and returns the model's response
    parsed as JSON. Sets its own request-level timeout (belt-and-suspenders
    with retry_orchestrator's own hard per-call timeout).

    FIXED after real canary run 35482706294 (Stage 1: 0/11 finalized, all
    Stage 1 calls failed): the previous version of this function sent
    "format": "json" (Ollama's loose "some JSON, shape unconstrained"
    mode) instead of the actual JSON Schema. Ollama's /api/chat `format`
    field also accepts a full JSON Schema object for constrained/
    structured decoding (a much stronger guarantee for a 7B model than
    "some JSON"), which is what every call in this file now passes -- see
    make_stage_call()'s json_schema argument.

    Raises on any failure (HTTP error, non-JSON response body, non-JSON
    model content) -- the orchestrator's retry loop treats any exception
    from this function as a normal failed attempt, not a crash.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "format": json_schema if json_schema is not None else "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw_body = resp.read().decode("utf-8")

    outer = json.loads(raw_body)
    content = outer.get("message", {}).get("content")
    if content is None:
        raise RuntimeError(f"Ollama response missing message.content: {outer!r}")
    return json.loads(content)


def _load_schema(name: str) -> dict:
    """Loads a schema file and strips the top-level meta keys ($schema,
    $id, schema_version, title) that are useful for humans/tests but are
    not part of what Ollama's structured-output `format` field expects --
    some Ollama versions reject or ignore a `format` object with unknown
    top-level keys, so only the actual constraint keys are sent."""
    schema_path = Path(__file__).resolve().parent / "schemas" / name
    with open(schema_path, "r", encoding="utf-8") as f:
        full = json.load(f)
    return {k: v for k, v in full.items() if k not in ("$schema", "$id", "schema_version", "title")}


def make_stage_call(model: str, url: str, timeout_s: float, schema_name: str):
    """Returns a call_fn bound to ONE specific stage's JSON Schema, passed
    to Ollama's `format` field for structured decoding. Each of Stage1,
    Stage2-decision, Stage2-actionability, and Stage3 gets its own call_fn
    with its own schema -- they must never share one, since their shapes
    are now deliberately different (that's the whole point of the
    Stage2a/2b split)."""
    schema = _load_schema(schema_name)

    def _call(system_prompt: str, user_payload: dict) -> dict:
        return call_ollama(
            system_prompt, user_payload, model=model, url=url,
            timeout_s=timeout_s, json_schema=schema,
        )
    return _call


# ---------------------------------------------------------------------------
# Serialization (CaseResult / HistoryEntry -> JSON)
# ---------------------------------------------------------------------------


def _serialize_case_result(r: CaseResult) -> dict:
    d = dataclasses.asdict(r)
    return d


def write_outputs(results: list[CaseResult], output_dir: Path, selector_result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "canary_case_results.json", "w", encoding="utf-8") as f:
        json.dump([_serialize_case_result(r) for r in results], f, ensure_ascii=False, indent=2, default=str)

    with open(output_dir / "canary_selector_result.json", "w", encoding="utf-8") as f:
        serializable_selector = dict(selector_result)
        serializable_selector["tags"] = {
            str(cid): dataclasses.asdict(tag) for cid, tag in selector_result["tags"].items()
        }
        json.dump(serializable_selector, f, ensure_ascii=False, indent=2, default=str)

    decision_counts: dict[str, int] = {}
    for r in results:
        if r.status == "FINALIZED" and r.final_stage2_output:
            d = r.final_stage2_output.get("decision", "UNKNOWN")
            decision_counts[d] = decision_counts.get(d, 0) + 1
    n_finalized = sum(1 for r in results if r.status == "FINALIZED")
    n_failed = sum(1 for r in results if r.status == "MODEL_CONTRACT_FAILURE")

    summary = {
        "n_cases": len(results),
        "n_finalized": n_finalized,
        "n_model_contract_failure": n_failed,
        "decision_counts": decision_counts,
        "mean_elapsed_seconds": (sum(r.elapsed_seconds for r in results) / len(results)) if results else None,
        "max_elapsed_seconds": max((r.elapsed_seconds for r in results), default=None),
    }
    with open(output_dir / "canary_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--exclude-case-ids", type=str, default="")
    parser.add_argument("--per-case-timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--per-call-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prepared-dir", type=str, default="data/prepared")
    parser.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)

    try:
        prep_summary = load_and_guard_prep_summary(prepared_dir)
        packets = load_packets(prepared_dir)
        validate_immutable_packet_set(prep_summary, packets)
    except GuardFailure as e:
        print(f"GUARD FAILURE: {e}", file=sys.stderr)
        return 1

    exclude_ids = None
    if args.exclude_case_ids.strip():
        exclude_ids = {int(x) for x in args.exclude_case_ids.split(",") if x.strip()}

    if args.max_cases < 1:
        print("GUARD FAILURE: --max-cases must be at least 1.", file=sys.stderr)
        return 1
    if args.per_case_timeout_seconds <= 0 or args.per_call_timeout_seconds <= 0:
        print("GUARD FAILURE: timeout values must be positive.", file=sys.stderr)
        return 1

    max_cases = min(args.max_cases, HARD_MAX_CASES)
    if args.max_cases > HARD_MAX_CASES:
        print(f"WARNING: --max-cases={args.max_cases} exceeds the hard cap of {HARD_MAX_CASES}; using {HARD_MAX_CASES}.")

    selector_result = select_fresh_canary(packets, per_bucket_target=2, exclude_case_ids=exclude_ids)
    selected_ids = selector_result["selected_case_ids"][:max_cases]
    if len(selector_result["selected_case_ids"]) > max_cases:
        print(f"NOTE: selector found {len(selector_result['selected_case_ids'])} candidate cases; "
              f"truncating to {max_cases} per --max-cases.")

    packets_by_id = {p["case_id"]: p for p in packets}
    selected_packets = [packets_by_id[cid] for cid in selected_ids]
    if not selected_packets:
        print("GUARD FAILURE: selector produced no eligible canary cases.", file=sys.stderr)
        return 1
    numerical_priors = {p["case_id"]: p.get("numerical_reference_read_only", {}) for p in selected_packets}

    print(f"Selected {len(selected_packets)} case(s): {selected_ids}")
    print(f"Bucket coverage: {json.dumps(selector_result['bucket_coverage'], indent=2)}")
    if selector_result["uncovered_buckets"]:
        print(f"WARNING: no eligible packets found for bucket(s): {selector_result['uncovered_buckets']}")

    config = OrchestratorConfig(
        per_case_timeout_seconds=args.per_case_timeout_seconds,
        per_call_timeout_seconds=args.per_call_timeout_seconds,
    )
    # FIXED after run 35482706294: each stage gets its OWN call_fn bound
    # to its OWN JSON Schema (passed to Ollama's structured-output
    # `format` field), never a shared loose "format": "json".
    stage1_call = make_stage_call(args.model, args.ollama_url, args.per_call_timeout_seconds, "STAGE1_EVIDENCE_DIRECTION_SCHEMA.json")
    stage2_decision_call = make_stage_call(args.model, args.ollama_url, args.per_call_timeout_seconds, "STAGE2_DECISION_SCHEMA.json")
    stage2_actionability_call = make_stage_call(args.model, args.ollama_url, args.per_call_timeout_seconds, "STAGE2_ACTIONABILITY_SCHEMA.json")
    stage3_call = make_stage_call(args.model, args.ollama_url, args.per_call_timeout_seconds, "STAGE3_CRITIC_SCHEMA.json")

    # Checkpoint every completed case so a cancelled or failed long run
    # still preserves all completed histories.
    results: list[CaseResult] = []
    for packet in selected_packets:
        one = run_batch(
            [packet], numerical_priors,
            stage1_call, stage2_decision_call, stage2_actionability_call, stage3_call,
            STAGE1_SYSTEM, STAGE2_DECISION_SYSTEM, STAGE2_ACTIONABILITY_SYSTEM, STAGE3_SYSTEM,
            build_stage1_user_payload, build_stage2_decision_user_payload,
            build_stage2_actionability_user_payload, build_stage3_user_payload,
            config=config,
        )[0]
        results.append(one)
        write_outputs(results, output_dir, selector_result)
        print(json.dumps({
            "completed_case_id": one.case_id,
            "status": one.status,
            "elapsed_seconds": one.elapsed_seconds,
            "failure_stage": one.failure_stage,
        }, ensure_ascii=False), flush=True)

    return 2 if any(r.status != "FINALIZED" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
