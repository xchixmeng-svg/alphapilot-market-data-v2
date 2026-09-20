# V6.2 Three-Stage Decision Architecture (revised after external review)

## What changed since the last delivery

An external review actually ran `py_compile`, the JSON schemas, and 34
synthetic tests against the previous delivery -- all passed -- but then
identified integration-level and logic-level problems the synthetic tests
didn't cover: missing files the workflow referenced, an unrealistic
timeout budget, retries that didn't feed the model any feedback, a
tautological boolean expression, validator exceptions that could crash a
batch, and several other real bugs. Every point in that review is fixed
in this delivery. **92 tests now pass** (up from 34), including new tests
written specifically to reproduce and catch each reported bug.

Full list of fixes, mapped to the review's numbering, is in
`CHANGELOG_REVIEW_FIXES.md`.

## What this replaces

This supersedes BOTH the earlier single-stage decision-contract patch
(`prompt_and_schema.py` / `normalize_and_validate.py` for
`scripts/run_v6_2_ai_replay64_shard.py`) AND the previous three-stage
delivery. Deploy this version only.

## Still true from the original delivery

- Root-cause verdict and architecture choice (3-stage over stronger-model)
  are unchanged; see the chat reply that accompanied the first delivery.
- `evidence_capability.py`'s system/case evidence split is unchanged.

## Files in this delivery

| File | Purpose |
|---|---|
| `evidence_capability.py` | System/case evidence split (unchanged) |
| `schemas/STAGE{1,2,3}_*.json` | Canonical schemas |
| `stage{1,2,3}_*_prompt.py` | SYSTEM prompts + payload builders (Stage1 prompt fixed: timeliness/relevance no longer conflated; related-party loans no longer defaulted to routine) |
| `deterministic_validators.py` | Rewritten: case_id check, numeric grounding, exception-safe, strict Stage3 |
| `safe_finalizer.py` | Unchanged API; now benefits from exception-safe validators underneath |
| `retry_orchestrator.py` | Rewritten: feedback-aware retries, hard timeouts, per-case exception isolation, full history |
| `fresh_canary_selector.py` | Rewritten: honest bucket names, no tautology, conservative MOPS classification, default exclusion list |
| `run_fresh_canary.py` | **NEW**: real Ollama/qwen2.5:7b HTTP client + CLI, fail-closed guards |
| `summarize_canary_results.py` | **NEW**: human-readable run summary |
| `tests/test_three_stage_contract.py` | 28 tests (was 14) |
| `tests/test_orchestrator_integration.py` | 14 tests (was 7), incl. timeout/retry-feedback/exception-safety regressions |
| `tests/test_fresh_canary_selector.py` | 20 tests (was 13), incl. tautology-fix and default-exclusion regressions |
| `tests/test_schema_runtime_parity.py` | **NEW**: 17 tests preventing schema/runtime drift |
| `tests/test_run_fresh_canary.py` | **NEW**: 13 tests, incl. mocked Ollama response parsing and fail-closed guards |
| `workflows/v6_2_contract_only.yml` | Updated to run all 5 test files |
| `workflows/v6_2_canary.yml` | Rewritten: real artifact download, real Ollama setup, `if: always()` upload |

## What I still could NOT verify

- **`run_fresh_canary.py`'s Ollama HTTP client has never been run against
  a real Ollama server.** This sandbox has no network access. The
  request/response shape follows Ollama's documented `/api/chat` API, and
  the JSON-parsing logic is unit-tested against a mocked response
  (`test_call_ollama_parses_valid_response`), but the actual network call
  is unverified. Expect to need small fixes (e.g. if your Ollama version's
  response envelope differs) the first time you run it for real.
- **No Stage 1/2/3 prompt has been run against real qwen2.5:7b output.**
  The prompt fixes for timeliness/relevance conflation and related-party
  loan handling are design fixes, not empirically validated against real
  model behavior. The fresh canary run is still what actually tests this.
- The `gh api .../artifacts/{id}/zip` step in `v6_2_canary.yml` follows
  GitHub's documented artifact-download API and mirrors the pattern
  described for your existing successful workflows, but was not tested
  against your actual repository/artifact.

## Key fixes explained

**Timeout budget (review point 7):** the old 240s per-case default could
never have completed a 3-stage pipeline given the reported ~149s single
real-call latency. `OrchestratorConfig` now defaults to
`per_call_timeout_seconds=300` and `per_case_timeout_seconds=2500`, updated
after the first real three-stage canary showed multiple Qwen calls
exceeding 180 seconds. Both are configurable.

**Hard timeout (review point 8):** every model call now goes through
`concurrent.futures.ThreadPoolExecutor.submit(...).result(timeout=...)`,
and the deadline is re-checked immediately after any call returns, before
its result is accepted. A result arriving late is discarded, never
finalized. Python cannot forcibly kill a native thread, so your actual
Ollama client should also set its own request timeout (this is
implemented in `run_fresh_canary.py`'s `call_ollama`) so the underlying
HTTP connection is aborted, not just abandoned at the orchestrator level.

**Retry feedback (review point 6):** every payload builder used by the
orchestrator now receives the previous attempt's raw output and its exact
validation errors on retry, injected as a `previous_attempt` field (or
`critic_feedback` for the Stage2-revision path). Verified by
`test_retry_second_attempt_payload_contains_first_attempt_errors` and
`test_stage2_revision_payload_contains_critic_feedback_and_prior_errors`.

**Exception isolation (review points 9, and general hardening):** every
`validate_*` function in `deterministic_validators.py` now wraps its body
in try/except and converts any exception into a normal contract-error
string. `retry_orchestrator.py` adds a second layer of defense (wrapping
validator calls) plus an outermost per-case try/except inside
`run_batch()`. A single malformed field can no longer crash a case, and a
single case can no longer crash a batch.

**Grounding (review point 12):** `_ground_numeric_claims()` extracts every
number in a Stage1 `exact_observations` string and checks it against the
actual raw numeric values in that SAME evidence family's packet data
(~1% tolerance). A fabricated number, or a real number borrowed from the
wrong family, is now rejected.

**case_id check (review point 11):** Stage1 output's `case_id` is checked
for strict type-and-value equality against the case actually requested.

**Stage3 strictness (review point 13):** `validate_stage3_output` now
checks every problem's `field` enum, non-empty string `claim_text`,
null-or-known-evidence-id `cited_evidence_id`, valid `problem_type` enum,
and non-empty `explanation` -- individually, with a specific error per
violation, not a single generic pass/fail.

**Actionable price grounding (review point 14):** entry/failure-exit
prices for CANDIDATE/HIGH_CONVICTION are now checked for plausibility
(within 0.5x-2x) against the actual price numbers Stage1 observed for
`price_volume_structure`.

**Tautology fix (review point 20):** `_institutional_flow_direction_signal()`
replaces the old `has_negative and (has_positive or True)` expression
(identical to `has_negative` regardless of `has_positive`) with an honest
4-way classification (NEGATIVE_ONLY / MIXED / POSITIVE_ONLY /
NEUTRAL_OR_UNKNOWN).

**Related-party loans (review point 17):** removed from the blanket
ROUTINE_MOPS_KEYWORDS auto-classification, both in the canary selector
and in the Stage1 prompt's instructions -- these now require
amount/counterparty/rationale judgment, defaulting to NOT_INTERPRETABLE
with a stated limitation when that detail is absent.

**Timeliness vs relevance (review point 16):** the Stage1 prompt no longer
tells the model that routine content should be marked STALE regardless of
actual age. `timeliness` is now defined strictly as data recency;
"routine and non-catalytic" is expressed via `relevance_to_hypothesis_space`
and `limitations` instead.

## Fresh canary defaults

`fresh_canary_selector.select_fresh_canary()` now defaults to excluding
the 12 case_ids already used across canary runs 35456479322 /
35458136917 / 35459949583 (`DEFAULT_EXCLUDE_CASE_IDS`), and
`v6_2_canary.yml`'s `exclude_case_ids` input defaults to the same list.
Override either if you want to explicitly include or exclude a different
set.
