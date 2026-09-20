# Changelog v3: fixes for review of real canary runs 35482706294 / 35485028854

This review was different from the two before it: it was based on actual
execution against real qwen2.5:7b via Ollama, not offline synthetic tests.
Every point below is mapped to a concrete code change and a test that
would have caught the specific reported symptom.

## 1. Stage 2 contract structural failure

**Reported**: 8/8 Stage 2 cases failed, including after retries. Model
repeatedly produced `decision=CANDIDATE` with `entry.status=
NOT_ACTIONABLE` and null prices.

**Root cause**: one shared schema/enum space let the model express an
internally-contradictory combination -- nothing prevented "decision says
actionable" and "entry fields say not actionable" from both validating
independently against their own (too-permissive) rules.

**Fix**: Stage 2 split into `validate_stage2_decision_output()` (schema:
`STAGE2_DECISION_SCHEMA.json`, no entry/failure_exit keys accepted at
all) and `validate_stage2_actionability_output()` (schema:
`STAGE2_ACTIONABILITY_SCHEMA.json`, called only for CANDIDATE/
HIGH_CONVICTION, `entry.status` enum excludes NOT_ACTIONABLE,
`failure_exit.trigger_type` enum excludes UNAVAILABLE). For REJECT/WATCH,
`safe_finalizer.assemble_non_actionable_stage2()` sets the null block
deterministically -- no model call.

**Tests**: `test_stage2_decision_output_rejects_entry_field_entirely`,
`test_actionability_schema_structurally_excludes_not_actionable`,
`test_actionability_schema_structurally_excludes_unavailable_trigger`,
`test_actionability_null_prices_rejected`,
`test_non_actionable_decision_never_calls_actionability_stage`,
`test_actionable_decision_calls_actionability_stage_exactly_once`.

## 2. `_text_mentions_family()` false positives

**Reported**: "reasonable valuation" matched institutional_flow, mops_
material_information, price_volume_structure, AND revenue via generic
single-word substring matching ("flow", "information", "structure",
"price").

**Fix**: `FAMILY_ALIASES` dict of curated, multi-word (or otherwise
distinctive) phrases per family. `_text_mentions_family()` only matches
against these; no bare generic word is ever an alias by itself.

**Tests**: `test_reasonable_valuation_does_not_false_positive_on_other_
families` (exact reproduction), `test_generic_words_alone_never_match_
any_family`, `test_alias_matching_still_detects_real_mentions`,
`test_value_judgment_regression_no_longer_over_triggers_across_families`.

## 3. Stage 1 / Stage 2 valuation rule contradiction

**Reported**: Stage 1 allowed SUPPORTIVE/CONTRADICTORY from "absolute
valuation extreme" without a benchmark; Stage 2 banned any value-judgment
language without a benchmark. Directly contradictory instructions.

**Fix**: unified on the stricter rule. Stage 1's prompt no longer
mentions judging absolute magnitude; `validate_stage1_output()` now hard-
rejects any valuation observation with `benchmark_available=false` whose
`direction` is not `NOT_INTERPRETABLE`.

**Tests**: `test_stage1_valuation_without_benchmark_must_be_not_
interpretable`, `test_stage1_valuation_without_benchmark_supportive_
also_rejected`, `test_stage1_valuation_without_benchmark_not_
interpretable_passes`, `test_stage1_valuation_with_benchmark_can_be_
supportive_or_contradictory`.

## 4. Evidence-role constraints insufficient

**Reported**: NEUTRAL families placed in primary/secondary evidence;
CONTRADICTORY valuation sometimes omitted from counter_evidence.

**Fix**: (a) every family with Stage-1 direction strictly CONTRADICTORY
now MUST appear in `counter_evidence_ids` -- new hard error if omitted;
(b) explicit, specifically-worded rejection whenever a NEUTRAL or
NOT_INTERPRETABLE family appears in ANY of primary/secondary/counter
(previously only an algebraic consequence of other checks, now a named
error).

**Tests**: `test_contradictory_family_must_appear_in_counter_evidence_
ids`, `test_contradictory_family_correctly_listed_passes_that_check`,
`test_neutral_family_forbidden_in_any_role`, `test_not_interpretable_
family_forbidden_in_any_role`.

## 5. Retry ineffective

**Reported**: second Stage 2 attempt typically identical to the first
(same decision, NOT_ACTIONABLE, null prices).

**Fix**: `errors_to_repair_request()` parses the (already fixed-format,
since this module generates them) error strings into a compact
`{field, problem, received, allowed}` list, sent as a `repair_request`
field on every retry payload alongside (not instead of) the raw errors.

**Tests -- explicitly required by the review ("必須有fixture證明每種
Stage 2 failure能在retry修復")**: `test_repair_request_driven_retry_
fixes_decision_enum_violation`, `test_repair_request_driven_retry_fixes_
missing_counter_evidence`, `test_repair_request_driven_retry_fixes_
actionability_price_violation` -- each uses a "smart mock" callable that
reads `repair_request` off the payload and applies exactly the indicated
fix, proving the feedback channel carries enough information for a
mechanical repair to converge. This proves the channel works, NOT that
the real model will reliably use it -- that is only testable with a real
canary round.

## 6. Stage 3 unverified

**Reported**: no case ever reached Stage 3 with real output.

**Status**: still true, still not claimed fixed. No change was possible
here without a real Stage 2 pass to build on.

## 7. Performance

**Reported**: 316.47s mean per case, 367.35s max, zero finalized, across
round 2.

**Analysis**: the time was being spent on doomed retries against an
unsatisfiable contract (#1), not primarily on raw model latency.

**Fix**: `HistoryEntry.elapsed_seconds` now records real wall-clock time
per individual model call (previously only per-case totals existed).
`summarize_canary_results.py`'s "Repair / retry activity & per-stage
latency" section reports attempts/errors/mean/max latency per stage name
(`stage1`, `stage2_decision`, `stage2_actionability`, `stage3`, and their
`*_revision` counterparts). `OrchestratorConfig.per_case_timeout_seconds`
raised from 1500s to 2400s to match the new worst-case call count (~12
under the split design vs ~8 before), with the derivation documented in
the dataclass's docstring. The common REJECT/WATCH case now needs FEWER
calls than before (no Stage 2b at all), so real elapsed time for most
cases should fall once #1 is actually fixed for real.

**Tests**: `test_history_entries_record_elapsed_seconds`.

## 8. Possible Qwen2.5-7B capability ceiling

**Reported**: if the minimal canary still can't satisfy Stage 2 after
restructuring, don't keep loosening the contract to accommodate the
model -- consider a stronger model for Stage 2/critic while keeping
Qwen2.5-7B for Stage 1 (which passed 8/8 for real in round 2).

**Status**: acknowledged as the explicit fallback plan if the next real
round still fails at Stage 2b or reveals persistent unreliability.
**Not implemented** -- this delivery is the structural fix attempt first,
per the review's own instruction not to keep loosening the contract. If
the next round shows the split schema itself is still beyond the model's
reliable capability (as opposed to the previous shared-schema confusion),
that is the trigger to act on this contingency, not before.

## Already-consumed case_ids (31 total, do not claim fresh)

Spans canary rounds 35456479322 / 35458136917 / 35459949583 /
35482706294 / 35485028854:

```
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,24,32,38,39,40,47,48,50,53,55,57,61
```

`fresh_canary_selector.DEFAULT_EXCLUDE_CASE_IDS` and `workflows/
v6_2_canary.yml`'s `exclude_case_ids` input default both updated to match.

## Root cause fixed in run_fresh_canary.py itself

Independent of the review's 8 numbered points: run 35482706294's 0/11
Stage 1 failure was traced to `run_fresh_canary.py` sending Ollama
`"format": "json"` (loose) instead of the real JSON Schema. Each stage's
`call_fn` (built by `make_stage_call()`) now carries its own schema file,
passed as Ollama's `format` field for structured decoding. Tests:
`test_call_ollama_sends_real_json_schema_as_format_when_provided`,
`test_make_stage_call_binds_distinct_schema_per_stage`,
`test_load_schema_strips_meta_keys_ollama_would_reject`.

## Test count

105 tests total (was 92): 35 contract (+7), 13 orchestrator integration,
20 canary selector, 19 schema parity (+2), 18 run_fresh_canary (+5).
