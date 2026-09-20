# V6.2 Three-Stage Decision Architecture (v3 -- after real-model canary failures)

## Status: NOT canary-ready. Do not run the 64-case.

Two prior versions of this delivery passed offline synthetic tests (34,
then 92) but had never been run against a real model. When they finally
were (canary runs 35482706294 and 35485028854, against real qwen2.5:7b
via Ollama), they failed:

- Run 35482706294: 0/11 finalized. Stage 1 failed 100% of the time,
  because the runner was sending Ollama the loose `"format": "json"`
  instead of the actual JSON Schema.
- Run 35485028854 (after that fix): Stage 1 passed 8/8. Stage 2 failed
  8/8, including after retries. The model repeatedly produced
  `decision=CANDIDATE` together with `entry.status=NOT_ACTIONABLE` and
  null prices -- an internally-contradictory combination the old single
  combined Stage 2 schema allowed it to express.
- Mean elapsed time per case: 316.47s, for zero finalized cases.

This delivery is a **structural** response to that, not another rule
added on top of the same shared schema. **It has not itself been run
against a real model yet.** Test it against a real canary round (using
the 31 already-exhausted case_ids below as exclusions) before drawing any
conclusion about whether it works, and report back before requesting
another round or the 64-case run.

## The core architectural change

Stage 2 is now **two separate model calls**, not one:

1. **`stage2_decision_prompt.py`** (Stage 2a) -- decision, evidence
   quality, thesis, evidence-id roles. Its JSON Schema has no `entry` or
   `failure_exit` fields at all. Sending them back is itself a contract
   violation, independent of their content.
2. **`stage2_actionability_prompt.py`** (Stage 2b) -- entry/failure_exit
   ONLY. Called **only** when Stage 2a's decision came back
   CANDIDATE/HIGH_CONVICTION. Its schema's `entry.status` enum is
   `{NOW, WAIT_FOR_PULLBACK, WAIT_FOR_CONFIRMATION}` -- `NOT_ACTIONABLE`
   is not a valid value in this call's vocabulary at all. Same for
   `failure_exit.trigger_type` excluding `UNAVAILABLE`.

For REJECT/WATCH, `entry`/`failure_exit` is **never asked of the model**.
`safe_finalizer.assemble_non_actionable_stage2()` sets it deterministically
to the one fixed value the contract permits (see that module's docstring
for why this is assembly, not invention). This also means the common
case (REJECT/WATCH -- the original decision-collapse investigation found
this was ~94% of cases) now costs **fewer** model calls than the old
design, not more.

The reported failure -- `decision=CANDIDATE` + `entry.status=
NOT_ACTIONABLE` + null prices -- required a single call to pick an
internally-contradictory combination from one shared enum space. That
specific combination is now unrepresentable: producing it would require
Stage 2a to emit fields it was never asked for and whose presence is
itself rejected, AND Stage 2b (which only runs for actionable decisions)
to emit a status value that isn't in its schema. Both would have to fail
in two different, structurally incompatible ways simultaneously.

## Every review point, addressed

| # | Review point | Fix |
|---|---|---|
| 1 | Stage 2 structural failure (CANDIDATE + NOT_ACTIONABLE + null price) | Stage 2a/2b split (above). See `CHANGELOG_REVIEW_FIXES.md` for the full mapping. |
| 2 | `_text_mentions_family()` false positives ("reasonable valuation" matching 5 unrelated families) | Replaced with `FAMILY_ALIASES`: curated per-family phrase lists, never generic single words. |
| 3 | Stage 1/Stage 2 valuation rules contradicted each other | Unified: valuation with `benchmark_available=false` MUST be `NOT_INTERPRETABLE`. Hard error in Stage 1 validation now, not just a Stage 2 text-pattern ban. |
| 4 | Evidence-role constraints insufficient | Added: every strictly-CONTRADICTORY family MUST appear in `counter_evidence_ids` (not optional); explicit, clearly-worded rejection of NEUTRAL/NOT_INTERPRETABLE families in any evidence-id role. |
| 5 | Retry ineffective (2nd attempt = same output) | `errors_to_repair_request()` converts errors into compact `{field, problem, received, allowed}` dicts sent on every retry. Proven via "smart mock" fixtures in `test_orchestrator_integration.py` that read `repair_request` and apply the fix -- see `test_repair_request_driven_retry_fixes_*`. |
| 6 | Stage 3 unverified | Still true. No case has ever reached Stage 3 with real model output. Not claimed fixed. |
| 7 | Performance (316s mean, zero output) | Root cause was retries burning time on a doomed contract, not raw latency -- fixed by #1/#5. `HistoryEntry.elapsed_seconds` now captures real per-call latency; `summarize_canary_results.py` reports per-stage breakdowns. Budget recalibrated (worst case ~12 calls under the new design; see `OrchestratorConfig`). |
| 8 | Possible Qwen2.5-7B ceiling | Acknowledged. If the next real canary round still fails at Stage 2 after this restructure, the recommended next step is Stage 1 stays on Qwen2.5-7B (it already passed 8/8 for real), Stage 2/critic move to a stronger model. Not implemented here -- a contingency, not a current change. |

Full point-by-point mapping, including the "already consumed" case_id
list and every named regression test, is in `CHANGELOG_REVIEW_FIXES.md`.

## Files in this delivery

| File | Purpose |
|---|---|
| `evidence_capability.py` | System/case evidence split (unchanged) |
| `schemas/STAGE1_EVIDENCE_DIRECTION_SCHEMA.json` | Unchanged |
| `schemas/STAGE2_DECISION_SCHEMA.json` | **Rewritten**: no entry/failure_exit |
| `schemas/STAGE2_ACTIONABILITY_SCHEMA.json` | **NEW**: entry/failure_exit only, non-actionable values excluded from enums |
| `schemas/STAGE3_CRITIC_SCHEMA.json` | Updated: `entry_or_failure_exit` field value, `ACTIONABILITY_PRICE_NOT_GROUNDED` problem type |
| `stage1_extractor_prompt.py` | Updated: unified valuation rule |
| `stage2_decision_prompt.py` | **Rewritten**: decision-only, tightened evidence-role rules |
| `stage2_actionability_prompt.py` | **NEW**: actionability-only |
| `stage3_critic_prompt.py` | Updated: actionability-price check added to scope |
| `deterministic_validators.py` | **Rewritten**: alias fix, unified valuation rule, split Stage2 validators, new hard rules, `errors_to_repair_request()` |
| `safe_finalizer.py` | **Rewritten**: pure deterministic assembly, no repair logic (unnecessary now) |
| `retry_orchestrator.py` | **Rewritten**: 4-call flow (stage1/stage2a/stage2b/stage3), structured repair requests on retry, per-call latency capture, recalibrated timeout budget |
| `fresh_canary_selector.py` | Updated: `DEFAULT_EXCLUDE_CASE_IDS` now 31 case_ids |
| `run_fresh_canary.py` | **Fixed root cause**: each stage now gets its own JSON Schema passed to Ollama's `format` field (was the loose `"json"` string, which is why Stage 1 failed 100% in run 35482706294) |
| `summarize_canary_results.py` | Updated: per-stage latency breakdown |
| `tests/*.py` | Rewritten/extended: **105 tests** (was 92) |
| `workflows/v6_2_canary.yml` | Updated: new exclude list default, recalibrated timeout |
| `workflows/v6_2_contract_only.yml` | Unchanged (still runs all 5 test files) |

## What I still could NOT verify

- **None of this has been run against a real model.** Same limitation as
  every prior version -- this sandbox has no network/Ollama access. The
  Stage 2a/2b split, the repair-request mechanism, and the JSON-Schema-
  passing fix are all design/code fixes for the specific failures the
  real runs exhibited, verified with mocks and synthetic fixtures, not
  with the actual model.
- **Whether qwen2.5:7b can reliably follow the new, more constrained
  actionability schema is unknown.** The schema now forbids it from
  saying "not actionable" in that call at all -- if the model still
  wants to express uncertainty about pricing even when the decision layer
  said CANDIDATE, forcing it into NOW/WAIT_FOR_PULLBACK/
  WAIT_FOR_CONFIRMATION could produce low-quality-but-technically-valid
  prices rather than a clean failure. Watch for this specifically in the
  next real canary round -- it's a new failure mode this design could in
  principle introduce even while fixing the reported one.
- Whether Ollama's structured-output `format` field actually accepts
  these schema shapes as-is (particularly the `type: ["number","null"]`
  style unions still used in a few older schema files, and whether
  stripping `$schema`/`$id`/`title` is sufficient) is unverified.

## Already-consumed case_ids -- do not claim as fresh

`fresh_canary_selector.DEFAULT_EXCLUDE_CASE_IDS` (31 total), spanning all
five canary rounds to date:

```
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,24,32,38,39,40,47,48,50,53,55,57,61
```

## Recommended next step

Run a small real canary round (e.g. `--max-cases 6`) against this version,
using the default exclusion list, and report back per-stage results
(Stage 1 pass rate, Stage 2a pass rate, Stage 2b call count and pass rate
for actionable decisions, Stage 3 reach rate, per-stage latency from
`summarize_canary_results.py`) before any larger round or the 64-case run.
