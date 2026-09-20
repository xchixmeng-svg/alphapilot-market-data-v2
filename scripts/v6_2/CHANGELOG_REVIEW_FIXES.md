# Changelog: fixes for the external review

Legend: [FIXED] = code change made and covered by a new/updated test.
[ADDRESSED-DOC] = design/documentation change, not independently testable
in this offline sandbox (e.g. requires a real network call).

## P0 -- would break the workflow

1. [FIXED] `run_fresh_canary.py` and `summarize_canary_results.py` now
   delivered as complete, working files (previously described as
   out-of-scope wrappers). See `tests/test_run_fresh_canary.py` for
   coverage of the parts that can be tested offline.
2. [FIXED] `v6_2_canary.yml` now downloads the prepared artifact via
   `gh api .../artifacts/{id}/zip` before running anything.
3. [FIXED] `v6_2_canary.yml` now installs Ollama, starts the server,
   waits for readiness, and pulls the model before calling
   `run_fresh_canary.py`.
4. [FIXED] The 2025/data-integrity guard is fail-closed:
   `load_and_guard_prep_summary()` raises `GuardFailure` (exit code 1,
   nothing runs) if `PREP_SUMMARY.json` is missing, malformed, or does
   not explicitly declare `2025_opened: false`. Also checks decision_date
   <= 20241231 and rejects duplicate case_id in `load_packets()`. Covered
   by 9 tests in `test_run_fresh_canary.py`.
5. [FIXED] `v6_2_canary.yml`'s upload step now has `if: always()`.

## P0 -- orchestrator bugs

6. [FIXED] Retries now pass `previous_attempt` (raw output + validation
   errors) or `critic_feedback` to the model on every attempt after the
   first. Covered by `test_retry_second_attempt_payload_contains_first_
   attempt_errors` and `test_stage2_revision_payload_contains_critic_
   feedback_and_prior_errors`.
7. [FIXED] Default timeouts recalibrated: `per_call_timeout_seconds=180`
   (headroom above the reported ~149s real call), `per_case_timeout_
   seconds=1500` (headroom above the 8-call worst case).
8. [FIXED] Hard timeout via `ThreadPoolExecutor.result(timeout=...)`, plus
   a deadline recheck immediately after every call returns, before the
   result is accepted. Covered by `test_hard_per_call_timeout_is_
   enforced_even_if_call_never_returns` and `test_late_result_after_
   deadline_is_not_finalized`.
9. [FIXED] Every `validate_*` function now catches its own exceptions
   internally; `retry_orchestrator.py` adds a second layer plus an
   outermost per-case try/except in `run_batch()`. Covered by
   `test_malformed_field_types_do_not_crash_run_case` and
   `test_run_batch_survives_unexpected_exception_in_one_case`, plus the
   `test_stage2_output_not_a_dict_does_not_crash` /
   `test_stage2_malformed_evidence_id_field_types_do_not_crash` pair that
   directly reproduces the reported `TypeError: 'int' object is not
   iterable`.
10. [FIXED] `HistoryEntry` now has `applied_fixes` and `warnings` fields;
    the Stage2 retry loop uses `_stage2_validate_with_finalizer()` which
    returns the full `FinalizeResult` tuple. Covered by
    `test_history_retains_applied_fixes_and_warnings`.

## P1 -- validator/grounding

11. [FIXED] `validate_stage1_output()` now takes `expected_case_id` and
    strictly checks type-and-value equality. Covered by
    `test_stage1_wrong_case_id_is_rejected` and
    `test_stage1_case_id_wrong_type_is_rejected`.
12. [FIXED] `_ground_numeric_claims()` cross-checks every number in
    `exact_observations` against the same family's raw packet data (~1%
    tolerance). Covered by `test_stage1_fabricated_number_is_rejected_by_
    grounding_check`, `test_stage1_genuine_number_passes_grounding_check`,
    `test_stage1_grounding_rejects_cross_family_number_borrowing`. Free
    text with no extractable number is intentionally left unchecked here
    (mechanically unverifiable) and remains Stage 3's job.
13. [FIXED] `validate_stage3_output()` now checks `field` enum,
    non-empty-string `claim_text`, null-or-known-id `cited_evidence_id`,
    `problem_type` enum, non-empty `explanation`, individually. Covered
    by 5 new tests including the exact malformed-response reproduction
    from the review.
14. [FIXED] Actionable entry/failure-exit prices are now checked for
    plausibility against Stage1's `price_volume_structure` numbers
    (0.5x-2x bound). Covered by 2 new tests.
15. [ADDRESSED-DOC] `tests/test_schema_runtime_parity.py` (17 tests) now
    checks every hand-written required-key-set and enum in
    `deterministic_validators.py` against the corresponding JSON Schema
    file. This does not make the schema the actual single source of truth
    at runtime (the `jsonschema` package could not be installed in this
    offline sandbox to wire real schema validation) -- it prevents silent
    drift between the two. Recommend adding `pip install jsonschema` and
    real schema validation in your CI, which has network access.

## P1 -- prompt semantics

16. [FIXED] Stage1 prompt no longer says routine content should be marked
    STALE regardless of actual age; `timeliness` is now strictly about
    data recency, `relevance_to_hypothesis_space`/`limitations` carry the
    "routine, not a catalyst" judgment.
17. [FIXED] Related-party/intercompany loan filings removed from
    `ROUTINE_MOPS_KEYWORDS`; Stage1 prompt now requires amount/
    counterparty/rationale judgment for these, defaulting to
    NOT_INTERPRETABLE with a stated limitation when that detail is
    absent. Covered by `test_related_party_loan_is_not_auto_classified_
    as_routine` (EN and ZH title variants).

## P2 -- fresh selector

18. [FIXED] Renamed `BUCKET_HIGH_PRIOR_STRONG_EVIDENCE` /
    `BUCKET_LOW_PRIOR_STRONG_SUPPORTIVE_EVIDENCE` to
    `..._BROAD_COVERAGE_NO_DETECTED_CONFLICT`, with docstrings stating
    explicitly this is a sampling-stratification proxy, not an
    evidence_quality judgment (that judgment happens only in Stage 1/2).
19. [ADDRESSED-DOC] `PRIOR_HIGH_THRESHOLD`/`PRIOR_LOW_THRESHOLD` changed
    to 0.60/0.40 (round numbers dividing [0,1] into thirds) and
    documented as declared a-priori sampling strata, never a decision
    admission rule. Covered by
    `test_thresholds_are_declared_round_numbers_not_sample_specific`.
20. [FIXED] Replaced the tautological
    `has_negative and (has_positive or True)` with
    `_institutional_flow_direction_signal()`, an honest 4-way
    classification. Covered by
    `test_institutional_flow_signal_negative_only_distinct_from_mixed`.

## Additional regression tests required by the review, all present

- wrong Stage1 case_id rejected -- `test_stage1_wrong_case_id_is_rejected`
- Stage2 evidence-ID fields int/dict/string don't crash --
  `test_stage2_malformed_evidence_id_field_types_do_not_crash`
- validator exception doesn't abort batch --
  `test_run_batch_survives_unexpected_exception_in_one_case`
- retry 2nd payload contains 1st errors --
  `test_retry_second_attempt_payload_contains_first_attempt_errors`
- safe-finalizer fixes/warnings in history --
  `test_history_retains_applied_fixes_and_warnings`
- model call timeout -> not FINALIZED --
  `test_hard_per_call_timeout_is_enforced_even_if_call_never_returns`,
  `test_late_result_after_deadline_is_not_finalized`
- malformed Stage3 field/claim_text/cited_evidence_id rejected -- 5 tests
  in `test_three_stage_contract.py`
- fabricated Stage1 numeric observation rejected --
  `test_stage1_fabricated_number_is_rejected_by_grounding_check`
- missing PREP_SUMMARY.json -> hard fail --
  `test_missing_prep_summary_raises_guard_failure`
- canary failure -> artifact still uploaded -- `if: always()` in
  `v6_2_canary.yml` (workflow-level, not unit-testable offline)
- workflow downloads artifact, starts qwen2.5:7b -- `v6_2_canary.yml`
  steps (workflow-level, not unit-testable offline; never run for real
  in this sandbox -- see README's "What I still could NOT verify")
- runner defaults exclude the 12 prior canary case_ids --
  `test_default_exclude_case_ids_matches_prior_canary_rounds`,
  `test_select_fresh_canary_defaults_to_excluding_prior_case_ids`, and
  the workflow's `exclude_case_ids` input default
