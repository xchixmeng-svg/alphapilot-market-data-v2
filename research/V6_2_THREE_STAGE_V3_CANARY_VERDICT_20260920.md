# AlphaPilot V6.2 three-stage v3 canary verdict — 2026-09-20

## Verdict

**FAIL. Do not run the formal 64-case audit and do not run another fresh Qwen2.5-7B canary on this implementation.**

Claude v3's Stage 2a/2b split removed the old `CANDIDATE + NOT_ACTIONABLE + null price` shape contradiction, but the real canary still finalized **0 of 6** cases. Every case failed Stage 2a after both attempts. Stage 2b and Stage 3 were never reached.

No hidden outcomes or hidden future paths were opened. The workflow selectively extracted only `PREP_SUMMARY.json` and `cases_shard_*.jsonl`. No 2025 data was opened. R10 and 0A–0F were not modified. The formal 64-case audit was not run.

## Reproducibility

- Branch: `research-v6-2-three-stage-canary-20260920`
- Tested source commit: `5879e5e61e83c98f4410e52dbf1174b8dc75fb6f`
- Contract workflow: `35492994878` — PASS, 114/114 tests
- Real-model canary: `35493091582` — FAIL
- Canary artifact: `10599969736`
- Prepared artifact: `10582014726`
- Model: `qwen2.5:7b`, Ollama structured JSON Schema, temperature 0

## Result

| Metric | Result |
|---|---:|
| Cases actually executed | 6 |
| Finalized | 0 |
| Model contract failures | 6 |
| Stage 1 attempts | 7 |
| Stage 1 timeout attempts | 1 |
| Stage 2a attempts | 12 |
| Stage 2a invalid attempts | 12/12 |
| Stage 2b reached | 0 |
| Stage 3 reached | 0 |
| Mean case elapsed | 609.93s |
| Max case elapsed | 843.95s |

Actually executed case IDs: `17,18,19,27,28,31`.

The selector output incorrectly retained eight pre-truncation IDs (`17,18,19,27,28,31,43,56`) even though `--max-cases=6`; only IDs present in `canary_case_results.json` were executed. Therefore `43` and `56` remain unused by this semantic canary.

## Failure map

| Case | Stage 2a final failure | Elapsed |
|---:|---|---:|
| 17 | `WATCH` with no primary supportive evidence | 843.95s |
| 18 | `WATCH` with no primary supportive evidence | 507.07s |
| 19 | `WATCH` with no primary supportive evidence | 552.79s |
| 27 | `WATCH` with no primary supportive evidence | 562.12s |
| 28 | benchmark/value-judgment validator errors, including false cross-family attribution | 581.93s |
| 31 | NEUTRAL revenue used as primary supportive evidence | 611.72s |

## Problems Claude must fix

### P0 — Stage 1 now creates the evidence-collapse input to Stage 2

Stage 1 was schema-valid but semantically unreliable:

- it repeatedly labelled `price_volume_structure` as `NEUTRAL`, then Stage 2 described the same observations as supportive buying interest;
- case 28 had positive foreign-flow observations across the reported windows but Stage 1 labelled institutional flow `NEUTRAL`;
- cases 19 and 27 contained conflicting institutional-flow horizons, which the prompt's own example says should be `MIXED`, but Stage 1 returned `NEUTRAL`;
- boilerplate `routine/periodic filing` limitations were attached to institutional flow and price/volume families;
- case 28's material fund-lending filing was called routine even though the prompt explicitly forbids defaulting fund-transfer/loan filings to routine;
- case 31's MOPS observation received a valuation benchmark limitation.

The schema validates syntax, not whether a limitation belongs to the family or whether the direction follows from the observations. Stage 2 cannot make a coherent decision when Stage 1 marks nearly every available family NEUTRAL/NOT_INTERPRETABLE.

Required revision:

1. Replace free-text limitations with structured `limitation_codes`, for example `NO_VALUATION_BENCHMARK`, `SINGLE_PERIOD_ONLY`, `ROUTINE_PERIODIC_FILING`, `MIXED_SUBSIGNALS`, `INSUFFICIENT_MOPS_CONTEXT`, `NONE`.
2. Enforce which codes are legal for each evidence family.
3. Add deterministic consistency checks for clearly mixed institutional-flow signs and for the specific fund-lending MOPS handling rule already stated in the prompt.
4. Add real packet fixtures reproducing cases 19, 27, 28, and 31. Synthetic happy-path fixtures are insufficient.

### P0 — `benchmark_available` is ambiguous outside valuation

The schema requires `benchmark_available` for every family. Qwen returned false for price/volume and revenue, then the validator applied value-judgment checks to those families. This generated errors such as a sentence containing revenue plus “cheap” being classified as a value judgment about revenue.

The validator also associates a value word anywhere in an entire field with every family alias appearing anywhere in that field. That is the same class of cross-clause false attribution as the prior family-token bug.

Required revision:

- replace the boolean with `benchmark_status = AVAILABLE | UNAVAILABLE | NOT_APPLICABLE`, enforcing `NOT_APPLICABLE` outside valuation; or make the field valuation-only through a conditional schema;
- apply cheap/expensive/overvalued checks only to valuation claims;
- require structured claim-to-evidence citations, or evaluate clause-local associations rather than scanning the entire thesis field;
- reproduce case 28 as a regression test.

### P0 — Retry repairs fields but does not repair decision semantics

For cases 17, 18, 19, and 27, attempt 1 used NEUTRAL or CONTRADICTORY evidence as positive support. Attempt 2 removed the invalid IDs but kept `WATCH`, leaving no primary supportive evidence. Under the stated contract, WATCH requires a credible supportive thesis; with no SUPPORTIVE/MIXED family the consistent result is REJECT.

Required revision:

- include deterministic payload fields `allowed_positive_evidence_ids`, `required_counter_evidence_ids`, and `forbidden_evidence_ids`;
- when repair removes the last positive-support ID, explicitly require the model to re-evaluate `decision`, `hypothesis`, and thesis text, not merely edit the arrays;
- add a real failure fixture where the correct repair is a decision transition to REJECT;
- do not weaken the validator by allowing WATCH with zero supportive evidence.

### P0 — Qwen2.5-7B did not satisfy Stage 2a even after the split

All 12 Stage 2a attempts were invalid. The contingency stated in Claude's own v3 README has now triggered: stop adapting the contract around Qwen2.5-7B and test a stronger reasoning model for Stage 2. Because Stage 1 also produced substantial semantic errors, evaluate whether Stage 1 must also move or be substantially structured/deterministic.

Do not run another fresh case set merely after prompt wording changes.

### P1 — Latency became worse, not better

The prior structured-output canary averaged 316.47 seconds per case. V3 averaged 609.93 seconds per case, about 93% slower, while still finalizing zero cases. The six-case workflow took roughly 62 minutes.

Ollama logs showed generation near 5 tokens/second, prompts around 3,000 tokens, and calls reaching the 300-second HTTP limit. A 64-case run is operationally unacceptable.

Required revision:

- shorten prompts and response schemas materially;
- set output-token limits if supported;
- report prompt tokens, generated tokens, and latency per stage;
- establish a canary latency budget before any larger run.

### P1 — Stage 2b and Stage 3 remain unverified

No real case reached either stage. The architectural split has therefore not been validated end to end against a real model.

### P1 — Reporting bugs

1. `canary_selector_result.json` records the full eight-case selector result instead of the six cases actually executed after `--max-cases` truncation. Persist both `eligible_selected_case_ids` and `executed_case_ids`, or overwrite `selected_case_ids` with the executed set.
2. The summary reported p95 as 611.7s even though six values included an 843.95s maximum. With six observations, nearest-rank p95 should be the maximum; use a documented quantile definition and test it.

## Fresh-case accounting

The six newly consumed semantic-canary cases are:

`{17,18,19,27,28,31}`

Combined with the earlier 31 consumed cases, future fresh semantic canaries must exclude:

`{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,24,27,28,31,32,38,39,40,47,48,50,53,55,57,61}`

Cases `43` and `56` appeared only in the selector's pre-truncation list and were not executed; they are not consumed by this run.

## Next decision

Stop here. Claude should return a revised Stage 1 contract, clause-local/valuation-only benchmark validation, semantic retry behavior, and a model-routing proposal. Do not run the formal 64-case audit. Do not run another fresh Qwen2.5-7B canary on v3.
