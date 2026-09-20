# AlphaPilot V6.2 three-stage canary verdict — 2026-09-20

## Verdict

**STOP. Claude v2 is not ready for the formal 64-case audit.**

The contract-only suite passes, but two real Qwen2.5-7B canaries produced **0 finalized cases**. The first run exposed a runner integration defect. After that defect was fixed, Stage 1 passed for every case in the second run, but Stage 2 failed its contract for every case. Stage 3 therefore has not been validated with a real model output.

No hidden outcome or future-path data was read for tuning or selection. No 2025 data was opened. R10 and 0A–0F were not modified. The formal 64-case audit was not run.

## Evidence

| Run | Cases | Finalized | Contract failures | Failure point | Mean / max elapsed |
|---|---:|---:|---:|---|---:|
| `35482706294` | 11 | 0 | 11 | Stage 1 | 250.44s / 344.44s |
| `35485028854` | 8 | 0 | 8 | Stage 2 | 316.47s / 367.35s |

Run `35482706294` selected case IDs `0,1,2,3,4,6,10,13,40,47,53`. Its artifact is `10596618179`.

Run `35485028854` selected case IDs `8,9,11,12,21,22,50,61`. Its artifact is `10597496215`.

The second run's Stage 1 outputs were schema-valid for all eight cases. All eight failed Stage 2 after both the initial attempt and one retry. The runner now correctly exits nonzero when any case has a model contract failure, so the second workflow is red by design.

## Problems Claude must fix

### P0 — Stage 2 decision and actionability contracts are coupled in a way Qwen cannot satisfy

The model commonly selected `CANDIDATE` while returning:

- `entry.status = NOT_ACTIONABLE`
- null entry prices
- otherwise schema-valid JSON

The JSON Schema enforces shape and enums but not the conditional semantic rule that `CANDIDATE` or `HIGH_CONVICTION` requires an actionable entry plan. The retry prompt supplied the exact validation errors, but the second attempt was materially the same.

Required revision: make the invalid combination structurally impossible. Either:

1. split Stage 2 into decision-only and a separate actionability substage, each with a small schema; or
2. use decision-conditional schemas (`oneOf` or `if/then`) or select a second schema after the decision is known.

Do not silently synthesize prices in the finalizer. If the packet cannot ground an entry, the model must choose a non-actionable decision under the explicit contract.

### P0 — `_text_mentions_family()` produces false benchmark errors

The validator currently treats any generic token from a family ID as a family mention. Words such as `information` and `structure` make a valuation sentence appear to mention unrelated families. A sentence like “reasonable valuation” was consequently reported as a benchmark violation for `institutional_flow`, `mops_material_information`, `price_volume_structure`, `revenue`, and `valuation`.

Required revision:

- replace token-any matching with exact family IDs and explicit, family-specific aliases;
- do not use generic fragments such as `information`, `structure`, `flow`, or `price` alone;
- add negative regression tests proving a valuation sentence does not match the other four families.

### P0 — Stage 1 and Stage 2 disagree about valuation without a peer benchmark

Stage 1 permits an absolute valuation extreme to be classified as contradictory or supportive, while Stage 2 rejects qualitative words such as `cheap`, `overvalued`, and `reasonable valuation` whenever a peer benchmark is unavailable. The model can therefore correctly carry a Stage 1 valuation signal into Stage 2 and still be rejected.

Required revision: define one consistent rule. Either:

- permit explicitly labelled absolute-valuation observations while prohibiting peer-relative claims; or
- force unbenchmarked valuation to `NOT_INTERPRETABLE` in Stage 1 and prohibit it from primary, secondary, and counter evidence in Stage 2.

The validator and both prompts must implement the same rule.

### P1 — Evidence-role constraints are not structurally enforced

Stage 2 sometimes placed a `NEUTRAL` family in `primary_evidence_ids` or `secondary_evidence_ids`. It also sometimes omitted a contradictory valuation family from `counter_evidence_ids`.

Required revision:

- derive allowed evidence-ID lists deterministically from Stage 1;
- constrain or validate primary/secondary IDs to supportive or mixed evidence according to the documented rule;
- constrain counter IDs to contradictory or mixed evidence according to the documented rule;
- state exactly how `NEUTRAL` and `NOT_INTERPRETABLE` may be used (normally neither should support a decision).

### P1 — Retry is not behaviorally effective

Both Stage 2 attempts failed for every case, commonly with the same decision, entry status, and null prices. Passing a long list of validator errors back to the same model is not producing correction.

Required revision:

- reduce each retry to a compact machine-generated repair instruction;
- include only the invalid paths, received values, and permitted alternatives;
- test that the retry request differs meaningfully from the first request;
- cap retries explicitly and retain the correct nonzero workflow exit on exhaustion.

### P1 — Real-model Stage 3 remains untested

Because no case passed Stage 2, there is no evidence that the critic/finalizer path works against real Qwen output. Unit and mock tests are insufficient to claim canary readiness.

Required revision: after Stage 2 is corrected, run only a small mechanical canary. Do not proceed to the formal 64-case audit until at least one real case traverses all stages and the canary meets an explicit pass criterion.

### P1 — Latency is operationally unacceptable

The second run averaged 316.47 seconds per case and finalized none. At that rate, a 64-case run would be expensive and slow before considering retries.

Required revision: report per-stage latency and call counts, then set an explicit operational budget. Extra decomposition may improve reliability but must be evaluated against this budget.

### P2 — Model capability may be the binding constraint

Qwen2.5-7B obeyed the Stage 1 response schema after the runner sent the actual JSON Schema, but it did not satisfy the denser Stage 2 semantic contract even with retry feedback. If a redesigned Stage 2 still fails a minimal canary, stop adapting the contract to this model and test a stronger model for Stage 2/critic while keeping Stage 1 on Qwen2.5-7B if desired.

## Required acceptance tests before another canary

1. A `CANDIDATE` or `HIGH_CONVICTION` response with `NOT_ACTIONABLE` or null required entry fields must be rejected by the response schema or be impossible to request.
2. A `WATCH` or `REJECT` response must not carry an actionable entry or a non-null failure-exit plan unless the contract explicitly allows and tests it.
3. “Reasonable valuation” must match only the valuation family.
4. An unbenchmarked absolute-valuation statement must produce the same allowed/prohibited result in Stage 1, Stage 2, and the validator.
5. `NEUTRAL` and `NOT_INTERPRETABLE` evidence IDs must not appear as positive support.
6. Retry fixtures must show that each enumerated Stage 2 failure can be repaired on the second attempt.
7. Runner exit code must remain nonzero whenever any canary case ends in `MODEL_CONTRACT_FAILURE`.

## Fresh-case accounting

Do not describe the 19 cases used by the two runs above as fresh again. Combined with the 12 earlier canary cases, the consumed set is:

`{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,22,24,32,38,39,40,47,48,50,53,55,57,61}`

Any later semantic canary must exclude that set. Replaying one of these cases is acceptable only for a clearly labelled mechanical regression check, never as fresh evidence.

## Next decision

Claude must return a revised design and code addressing the P0 items first. Do not run a third fresh Qwen2.5-7B canary on the current architecture. Do not run the formal 64-case audit.
