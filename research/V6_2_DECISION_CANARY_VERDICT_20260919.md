# AlphaPilot V6.2 Decision Contract Canary Verdict

Date: 2026-09-19

## Verdict

**FAIL — do not merge this decision runtime and do not run a new 64-case replay.**

The original reflexive-REJECT collapse disappeared, but Qwen 2.5 7B replaced it
with a reflexive `CANDIDATE + STRONG` collapse. Contract adherence also remained
unacceptable after up to three repair attempts.

This verdict concerns the AI decision layer only. It does not change R10,
0A–0F, the numerical engine, the prepared artifact, or any outcome definition.

## Runs

| Run | Design | Valid | Errors | Valid decision distribution |
|---|---|---:|---:|---|
| 35456479322 | Three case-gap packets plus five high-prior packets | 6/8 | 2 | 6 CANDIDATE, all STRONG |
| 35458136917 | Two prior contract failures plus four lowest-prior controls | 5/6 | 1 | 5 CANDIDATE, all STRONG |
| 35459949583 | Same six cases after balanced-evidence prompt and multilingual limitation guard | 4/6 | 2 | 4 CANDIDATE, all STRONG |

Across the 12 unique canary cases, every case that produced a contract-valid
decision at least once was classified `CANDIDATE + STRONG`. Case 15 never
produced a valid response: it repeatedly combined a non-actionable decision
with an actionable entry or failure exit.

The four outcome-blind low-prior controls had `p_hit10_h120` from approximately
0.495 to 0.545. Every one was classified `CANDIDATE + STRONG` in the first
control run in which it returned a valid response. This is evidence of reverse
decision collapse, not healthy selectivity.

## Evidence-direction failures

These are packet-versus-response checks and do not use hidden outcomes.

- Case 14 / 8027: packet PE was about 215.37, revenue YoY was about -19.84%,
  and the routine quarterly-report filing was about 2,038 hours old. The model
  nevertheless described valuation as reasonable and short-term performance as
  improving, then returned `CANDIDATE + STRONG`.
- Case 16 / 2109: foreign-flow 5/20-session means were negative and revenue YoY
  was about -8.25%. A subsidiary loan disclosure was promoted into strong
  positive financial-support evidence, producing `CANDIDATE + STRONG`.
- Case 24 / 1604: one-month revenue growth was treated as durable growth and raw
  valuation multiples were called attractive without a supplied benchmark.
  The response claimed its bear thesis was represented by counter-evidence while
  `counter_evidence_ids` was empty.
- Case 32 / 4904: mixed institutional flow, negative revenue MoM and raw PE/PB
  values were summarized as strongly supportive evidence without a supplied
  valuation benchmark.

## Contract findings

- Synthetic contract suite: 23/23 PASS.
- Existing prepared packets: 64/64 structurally compatible in read-only tests.
- Runtime model compliance: FAIL. Repeated failures involved decision/entry/
  failure-exit inconsistency and empty required narratives.
- Permanent system limitations are now structurally separated and multilingual
  mentions outside `system_limitations` are rejected. This fixed the original
  bookkeeping ambiguity but did not make the 7B model reason reliably about the
  direction or sufficiency of available evidence.

## Engineering decision

Do not continue tuning this prompt on the same canary cases; that would create
sample-specific prompt overfitting. The next experiment must change the model or
the decision architecture, then use fresh outcome-blind canary cases.

Acceptable next routes are:

1. use a materially stronger reasoning model for the qualitative decision and
   keep the current deterministic validator; or
2. split the task into evidence-direction extraction, thesis/decision judgment,
   and a separate critic pass, with deterministic cross-pass consistency gates.

No full replay should be authorized until a fresh balanced canary demonstrates
both non-collapsed decisions and full contract-valid completion.
