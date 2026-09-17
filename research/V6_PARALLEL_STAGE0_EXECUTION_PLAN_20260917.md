# V6 Parallel Stage0 Execution Plan — 2026-09-17

Purpose: stop serial idle time while preserving preregistration, PIT discipline, and visible progress for every lane.

## Isolation
- Formal R10 is untouched.
- 0B Industry remains on the proven checkpoint_v3 path and cache namespace.
- 0C Company/Fundamental, 0D Macro Audit, and 0E Event/Time Audit run as independent jobs with independent artifacts/caches.
- No layer model, manual layer weighting, Top-N policy, fixed holding period, or formal OOS is allowed here.
- 0F remains blocked until all required Stage0 layers independently pass their gates.

## Parallel lanes
1. 0B Industry: continue persistent checkpoint accumulation; never re-fetch SUCCESS/VERIFIED_NO_DATA units; preserve throttle-stop behavior.
2. 0C Company/Fundamental: build PIT/lineage inventory and builder/audit from existing repository data first; do not redownload existing 2016–2025 OHLCV/institutional/fundamental inputs.
3. 0D Macro: audit source/equivalence/coverage and publication-lag semantics; do not promote PROVISIONAL to PASS without evidence.
4. 0E Event/Time: build an auditable inventory of event/time sources; only reconstructible published_at/available_at fields may enter formal OOS. Non-reconstructible sources are Live/Shadow only.

## Mandatory visible progress contract
Every lane MUST emit a machine-readable progress JSON plus a human-readable progress summary artifact on every run. At minimum it must expose:
- lane
- status
- current_completed
- target_total (if finite)
- completion_pct (if meaningful)
- newly_completed_this_run
- remaining
- current_blocker
- last_successful_unit/date
- input/source hashes where applicable
- artifact name
- updated_at_utc

A run that only reruns unchanged work must report newly_completed_this_run=0 and MUST NOT be counted as research progress.

## Concurrency safety
- Every lane uses a unique cache namespace and artifact name.
- Jobs must not write the same cache directory or output path.
- Parallel jobs should compute and upload artifacts; repository writes are serialized and only used for code/contracts, not for per-run data accumulation.
- 0F cannot start from partial lane artifacts.
