# V6 Parallel Stage0 Progress Dashboard

This file defines the visible dashboard fields. Runtime values are produced as workflow artifacts; per-run results are not committed here to avoid parallel write conflicts.

| Lane | Status | Completed / Target | % | New this run | Remaining | Blocker | Last update |
|---|---|---:|---:|---:|---:|---|---|
| 0A Market | PROVISIONAL | 2451 rows | n/a | 0 | audit gates | source/equivalence/corp-action audits | runtime artifact |
| 0B Industry | BUILDING | runtime | runtime | runtime | runtime | runtime | runtime artifact |
| 0C Company/Fundamental | BUILDING | runtime | runtime | runtime | runtime | runtime | runtime artifact |
| 0D Macro | PROVISIONAL | 2793 rows | n/a | runtime | audit gates | equivalence/source expansion | runtime artifact |
| 0E Event/Time | BUILDING | runtime | runtime | runtime | runtime | PIT source inventory/audit | runtime artifact |
| 0F Final Assembly | NOT_STARTED | 0 | 0% | 0 | blocked on 0A–0E | required layers not frozen | n/a |

Rule: only effective new completed units or newly passed audits count as progress. Reruns, empty commits, and repeated unchanged outputs do not.
