# Stage9 Context-Conditioned Edge Discovery

Purpose: test whether factor edge depends on contemporaneous market structure, without tuning on 2019, 2020, or 2021-2025.

Discovery: 2016-2018 only. Context thresholds are rolling past-only percentiles. 2019 is OOS1. 2020 is OOS2. 2021-2025 untouched. Formal R10 untouched. Cached 2015-2020 reference artifact only; no market refetch.

This stage is diagnostic. It does not promote a live strategy by itself. Any edge must first keep its discovery direction in 2019 and then again in 2020 before it is eligible for transaction-level strategy construction.
