# V6.2 Data Preservation Contract

Status: HARD DATA-GOVERNANCE CONTRACT
Date: 2026-09-19

## Problem
GitHub Actions artifacts are temporary workflow outputs and must never be the only copy of research-critical market data.

## Canonical preservation rule
Every research-critical dataset must have:
1. an immutable archived byte snapshot outside the temporary Actions-artifact lifecycle;
2. a repository-tracked manifest containing artifact/source identity, SHA256, lineage, scope, and seal status;
3. deterministic code or documented provenance sufficient to rebuild derived datasets from preserved canonical inputs;
4. at least one integrity check before every formal training/replay.

## Storage tiers

### Tier A — Canonical preserved inputs
Must be permanently snapshotted:
- frozen 0F decision/feature panel;
- frozen 0E historical material-information source;
- admitted TWSE institutional PIT;
- admitted TPEx institutional PIT;
- supplemental-admission manifest;
- admitted Evidence Bundle V1;
- later admitted evidence bundles that contain new raw evidence not reproducible from already-preserved Tier A inputs.

### Tier B — Reproducible derived outputs
Examples:
- numerical-model matrices;
- calibration tables;
- AI reasoning benchmark outputs;
- historical replay results.

These should have hashes and lineage, but they may be regenerated if all Tier A data + code commit are preserved. Scientific milestone outputs should still be snapshotted at freeze points.

## Actions artifacts
Actions artifacts are cache/transport only.
No formal model, replay, or audit may depend on an artifact ID without a canonical-preservation record.

## Integrity
Every preserved snapshot must record:
- original artifact ID / source;
- byte size;
- SHA256 of the downloaded archive;
- chunk SHA256 if split;
- repository commit used to create/archive it;
- creation date;
- seal scope (e.g. 2020-2024 development, 2025 sealed).

A hash mismatch is a hard failure.

## No silent replacement
Canonical snapshots are immutable. A corrected dataset receives a new version/snapshot and the old one is marked revoked/superseded; it is never silently overwritten.

## 2025 / 2026
Preservation must not violate research seals:
- 2025 content remains sealed until formal authorization;
- 2026 remains live-only under current V6.2 research rules.
