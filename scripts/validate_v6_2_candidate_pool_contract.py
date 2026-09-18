#!/usr/bin/env python3
"""Static contract checks for V6.2 candidate-pool outputs.

This is intentionally model-agnostic. It enforces the product contract:
candidate admission cannot be based on a forced daily quota.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {
    "date",
    "code",
    "candidate_flag",
    "candidate_evidence_score",
    "candidate_confidence",
}


def validate(path: Path) -> dict:
    x = pd.read_csv(path, dtype={"code": str})
    missing = REQUIRED - set(x.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")
    x["candidate_flag"] = x["candidate_flag"].astype(bool)

    counts = x.groupby("date")["candidate_flag"].sum()
    if counts.empty:
        raise RuntimeError("empty candidate output")

    # A constant positive count across every date is treated as suspicious forced-quota behavior.
    # Zero-candidate dates are allowed and explicitly valid.
    positive = counts[counts > 0]
    constant_positive_quota = (
        len(counts) >= 20
        and len(positive) == len(counts)
        and counts.nunique() == 1
    )
    if constant_positive_quota:
        raise RuntimeError(
            f"forced-quota signature detected: exactly {int(counts.iloc[0])} candidates every date"
        )

    selected = x[x["candidate_flag"]].copy()
    if "daily_rank" in selected.columns and "admission_reason" in selected.columns:
        bad = selected["admission_reason"].astype(str).str.contains(
            r"^top\s*\d+$|rank[-_ ]?only|daily[-_ ]?rank",
            case=False,
            regex=True,
            na=False,
        )
        if bad.any():
            raise RuntimeError("rank-only candidate admission detected")

    return {
        "dates": int(counts.size),
        "candidate_days": int((counts > 0).sum()),
        "no_candidate_days": int((counts == 0).sum()),
        "total_candidates": int(counts.sum()),
        "min_candidates_per_day": int(counts.min()),
        "median_candidates_per_day": float(counts.median()),
        "max_candidates_per_day": int(counts.max()),
        "distinct_daily_candidate_counts": int(counts.nunique()),
        "forced_quota_signature": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_csv")
    ns = ap.parse_args()
    print(validate(Path(ns.candidate_csv)))


if __name__ == "__main__":
    main()
