#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANES = ROOT / "clean_results" / "lanes"
OUT = ROOT / "clean_results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted(LANES.glob("*.json"))
    if not files:
        raise RuntimeError("no lane result files found")

    all_rows = []
    lane_summaries = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        lane_summaries.append(
            {
                "file": path.name,
                "family": obj.get("family"),
                "shard": obj.get("shard"),
                "tested": obj.get("tested", 0),
                "qualified": obj.get("qualified", 0),
                "best": obj.get("best"),
            }
        )
        all_rows.extend(obj.get("top", []))

    all_rows.sort(
        key=lambda x: (bool(x.get("qualified")), float(x.get("cagr", -999)), -abs(float(x.get("max_dd", -999)))),
        reverse=True,
    )
    qualified = [x for x in all_rows if x.get("qualified")]
    leaderboard = {
        "status": "INTERMEDIATE_ONLY",
        "tested_total": sum(int(x.get("tested", 0)) for x in lane_summaries),
        "qualified_total": sum(int(x.get("qualified", 0)) for x in lane_summaries),
        "lane_count": len(lane_summaries),
        "best_intermediate": qualified[0] if qualified else (all_rows[0] if all_rows else None),
        "top_20": (qualified if qualified else all_rows)[:20],
        "lanes": lane_summaries,
        "note": "Intermediate full-sample leaderboard only. Not a validated winner. Final acceptance still requires untouched holdout, expanding walk-forward/OOS, neighboring-parameter stability, year-by-year robustness, adverse execution/fill stress, and contract audits.",
    }
    (OUT / "parallel_leaderboard.json").write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": leaderboard["status"],
        "tested_total": leaderboard["tested_total"],
        "qualified_total": leaderboard["qualified_total"],
        "best": None if leaderboard["best_intermediate"] is None else {
            "family": leaderboard["best_intermediate"].get("family"),
            "cagr": leaderboard["best_intermediate"].get("cagr"),
            "max_dd": leaderboard["best_intermediate"].get("max_dd"),
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
