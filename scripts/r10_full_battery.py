#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended AlphaPilot R10-MAX historical stress battery.

This module intentionally reuses the locked R10 stress engine. It adds
independent historical regimes while preserving the rule freeze. Results are
labelled PARTIAL whenever R0.5 cannot be reconstructed without the required
TWSE daily institutional history.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from scripts import backtest_r10_stress as bt

EXTRA = {
    "euro2011": {
        "label": "2011 Euro-area / US downgrade selloff",
        "warmup_start": "2010-01-01",
        "eval_start": "2011-01-03",
        "eval_end": "2011-12-30",
        "years": [2010, 2011],
        "r05": False,
        "mode": "PARTIAL_R10_R7_ONLY",
        "reason": "TWSE stock-level daily institutional T86 starts later; R0.5 is disabled rather than fabricated.",
    },
    "china2015": {
        "label": "2015 China devaluation / global equity shock",
        "warmup_start": "2014-01-01",
        "eval_start": "2015-01-05",
        "eval_end": "2015-12-31",
        "years": [2014, 2015],
        "r05": True,
        "mode": "FULL_R10_RECONSTRUCTION",
        "reason": "R7 + R0.5 reconstructed causally from official OHLCV and available daily institutional history.",
    },
    "tradewar2018": {
        "label": "2018 US-China trade-war / Q4 selloff",
        "warmup_start": "2017-01-01",
        "eval_start": "2018-01-02",
        "eval_end": "2018-12-28",
        "years": [2017, 2018],
        "r05": True,
        "mode": "FULL_R10_RECONSTRUCTION",
        "reason": "R7 + R0.5 reconstructed causally from official OHLCV and daily institutional history.",
    },
    "crash2024": {
        "label": "2024 AI bull + 2024-08-05 crash",
        "warmup_start": "2023-01-01",
        "eval_start": "2024-01-02",
        "eval_end": "2024-12-31",
        "years": [2023, 2024],
        "r05": True,
        "mode": "FULL_R10_IN_SAMPLE_STRESS",
        "reason": "Historical stress segment overlaps the locked 2021-2025 research sample; useful for event diagnostics, not OOS proof.",
    },
    "tariff2025": {
        "label": "2025 tariff crash + recovery",
        "warmup_start": "2024-01-01",
        "eval_start": "2025-01-02",
        "eval_end": "2025-12-31",
        "years": [2024, 2025],
        "r05": True,
        "mode": "FULL_R10_IN_SAMPLE_STRESS",
        "reason": "Historical stress segment overlaps the locked 2021-2025 research sample; useful for event diagnostics, not OOS proof.",
    },
}

bt.SCENARIOS.update(EXTRA)
ALL = [
    "validation2021_2025",
    "gfc2008",
    "euro2011",
    "china2015",
    "tradewar2018",
    "covid2020",
    "bear2022",
    "crash2024",
    "tariff2025",
]


def validation_grade(result: dict) -> dict:
    """Conservative gate: reconstructed engine must first match locked sample."""
    if result.get("status") != "PASS":
        return {"grade": "FAIL", "reason": "validation run failed"}
    d = result.get("validation_delta") or {}
    checks = {
        "end_nav_pct_abs_le_5pct": abs(float(d.get("end_nav_pct", 99))) <= 0.05,
        "cagr_abs_le_2pp": abs(float(d.get("cagr", 99))) <= 0.02,
        "max_dd_abs_le_2pp": abs(float(d.get("max_dd", 99))) <= 0.02,
        "trades_abs_le_10pct": abs(float(d.get("trades", 999))) <= max(10, round(bt.LOCKED_BENCHMARK["completed_trades"] * 0.10)),
    }
    ok = all(checks.values())
    return {"grade": "PASS" if ok else "FAIL", "checks": checks, "delta": d}


def run_one(name: str) -> dict:
    if name not in bt.SCENARIOS:
        raise SystemExit(f"unknown scenario {name}")
    result = bt.simulate(name, bt.SCENARIOS[name])
    if name == "validation2021_2025":
        result["regression_gate"] = validation_grade(result)
        out = bt.OUT_ROOT / "latest" / name / "summary.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=ALL)
    args = ap.parse_args()
    try:
        r = run_one(args.scenario)
        print(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        payload = {
            "status": "ERROR",
            "scenario": args.scenario,
            "engine_version": bt.VERSION,
            "generated_at": datetime.now().astimezone().isoformat(),
            "error": str(exc),
        }
        out = bt.OUT_ROOT / "latest" / args.scenario
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
