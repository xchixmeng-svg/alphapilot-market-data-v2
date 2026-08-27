#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic tail-risk diagnostics for locked R10-MAX.

These are not historical backtests. They answer what the Portfolio Layer can
and cannot protect against when T+1 exits are delayed by gaps or limit-downs.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stress_results" / "latest" / "synthetic_tail"
OUT.mkdir(parents=True, exist_ok=True)

EXPOSURES = [0.20, 0.50, 0.75, 0.95]
LIMIT_DOWN_DAYS = [1, 2, 3, 4, 5]
SELL_SLIPPAGES = [0.005, 0.01, 0.02, 0.03]
GAPS = [-0.05, -0.10, -0.15, -0.20]
SELL_FEE = 0.000855
SELL_TAX = 0.003

rows = []
for exposure in EXPOSURES:
    for n in LIMIT_DOWN_DAYS:
        sleeve_factor = 0.9 ** n
        nav_factor = (1.0 - exposure) + exposure * sleeve_factor
        rows.append({
            "test": "consecutive_limit_down_unfillable",
            "exposure": exposure,
            "days": n,
            "shock": -0.10,
            "sell_slippage": None,
            "portfolio_loss": nav_factor - 1.0,
            "note": "Cash unchanged; stock sleeve compounds -10% per day; assumes exit cannot be filled during sequence.",
        })

for exposure in EXPOSURES:
    for gap in GAPS:
        for slip in SELL_SLIPPAGES:
            # Stock sleeve gaps first, then is sold at another adverse slippage;
            # fees/tax reduce proceeds. Cash sleeve is unaffected.
            stock_after = exposure * (1.0 + gap) * (1.0 - slip) * (1.0 - SELL_FEE - SELL_TAX)
            nav_factor = (1.0 - exposure) + stock_after
            rows.append({
                "test": "overnight_gap_then_sell",
                "exposure": exposure,
                "days": 1,
                "shock": gap,
                "sell_slippage": slip,
                "portfolio_loss": nav_factor - 1.0,
                "note": "Gap occurs before T+1 exit; sell price then suffers adverse slippage plus fee/tax.",
            })

# One 25%-NAV name shock: isolates concentration-cap protection.
for shock in [-0.10, -0.20, -0.30, -0.40, -0.50]:
    rows.append({
        "test": "single_name_25pct_cap",
        "exposure": 0.25,
        "days": 1,
        "shock": shock,
        "sell_slippage": None,
        "portfolio_loss": 0.25 * shock,
        "note": "Isolated single-name mark-to-market shock before any exit costs.",
    })

csv_path = OUT / "tail_matrix.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

critical = {
    "95pct_exposure_limit_down": {
        str(n): (1 - 0.95 + 0.95 * (0.9 ** n)) - 1 for n in LIMIT_DOWN_DAYS
    },
    "single_25pct_name_down_30pct": 0.25 * -0.30,
    "interpretation": [
        "DD Guard reduces future risk but cannot erase an overnight gap already suffered.",
        "A market with repeated unfillable limit-down sessions can push drawdown far beyond the historical -14% force-reduce trigger before an exit is executable.",
        "This matrix is deterministic stress arithmetic, not a forecast or historical performance claim.",
    ],
}
summary = {
    "status": "PASS",
    "type": "SYNTHETIC_TAIL_DIAGNOSTIC",
    "generated_at": datetime.now().astimezone().isoformat(),
    "locked_constraints_tested": {
        "normal_max_exposure": 0.95,
        "single_name_cap": 0.25,
        "t_plus_1_exit": True,
        "base_sell_slippage": 0.005,
    },
    "critical_cases": critical,
    "rows": len(rows),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
