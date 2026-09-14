#!/usr/bin/env python3
"""Run the existing low-base persistence research with second-recommendation variants."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "scripts" / "backtest_r10_lowbase_persistence.py"
spec = importlib.util.spec_from_file_location("lowbase", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.VARIANTS = [
    {"name": "LB_P2_5", "min_count": 2, "window": 5, "consecutive": False},
    {"name": "LB_P2_10", "min_count": 2, "window": 10, "consecutive": False},
    {"name": "LB_P3_5", "min_count": 3, "window": 5, "consecutive": False},
    {"name": "LB_P3_10", "min_count": 3, "window": 10, "consecutive": False},
]
mod.RUN_ROOT = ROOT / "lowbase_p2_results"
mod.BASELINE_DIR = mod.RUN_ROOT / "baseline"
mod.main()
