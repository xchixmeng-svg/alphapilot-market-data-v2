#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the documented one-time R0.5 +80% Runner classification.

This is a source repair, not a historical-result patch. No trade/NAV fixture is
read. RUNNER is classified once at >=80% into MEGA or TARGET; those states are
terminal and cannot be reclassified later by a changing amount ratio.
"""
from pathlib import Path

p = Path(__file__).resolve().parent / "backtest_r10_stress.py"
s = p.read_text(encoding="utf-8")
old = '''    if p.mode in {"RUNNER", "MEGA", "TARGET"} and ret >= 0.80 - eps:\n        if ar >= 1.20 - eps: p.mode = "MEGA"\n        elif p.mode != "MEGA": p.mode = "TARGET"\n'''
new = '''    # +80% is a ONE-TIME classification point. Once classified as MEGA or\n    # TARGET, later volume changes must not switch the state again.\n    if p.mode == "RUNNER" and ret >= 0.80 - eps:\n        p.mode = "MEGA" if ar >= 1.20 - eps else "TARGET"\n'''
if new in s:
    print("R0.5 runner state already patched")
elif old in s:
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print("PATCHED", p)
else:
    raise SystemExit("R0.5 runner-state anchor not found; refuse silent patch")
