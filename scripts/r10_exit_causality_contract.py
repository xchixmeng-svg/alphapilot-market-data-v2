#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic no-look-ahead contracts for R10 exits. No historical answer data."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import backtest_r10_stress as bt

ROOT = Path(__file__).resolve().parent.parent

# R0.5 exit decision must be a T-close state machine. Intraday OHLC fields are
# execution-day data and may not decide the previous day's sell signal.
src = inspect.getsource(bt.r05_exit_reason)
assert "r.aclose" in src, src
for forbidden in ("r.low", "r.open", "r.high"):
    assert forbidden not in src, f"R0.5 exit decision reads intraday field: {forbidden}"

# Boundary check: 90.01% of entry has NOT triggered a -10% close stop.
p = bt.Position("R05", "1234", "TEST", 1000, 20260101, 100.0, 100.0, 100000.0, 100.0)
r = SimpleNamespace(aclose=90.01, amount_ratio=1.0)
assert bt.r05_exit_reason(p, r) is None, "R0.5 hard stop fired before T-close <= -10%"

# Exactly at the close threshold, T may decide to exit.
p = bt.Position("R05", "1234", "TEST", 1000, 20260101, 100.0, 100.0, 100000.0, 100.0)
r = SimpleNamespace(aclose=90.0, amount_ratio=1.0)
assert bt.r05_exit_reason(p, r) == "HARD"

# The clean engine must schedule that T decision for the next trading date.
true_source = (ROOT / "scripts" / "r10_true_validation.py").read_text(encoding="utf-8")
assert 'exdate = next_date[di]' in true_source
assert 'bt.SellOrder(di, exdate' in true_source
assert true_source.index('pending_sells.pop(di') < true_source.index('pending_buys.pop(di')

print("R10 EXIT CAUSALITY CONTRACT: PASS")
