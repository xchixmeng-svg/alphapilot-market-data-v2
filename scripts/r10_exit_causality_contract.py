#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic no-look-ahead contracts for R10 exits. No historical answer data.

HARD USER/STRATEGY CONTRACT:
- ALL exits, including every R0.5 HARD/TRAIL/TARGET/MEGA/TIME exit, are decided
  with information available by T close only.
- ALL such sell orders execute on the next trading day (N+1 / T+1) from that
  day's OPEN using the documented adverse-sell model.
- There is NO R0.5 same-day intraday-stop exception in the active R10 strategy.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import backtest_r10_stress as bt

ROOT = Path(__file__).resolve().parent.parent

# R0.5 exit decision must be a T-close state machine. Intraday OHLC fields are
# forbidden from deciding the sell signal; this applies to HARD, TRAIL,
# TARGET, MEGA and TIME exits without exception.
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

# +80% classification is one-time. TARGET/MEGA are terminal states; a later
# amount-ratio change cannot silently switch TARGET -> MEGA or MEGA -> TARGET.
p = bt.Position("R05", "1234", "TEST", 1000, 20260101, 100.0, 100.0, 100000.0, 180.0, "TARGET", 20)
r = SimpleNamespace(aclose=185.0, amount_ratio=2.5)
reason = bt.r05_exit_reason(p, r)
assert p.mode == "TARGET", f"TARGET was illegally reclassified: {p.mode}"
assert reason is None, f"unexpected TARGET exit in terminal-state test: {reason}"

p = bt.Position("R05", "1234", "TEST", 1000, 20260101, 100.0, 100.0, 100000.0, 190.0, "MEGA", 20)
r = SimpleNamespace(aclose=185.0, amount_ratio=0.5)
reason = bt.r05_exit_reason(p, r)
assert p.mode == "MEGA", f"MEGA was illegally reclassified: {p.mode}"
assert reason is None, f"unexpected MEGA exit in terminal-state test: {reason}"

# A RUNNER crossing +80% is classified exactly once by the T-close amount ratio.
p = bt.Position("R05", "1234", "TEST", 1000, 20260101, 100.0, 100.0, 100000.0, 170.0, "RUNNER", 20)
r = SimpleNamespace(aclose=181.0, amount_ratio=1.1)
assert bt.r05_exit_reason(p, r) is None
assert p.mode == "TARGET", f"RUNNER should classify to TARGET, got {p.mode}"

p = bt.Position("R05", "1234", "TEST", 1000, 20260101, 100.0, 100.0, 100000.0, 170.0, "RUNNER", 20)
r = SimpleNamespace(aclose=181.0, amount_ratio=1.3)
assert bt.r05_exit_reason(p, r) is None
assert p.mode == "MEGA", f"RUNNER should classify to MEGA, got {p.mode}"

# The clean engine must schedule every T decision for the next trading date and
# execute every sell from that N+1 OPEN. No strategy-specific intraday sell path
# is permitted.
true_source = (ROOT / "scripts" / "r10_true_validation.py").read_text(encoding="utf-8")
assert 'exdate = next_date[di]' in true_source
assert 'bt.SellOrder(di, exdate' in true_source
assert 'px = bt.legal_sell_price(float(r.open))' in true_source
assert true_source.index('pending_sells.pop(di') < true_source.index('pending_buys.pop(di')

# Guard against accidentally adding a second R0.5 intraday/same-day execution
# path to the active engine.
for forbidden in (
    'r.low) <=',
    'r.low <',
    'intraday_stop',
    'same_day_stop',
    'R05_INTRADAY',
):
    assert forbidden not in true_source, f"active R10 contains forbidden same-day R0.5 sell path: {forbidden}"

print("R10 EXIT CAUSALITY CONTRACT: PASS — ALL sells T-close -> N+1 OPEN")
