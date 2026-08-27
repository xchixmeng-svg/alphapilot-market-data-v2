#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10 scanner accuracy patch: locked 0050 Total Return benchmark for 2026."""
from __future__ import annotations
import json
import pandas as pd
import numpy as np
import build_r10_scan as core

# Locked 2026 forward-test cash distributions already used by AlphaPilot.
CASH_DISTRIBUTIONS = {
    20260122: 1.0,
    20260721: 0.6,
}


def exact_0050_total_return(target):
    """Build causal 0050 Total Return from official raw OHLCV + known cash distributions.

    Missing 0050 source dates are forward-filled causally (TR NAV unchanged), matching the
    locked forward-test policy. No future values are used.
    """
    raw = core.history(target)
    dates = sorted(int(x) for x in raw["date"].unique())
    etf = raw[raw["code"].astype(str) == "0050"].copy()
    closes = dict(zip(etf["date"].astype(int), pd.to_numeric(etf["close"], errors="coerce")))

    rows = []
    prev_close = None
    tr = None
    for d in dates:
        c = closes.get(d, np.nan)
        if not np.isfinite(c):
            if tr is not None:
                rows.append((d, float(tr)))
            continue
        c = float(c)
        if prev_close is None:
            tr = 100.0
        else:
            div = float(CASH_DISTRIBUTIONS.get(d, 0.0))
            tr *= (c + div) / float(prev_close)
        prev_close = c
        rows.append((d, float(tr)))

    out = pd.DataFrame(rows, columns=["date", "mkt"])
    if len(out) < 120:
        raise RuntimeError(f"0050 Total Return insufficient rows: {len(out)}")
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


# Replace only the benchmark construction. All R7/R0.5 selection parameters stay locked.
core.tai50 = exact_0050_total_return
core.VERSION = "AlphaPilot-R10-Scanner-V1.1-0050TR"

if __name__ == "__main__":
    core.main()

    # Correct metadata written by the V1.0 core so the audit file states the actual benchmark.
    target = core.date.fromisoformat(core.TFILE.read_text(encoding="utf-8").strip())
    state_path = core.DATA / target.isoformat() / "r10_scan" / "market_state.json"
    if state_path.exists():
        obj = json.loads(state_path.read_text(encoding="utf-8"))
        obj["version"] = core.VERSION
        obj["benchmark"] = {
            "name": "0050 Total Return",
            "source": "official 0050 OHLCV + locked causal cash distributions",
            "cash_distributions": {"2026-01-22": 1.0, "2026-07-21": 0.6},
            "missing_value_policy": "causal forward-fill only; no future backfill",
            "rows": int(len(exact_0050_total_return(target))),
        }
        state_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    manifest_path = core.DATA / target.isoformat() / "r10_scan" / "scan_manifest.json"
    if manifest_path.exists():
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        obj["version"] = core.VERSION
        obj["benchmark"] = "0050 Total Return"
        manifest_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
