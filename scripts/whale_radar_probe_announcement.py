#!/usr/bin/env python3
"""Positive-control probe for Whale Radar historical MOPS source.

Use exactly the same transport and parameters as whale_radar_backfill_sec43.py.
This avoids a false failure caused by probing t05st01 with incomplete legacy
parameters while the backfiller uses the current MOPS gateway/direct route.
"""
from __future__ import annotations

import json

from whale_radar_backfill_sec43 import POSITIVE_CONTROL, probe_positive_control


def main() -> int:
    transport, diagnostics = probe_positive_control()
    for row in diagnostics:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    if not transport:
        print("POSITIVE_CONTROL_FAILED", json.dumps(POSITIVE_CONTROL, ensure_ascii=False))
        return 3
    print(
        "POSITIVE_CONTROL_OK",
        json.dumps(
            {"control": POSITIVE_CONTROL, "transport": transport},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
