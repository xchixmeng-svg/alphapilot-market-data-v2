#!/usr/bin/env python3
from __future__ import annotations

# Realistic-execution rebuild: derive fills from the 410 locked orders + raw T+1 OHLCV.
# It intentionally does NOT accept the workbook's 241 completed trades as fill truth.
import json
from pathlib import Path
import pandas as pd

import r10_exact_replay as ex

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'clean_results'
OUT.mkdir(exist_ok=True)


def main():
    b = ex.load_bundle()
    raw = ex.load_raw()
    orders = ex.order_frame(b)
    audit = ex.validate_orders(raw, orders)

    # First immutable milestone: recompute every locked order's market outcome independently.
    # This exposes exactly which legacy order statuses/fills cannot survive realistic execution.
    rebuilt = audit.copy()
    rebuilt['legacy_changed'] = ~rebuilt['match']
    rebuilt.to_csv(OUT / 'r10_realistic_order_rebuild.csv', index=False)

    changed = rebuilt[rebuilt.legacy_changed]
    summary = {
        'strategy': 'AlphaPilot R10-MAX Realistic Execution Rebuild',
        'phase': 'LOCKED_ORDER_MARKET_REBUILD',
        'reference_orders': int(len(orders)),
        'orders_recomputed_from_raw_ohlcv': int(len(rebuilt)),
        'legacy_exact_matches': int(rebuilt['match'].sum()),
        'legacy_orders_changed': int(changed.shape[0]),
        'realistic_filled_orders': int((rebuilt['actual'] == 'FILLED').sum()),
        'realistic_missed_orders': int((rebuilt['actual'] == 'MISSED').sum()),
        'changed_orders': changed[['strategy','order_date','code','limit','open','low','expected','actual','expected_fill','actual_fill','reason']].to_dict('records'),
        'portfolio_rebuild_status': 'PENDING_EVENT_RECONSTRUCTION',
        'performance_numbers_valid': False,
        'note': 'Do not report NAV/CAGR/DD yet. Portfolio exits/sizing must be replayed after changed fills are propagated chronologically.'
    }
    (OUT / 'r10_realistic_execution_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
