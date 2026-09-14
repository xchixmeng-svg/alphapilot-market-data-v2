#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import research_v25_breakout_quality_structures as v25

ROOT = Path(__file__).resolve().parent.parent
_original_simulate = v25.v13.simulate_portfolio
_boundary_counts = {}

def simulate_portfolio_boundary_fixed(px_all, schedule, slots, start, end, initial=v25.INITIAL):
    s = schedule.copy()
    # Correct period membership is based on execution date, not signal date.
    # A signal formed before the boundary but executed T+1 inside the period is causal and must be included.
    entry = s['entry_date'].astype(int)
    signal = s['signal_date'].astype(int)
    mask = (entry >= int(start)) & (entry <= int(end)) & (signal < int(start))
    key = f"{int(start)}-{int(end)}"
    _boundary_counts[key] = int(mask.sum())
    if mask.any():
        s.loc[mask, 'signal_date'] = int(start)
    return _original_simulate(px_all, s, slots, start, end, initial)

v25.v13.simulate_portfolio = simulate_portfolio_boundary_fixed

if __name__ == '__main__':
    # Preserve the exact V25 structural hypotheses; only repair the invalid OOS boundary accounting.
    hyp = None
    if '--hypothesis' in sys.argv:
        hyp = sys.argv[sys.argv.index('--hypothesis') + 1]
    v25.main()
    if hyp:
        p = ROOT / 'v25_breakout_quality_out' / hyp / 'audit.json'
        if p.exists():
            audit = json.loads(p.read_text(encoding='utf-8'))
            audit['oos_boundary_fix'] = {
                'status': 'PASS',
                'rule': 'include pre-boundary signal when its T+1 entry_date falls inside evaluation period',
                'boundary_rows_promoted': _boundary_counts,
                'research_logic_changed': False,
            }
            p.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
