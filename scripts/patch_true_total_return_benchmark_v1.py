#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / 'r10_true_validation.py'
s = p.read_text(encoding='utf-8')

old = 'import r10_fast_validation as fast\n'
new = 'import r10_fast_validation as fast\nimport r10_total_return_benchmark as trb\n'
if 'import r10_total_return_benchmark as trb\n' not in s:
    if old not in s:
        raise SystemExit('import anchor not found')
    s = s.replace(old, new, 1)

old = '''    raw = bt.load_scenario_ohlcv(cfg)
    feat, corp_events, bm = bt.build_features(raw)
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= int(x) <= eval_end)
'''
new = '''    raw = bt.load_scenario_ohlcv(cfg)
    feat, corp_events, _price_proxy_bm = bt.build_features(raw)
    # Formal R7 uses the archived 0050 ETF Total Return series. Reconstruct it
    # causally from raw 0050 close + archived cash distributions + known split,
    # then apply the documented past-only forward-fill policy.
    all_feat_dates = sorted(int(x) for x in feat.date.unique())
    bm, benchmark_meta = trb.load_0050_total_return(all_feat_dates, raw)
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= int(x) <= eval_end)
'''
if 'bm, benchmark_meta = trb.load_0050_total_return' not in s:
    if old not in s:
        raise SystemExit('benchmark build anchor not found')
    s = s.replace(old, new, 1)

anchor = '        "causality": "T-close decision; T+1 sells before T+1 buys; raw T+1 Open/Low execution only",\n'
if '        "r7_benchmark": benchmark_meta,\n' not in s:
    if anchor not in s:
        raise SystemExit('provenance causality anchor not found')
    s = s.replace(anchor, '        "r7_benchmark": benchmark_meta,\n' + anchor, 1)

p.write_text(s, encoding='utf-8')
print('PATCHED', p)
