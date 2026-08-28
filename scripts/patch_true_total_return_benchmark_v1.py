#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parent / 'r10_true_validation.py'
s = p.read_text(encoding='utf-8')

old = 'import r10_fast_validation as fast\n'
new = 'import r10_fast_validation as fast\nimport r10_total_return_benchmark as trb\n'
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
    # Formal R7 uses the Taiwan-50/0050 TOTAL-RETURN benchmark, not a
    # continuity-adjusted ETF price proxy. Use the same official TWSE TAI50I
    # total-return field as the live scanner and align it past-only.
    all_feat_dates = sorted(int(x) for x in feat.date.unique())
    bm, benchmark_meta = trb.load_tai50_total_return(all_feat_dates, 2020, 2025)
    eval_start, eval_end = bt.intdate(cfg["eval_start"]), bt.intdate(cfg["eval_end"])
    eval_dates = sorted(int(x) for x in feat.date.unique() if eval_start <= int(x) <= eval_end)
'''
if old not in s:
    raise SystemExit('benchmark build anchor not found')
s = s.replace(old, new, 1)

old = '''        "institutional_sha256": _hash_frame(
            ins.fillna(0), ["date", "code", "foreign_net", "trust_net", "dealer_net", "Foreign3D", "Foreign10D", "Trust5D"]
        ),
        "causality": "T-close decision; T+1 sells before T+1 buys; raw T+1 Open/Low execution only",
'''
new = '''        "institutional_sha256": _hash_frame(
            ins.fillna(0), ["date", "code", "foreign_net", "trust_net", "dealer_net", "Foreign3D", "Foreign10D", "Trust5D"]
        ),
        "r7_benchmark": benchmark_meta,
        "causality": "T-close decision; T+1 sells before T+1 buys; raw T+1 Open/Low execution only",
'''
if old not in s:
    raise SystemExit('provenance benchmark anchor not found')
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf-8')
print('PATCHED', p)
