#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import r10_exact_replay as ex

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'clean_results'
OUT.mkdir(exist_ok=True)
GEN = OUT / 'r10_clean_retest_orders.csv'


def norm_code(v):
    return str(v).strip().zfill(4)


def round_limit(v):
    if pd.isna(v): return None
    return round(float(v), 6)


def main():
    teacher = ex.order_frame(ex.load_bundle()).copy()
    teacher['code'] = teacher['code'].map(norm_code)
    teacher['order_date'] = pd.to_numeric(teacher['order_date'], errors='raise').astype(int)
    teacher['t1_limit_r'] = teacher['t1_limit'].map(round_limit)
    teacher['teacher_key'] = list(zip(teacher['order_date'], teacher['code'], teacher['t1_limit_r']))

    if not GEN.exists():
        raise RuntimeError(f'missing generated order file: {GEN}')
    gen = pd.read_csv(GEN, dtype={'code':str})
    gen = gen[gen.side.astype(str).eq('BUY')].copy()
    gen['code'] = gen['code'].map(norm_code)
    gen['execute_date'] = pd.to_numeric(gen['execute_date'], errors='raise').astype(int)
    gen['limit_r'] = gen['limit_price'].map(round_limit)
    gen['gen_key'] = list(zip(gen['execute_date'], gen['code'], gen['limit_r']))

    tkeys = set(teacher.teacher_key)
    gkeys = set(gen.gen_key)
    exact = tkeys & gkeys
    missed = tkeys - gkeys
    extra = gkeys - tkeys

    # Also score date+code agreement separately to isolate limit-price issues.
    tdc = set(zip(teacher.order_date, teacher.code))
    gdc = set(zip(gen.execute_date, gen.code))
    dc_exact = tdc & gdc

    summary = {
        'strategy': 'AlphaPilot R10-MAX Full Strategy Learning Gate',
        'teacher_orders': int(len(teacher)),
        'generated_buy_orders': int(len(gen)),
        'exact_date_code_limit_matches': int(len(exact)),
        'date_code_matches': int(len(dc_exact)),
        'teacher_orders_missing_from_engine': int(len(missed)),
        'engine_extra_orders_not_in_teacher': int(len(extra)),
        'exact_recall': float(len(exact) / len(tkeys)) if tkeys else 0.0,
        'exact_precision': float(len(exact) / len(gkeys)) if gkeys else 0.0,
        'date_code_recall': float(len(dc_exact) / len(tdc)) if tdc else 0.0,
        'learned': bool(len(missed) == 0 and len(extra) == 0 and len(gen) == 410),
        'gate': 'PASS' if (len(missed) == 0 and len(extra) == 0 and len(gen) == 410) else 'FAIL_WITH_DIAGNOSTICS',
        'note': 'PASS requires the engine to regenerate the locked 410 buy orders from market data; teacher fills are not used as execution truth.'
    }

    miss_df = teacher[teacher.teacher_key.isin(missed)][['strategy','order_date','code','t1_limit','status','fill_price']]
    extra_df = gen[gen.gen_key.isin(extra)][['decision_date','execute_date','code','shares','limit_price','filled','reason','fill_price','target_cash']]
    miss_df.to_csv(OUT/'r10_learning_missing_teacher_orders.csv', index=False)
    extra_df.to_csv(OUT/'r10_learning_extra_engine_orders.csv', index=False)
    (OUT/'r10_learning_gate_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
