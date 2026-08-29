#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import clean_r10_retest as r10
import r10_exact_replay as ex

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.clean_cache'
OUT = ROOT / 'clean_results'
OUT.mkdir(exist_ok=True)
TARGET_DATE = 20210104


def load_inst() -> pd.DataFrame:
    parts=[]
    for y in range(2020, 2026):
        p=CACHE/f'institutional_{y}.parquet'
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError('institutional cache missing')
    q=pd.concat(parts,ignore_index=True)
    q['code']=q.code.astype(str).str.strip().str.zfill(4)
    return q.drop_duplicates(['date','code'],keep='last').sort_values(['date','code']).reset_index(drop=True)


def main():
    raw=r10.load_raw()
    inst=load_inst()
    r7=r10.build_r7_features(raw)
    r05=r10.build_r05_features(raw,inst)

    r7d=r7[r7.date.astype(int).eq(TARGET_DATE)].copy()
    r05d=r05[r05.date.astype(int).eq(TARGET_DATE)].copy()
    if r7d.empty:
        raise RuntimeError('no R7 features for target date')

    eng=r10.R10(raw,r7,r05)
    sample=r7d.iloc[0]
    regime,exposure,slots=eng.r7_regime(sample)

    r7elig=r7d[(r7d.avgamt20>=30_000_000)&(r7d.aclose>r7d.ma120)&(r7d.nearhigh>=0.78)&r7d.score.notna()].copy()
    r7elig=r7elig.sort_values(['score','code'],ascending=[False,True])

    r05elig=r05d[(r05d.aclose>=10)&(r05d.aclose<=40)&(r05d.avgamt20>=50_000_000)&(r05d.amt_ratio>=1.0)&(r05d.ret20>=0)&(r05d.ret20<=0.20)&(r05d.ma20_gap<=0.18)&(r05d.near_prior60>=-0.15)&(r05d.aclose>r05d.prior10high)&r05d.score.notna()].copy()
    r05elig=r05elig.sort_values(['score','code'],ascending=[False,True])

    teacher=ex.order_frame(ex.load_bundle()).copy()
    teacher['order_date_num']=teacher['order_date'].astype(str).str.replace(r'\D','',regex=True).str[:8].astype(int)
    td=teacher[teacher.order_date_num.eq(20210105)][['strategy','code','t1_limit']].copy()
    td['code']=td.code.astype(str).str.zfill(4)

    summary={
        'target_decision_date': TARGET_DATE,
        'r7_regime': regime,
        'r7_exposure': exposure,
        'r7_slots': slots,
        'r7_market_snapshot': {
            'mkt': float(sample.mkt),
            'ma60': float(sample.mkt_ma60),
            'ma120': float(sample.mkt_ma120),
            'ret20': float(sample.mkt_ret20),
            'ret60': float(sample.mkt_ret60),
            'breadth60': float(sample.breadth60),
            'breadth20mean': float(sample.breadth20mean),
            'advance10': float(sample.advance10),
        },
        'r7_top10': r7elig[['code','name','score','aclose','avgamt20','nearhigh']].head(10).to_dict('records'),
        'r05_risk_on': bool(r05d.iloc[0].risk_on) if not r05d.empty else False,
        'r05_top10': r05elig[['code','name','score','aclose','avgamt20','amt_ratio','ret20','ma20_gap','near_prior60','prior10high']].head(10).to_dict('records'),
        'teacher_20210105': td.to_dict('records'),
    }
    (OUT/'r10_first_divergence_diagnostic.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
