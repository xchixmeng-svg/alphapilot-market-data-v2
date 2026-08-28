#!/usr/bin/env python3
from __future__ import annotations
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd

import clean_event_loop as cel
from clean_event_loop import PortfolioEngine
from clean_strategy_research import ResearchStrategy, build_feature_store
from clean_research_run import candidate_pool, metrics, FEATURE_COLS, BarStore, INITIAL, DD_GATE

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'.clean_cache'
OUT=ROOT/'clean_results'; OUT.mkdir(exist_ok=True)
DISCOVERY_START=20210101
DISCOVERY_END=20231231
HOLDOUT_START=20240101
HOLDOUT_END=20251231


def load_data():
    raw=pd.concat([pd.read_parquet(CACHE/f'ohlcv_{y}.parquet') for y in range(2021,2026)],ignore_index=True)
    raw=raw[raw.date.between(20210101,20251231)].copy()
    raw=raw[~raw.code.astype(str).str.startswith('00')].copy()
    feat=pd.read_parquet(CACHE/'features_2020_2025.parquet',columns=FEATURE_COLS)
    feat=feat[feat.date.between(20200101,20251231)].copy()
    return raw,build_feature_store(feat)


def run_window(raw,store,p,start,end,initial=INITIAL):
    q=raw[raw.date.between(start,end)].copy()
    dates=[str(int(x)) for x in sorted(q.date.unique())]
    if len(dates)<150: raise RuntimeError(f'too few dates {start}-{end}: {len(dates)}')
    return metrics(PortfolioEngine(initial).run(dates,BarStore(q),ResearchStrategy(store,p)))


def score(m):
    if m['max_dd']<=DD_GATE or m['completed_trades']<10: return -1e9
    return m['cagr']-0.35*abs(m['max_dd'])-0.03*m['annual_std']


def expanding_walk_forward(raw,store,pool):
    folds=[]
    for test_year in range(2022,2026):
        train_end=(test_year-1)*10000+1231
        ranked=[]
        for p in pool:
            m=run_window(raw,store,p,20210101,train_end)
            ranked.append((score(m),p,m))
        ranked.sort(key=lambda x:x[0],reverse=True)
        _,p,train=ranked[0]
        test=run_window(raw,store,p,test_year*10000+101,test_year*10000+1231)
        folds.append({'test_year':test_year,'selected_params':asdict(p),'train':train,'test':test})
    rets=[x['test']['end_nav']/INITIAL-1 for x in folds]
    dds=[x['test']['max_dd'] for x in folds]
    gate=(sum(r>0 for r in rets)>=3 and min(dds)>DD_GATE and np.prod([1+r for r in rets])>1.0)
    return {'status':'PASS' if gate else 'FAIL','folds':folds,'positive_test_years':sum(r>0 for r in rets),'compound_oos_return':float(np.prod([1+r for r in rets])-1),'worst_oos_dd':float(min(dds))}


def neighbors(p):
    vals=[]
    for field,deltas in [('score_threshold',[-.03,.03]),('limit_discount',[-.005,.005]),('max_position_pct',[-.04,.04]),('trailing_pct',[-.04,.04])]:
        for d in deltas:
            v=getattr(p,field)+d
            if field=='score_threshold' and not (0.55<=v<=0.85): continue
            if field=='limit_discount' and not (0<=v<=0.02): continue
            if field=='max_position_pct' and not (0.10<=v<=0.20): continue
            if field=='trailing_pct' and not (0.04<=v<=0.20): continue
            vals.append(replace(p,**{field:round(v,4)}))
    return vals


def stability_window(raw,store,p,start,end,min_ratio=0.60):
    base=run_window(raw,store,p,start,end)
    rows=[]
    for n in neighbors(p):
        m=run_window(raw,store,n,start,end)
        rows.append({'params':asdict(n),**m})
    floor=max(0.0,base['cagr']*0.50)
    good=[x for x in rows if x['max_dd']>DD_GATE and x['completed_trades']>=5 and x['cagr']>=floor]
    ratio=len(good)/len(rows) if rows else 0.0
    return {'status':'PASS' if ratio>=min_ratio else 'FAIL','base':base,'neighbor_count':len(rows),'robust_neighbor_count':len(good),'robust_ratio':ratio,'neighbors':rows}


def stressed_run(raw,store,p):
    orig_commission=cel.commission; orig_sell=cel.sell_fill_price; orig_buy=cel.buy_fill_price
    try:
        cel.commission=lambda v: orig_commission(v)*1.5
        cel.sell_fill_price=lambda op: op*0.97
        def harsh_buy(order,next_open,next_low):
            trigger=order.limit_price*0.995
            if next_low>trigger: return None
            if next_open<=trigger: return min(order.limit_price,next_open*1.01)
            return order.limit_price
        cel.buy_fill_price=harsh_buy
        return run_window(raw,store,p,20210101,20251231)
    finally:
        cel.commission=orig_commission; cel.sell_fill_price=orig_sell; cel.buy_fill_price=orig_buy


def main():
    raw,store=load_data(); pool=candidate_pool(72)

    discovery=[]
    for p in pool:
        m=run_window(raw,store,p,DISCOVERY_START,DISCOVERY_END)
        discovery.append((score(m),p,m))
    discovery.sort(key=lambda x:x[0],reverse=True)
    shortlist=[x for x in discovery if x[0]>-1e8][:8]
    if not shortlist:
        report={'status':'NO_CANDIDATE','reason':'No discovery candidate passed DD/trade gates'}
    else:
        wf=expanding_walk_forward(raw,store,pool)
        checks=[]
        for rank,(s,p,disc) in enumerate(shortlist,1):
            hold=run_window(raw,store,p,HOLDOUT_START,HOLDOUT_END)
            disc_stab=stability_window(raw,store,p,DISCOVERY_START,DISCOVERY_END)
            hold_stab=stability_window(raw,store,p,HOLDOUT_START,HOLDOUT_END,min_ratio=0.50)
            stress=stressed_run(raw,store,p)
            hold_ok=(hold['max_dd']>DD_GATE and hold['end_nav']>INITIAL and hold['completed_trades']>=8 and hold['positive_years']>=1)
            stress_ok=(stress['max_dd']>DD_GATE and stress['end_nav']>INITIAL)
            passed=(hold_ok and disc_stab['status']=='PASS' and hold_stab['status']=='PASS' and stress_ok)
            checks.append({'discovery_rank':rank,'discovery_score':s,'params':asdict(p),'discovery':disc,'holdout_2024_2025':hold,'discovery_stability':disc_stab,'holdout_stability':hold_stab,'stress':stress,'pass':passed})

        valid=[x for x in checks if x['pass']]
        # Candidate order is frozen by discovery-only ranking. Holdout is pass/fail only.
        winner=valid[0] if valid else None
        if winner:
            p=next(p for _,p,_ in shortlist if asdict(p)==winner['params'])
            winner['full_2021_2025']=run_window(raw,store,p,20210101,20251231)
        final_pass=(wf['status']=='PASS' and winner is not None)
        report={
          'status':'PASS' if final_pass else 'FAIL',
          'selection_protocol':'Rank on 2021-2023 only; 2024-2025 holdout is pass/fail only; final full-period metrics computed only after candidate lock.',
          'walk_forward':wf,'candidate_checks':checks,'best_validated':winner if final_pass else None,
          'note':'Final winner requires discovery DD gate, untouched 2024-2025 holdout, expanding walk-forward, neighboring-parameter stability in both discovery and holdout, and adverse-execution stress PASS.'
        }
    (OUT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'status':report['status'],'best':None if not report.get('best_validated') else report['best_validated'].get('full_2021_2025')}
    print(json.dumps(summary,ensure_ascii=False))
    if report['status']!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
