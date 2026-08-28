#!/usr/bin/env python3
from __future__ import annotations
import json, math
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd

import clean_event_loop as cel
from clean_event_loop import Bar, PortfolioEngine
from clean_strategy_research import Params, ResearchStrategy, build_feature_store
from clean_research_run import candidate_pool, metrics, FEATURE_COLS, BarStore, INITIAL, DD_GATE

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'.clean_cache'
OUT=ROOT/'clean_results'; OUT.mkdir(exist_ok=True)


def load_data():
    raw=pd.concat([pd.read_parquet(CACHE/f'ohlcv_{y}.parquet') for y in range(2021,2026)],ignore_index=True)
    raw=raw[raw.date.between(20210101,20251231)].copy()
    raw=raw[~raw.code.astype(str).str.startswith('00')].copy()
    feat=pd.read_parquet(CACHE/'features_2020_2025.parquet',columns=FEATURE_COLS)
    feat=feat[feat.date.between(20200101,20251231)].copy()
    return raw,feat,build_feature_store(feat)


def run_window(raw, store, p, start, end, initial=INITIAL):
    q=raw[raw.date.between(start,end)].copy()
    dates=[str(int(x)) for x in sorted(q.date.unique())]
    if len(dates)<150: raise RuntimeError(f'too few dates {start}-{end}: {len(dates)}')
    r=PortfolioEngine(initial).run(dates,BarStore(q),ResearchStrategy(store,p))
    m=metrics(r)
    m['window_start']=start; m['window_end']=end
    return m


def train_score(m):
    if m['max_dd']<=DD_GATE or m['completed_trades']<10: return -1e9
    return m['cagr'] - 0.30*abs(m['max_dd']) - 0.03*m['annual_std']


def expanding_walk_forward(raw,store,pool):
    folds=[]
    for test_year in range(2022,2026):
        train_end=(test_year-1)*10000+1231
        scored=[]
        for p in pool:
            m=run_window(raw,store,p,20210101,train_end)
            scored.append((train_score(m),p,m))
        scored.sort(key=lambda x:x[0],reverse=True)
        score,p,train=scored[0]
        test=run_window(raw,store,p,test_year*10000+101,test_year*10000+1231)
        folds.append({'test_year':test_year,'selected_params':asdict(p),'train':train,'test':test})
    test_returns=[x['test']['end_nav']/INITIAL-1 for x in folds]
    test_dd=[x['test']['max_dd'] for x in folds]
    pass_gate=(sum(r>0 for r in test_returns)>=3 and min(test_dd)>DD_GATE and np.prod([1+r for r in test_returns])>1.0)
    return {'status':'PASS' if pass_gate else 'FAIL','folds':folds,'positive_test_years':sum(r>0 for r in test_returns),'compound_oos_return':float(np.prod([1+r for r in test_returns])-1),'worst_oos_dd':float(min(test_dd))}


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


def stability(raw,store,p):
    base=run_window(raw,store,p,20210101,20251231)
    rows=[]
    for n in neighbors(p):
        m=run_window(raw,store,n,20210101,20251231)
        rows.append({'params':asdict(n),**m})
    good=[x for x in rows if x['max_dd']>DD_GATE and x['completed_trades']>=20 and x['cagr']>=max(0,base['cagr']*0.60)]
    ratio=len(good)/len(rows) if rows else 0
    return {'status':'PASS' if ratio>=0.60 else 'FAIL','base':base,'neighbor_count':len(rows),'robust_neighbor_count':len(good),'robust_ratio':ratio,'neighbors':rows}


def stressed_run(raw,store,p):
    orig_commission=cel.commission; orig_sell=cel.sell_fill_price; orig_buy=cel.buy_fill_price
    try:
        cel.commission=lambda v: orig_commission(v)*1.5
        cel.sell_fill_price=lambda op: op*0.97
        def harsh_buy(order,next_open,next_low):
            # Worse fill quality: order must penetrate another 0.5% below its limit.
            trigger=order.limit_price*0.995
            if next_low>trigger: return None
            if next_open<=trigger: return min(order.limit_price,next_open*1.01)
            return order.limit_price
        cel.buy_fill_price=harsh_buy
        return run_window(raw,store,p,20210101,20251231)
    finally:
        cel.commission=orig_commission; cel.sell_fill_price=orig_sell; cel.buy_fill_price=orig_buy


def main():
    raw,feat,store=load_data(); pool=candidate_pool(72)
    full=[]
    for p in pool:
        m=run_window(raw,store,p,20210101,20251231)
        full.append((train_score(m),p,m))
    full.sort(key=lambda x:x[0],reverse=True)
    candidates=[x for x in full if x[2]['max_dd']>DD_GATE and x[2]['completed_trades']>=20][:5]
    if not candidates:
        report={'status':'NO_CANDIDATE','reason':'No broad-search candidate passed DD/trade gates'}
    else:
        wf=expanding_walk_forward(raw,store,pool)
        checked=[]
        for _,p,m in candidates:
            stab=stability(raw,store,p); stress=stressed_run(raw,store,p)
            passed=(stab['status']=='PASS' and stress['max_dd']>DD_GATE and stress['end_nav']>INITIAL)
            checked.append({'params':asdict(p),'full':m,'stability':stab,'stress':stress,'pass':passed})
        valid=[x for x in checked if x['pass']]
        valid.sort(key=lambda x:x['full']['cagr'],reverse=True)
        final_pass=(wf['status']=='PASS' and bool(valid))
        report={'status':'PASS' if final_pass else 'FAIL','walk_forward':wf,'candidate_checks':checked,'best_validated':valid[0] if final_pass else None,'note':'A final winner requires both expanding walk-forward PASS and candidate stability/stress PASS.'}
    (OUT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'best':None if not report.get('best_validated') else report['best_validated']['full']},ensure_ascii=False))
    if report['status']!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
