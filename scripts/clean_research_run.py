#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math, random
from dataclasses import asdict
from pathlib import Path
from collections.abc import Mapping, Iterator
import numpy as np
import pandas as pd

from clean_event_loop import Bar, PortfolioEngine
from clean_strategy_research import Params, ResearchStrategy, build_feature_store

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'.clean_cache'
OUT=ROOT/'clean_results'
OUT.mkdir(exist_ok=True)
INITIAL=1_000_000.0
EVAL_START=20210101
EVAL_END=20251231
DD_GATE=-0.20

FEATURE_COLS=[
 'date','code','name','close','aclose','avgamt20','ma20','ma60','ma120','prior_high20',
 'ret20','ret60','rs20','rs60','dist_ma20','break20','clv',
 'pct_rs20','pct_rs60','pct_ret20','pct_ret60','pct_amt_ratio',
 'pct_foreign5','pct_foreign20','pct_trust5','pct_trust20',
 'mkt_aclose','mkt_ma60','mkt_ma120','mkt_ret20','mkt_ret60'
]

class DayBars(Mapping):
    def __init__(self,q: pd.DataFrame): self.q=q
    def __len__(self): return len(self.q)
    def __iter__(self)->Iterator[str]: return iter(self.q.index)
    def __getitem__(self,code):
        r=self.q.loc[code]
        if isinstance(r,pd.DataFrame): r=r.iloc[-1]
        return Bar(str(int(r.date)),str(code),str(r['name']),float(r.open),float(r.high),float(r.low),float(r.close))
    def get(self,code,default=None):
        try:return self[code]
        except KeyError:return default

class BarStore:
    def __init__(self,raw: pd.DataFrame):
        self.by_date={str(int(d)):q.set_index('code',drop=False) for d,q in raw.groupby('date',sort=True)}
    def __getitem__(self,date): return DayBars(self.by_date[date])


def annual_stats(nav_rows):
    rows=pd.DataFrame([{'date':int(x.date),'nav':x.nav} for x in nav_rows])
    rows['year']=rows.date.astype(str).str[:4].astype(int)
    out={}; prev=INITIAL
    for y in range(2021,2026):
        q=rows[rows.year.eq(y)]
        if q.empty: continue
        vals=np.r_[prev,q.nav.to_numpy(float)]
        peaks=np.maximum.accumulate(vals); dd=float(np.min(vals/peaks-1))
        end=float(vals[-1]); ret=end/prev-1
        out[str(y)]={'return':ret,'max_dd':dd,'end_nav':end}
        prev=end
    return out


def metrics(result):
    end=float(result['end_nav']); cagr=(end/INITIAL)**(1/5)-1
    ann=annual_stats(result['nav_rows']); rets=np.array([ann[str(y)]['return'] for y in range(2021,2026) if str(y) in ann],float)
    fills=sum(1 for x in result['order_log'] if x.get('side')=='BUY' and x.get('filled'))
    buy_orders=sum(1 for x in result['order_log'] if x.get('side')=='BUY')
    return {
      'end_nav':end,'cagr':float(cagr),'max_dd':float(result['max_drawdown']),
      'completed_trades':int(result['completed_trades']),'orders':int(result['orders']),
      'buy_fill_rate':float(fills/buy_orders) if buy_orders else 0.0,
      'positive_years':int((rets>0).sum()) if len(rets) else 0,
      'worst_year':float(rets.min()) if len(rets) else -1.0,
      'annual_std':float(rets.std()) if len(rets) else 9.0,'annual':ann,
    }


def candidate_pool(n=72):
    product=list(itertools.product(
      ['MOM_RS_FLOW','BREAK_FLOW','PULLBACK_RS'],
      [0.64,0.70,0.76],[0.0,0.005,0.01],[0.12,0.16,0.20],
      [4,5],[2,3],[20,40],[0.08,0.12,0.16],[40,80,120],[0.30,0.45]
    ))
    rng=random.Random(20260828); rng.shuffle(product)
    # Seed a neutral central point per family, then deterministic broad sample.
    seeds=[(f,0.70,0.005,0.16,5,3,20,0.12,80,0.30) for f in ['MOM_RS_FLOW','BREAK_FLOW','PULLBACK_RS']]
    chosen=seeds+product[:max(0,n-len(seeds))]
    out=[]; seen=set()
    for x in chosen:
        if x in seen: continue
        seen.add(x)
        f,thr,disc,pos,ss,sn,ema,tr,mh,dec=x
        out.append(Params(family=f,score_threshold=thr,limit_discount=disc,max_position_pct=pos,max_slots_strong=ss,max_slots_neutral=sn,exit_ma=ema,trailing_pct=tr,max_hold=mh,momentum_decay_pct=dec))
    return out


def main():
    raw=pd.concat([pd.read_parquet(CACHE/f'ohlcv_{y}.parquet') for y in range(2021,2026)],ignore_index=True)
    raw=raw[raw.date.between(EVAL_START,EVAL_END)].copy()
    features=pd.read_parquet(CACHE/'features_2020_2025.parquet',columns=FEATURE_COLS)
    features=features[features.date.between(20200101,EVAL_END)].copy()
    # Execution universe uses raw prices. 0050 and other 00xx codes are benchmark-only, not tradable.
    raw=raw[~raw.code.astype(str).str.startswith('00')].copy()
    eval_dates=[str(int(x)) for x in sorted(raw.date.unique())]
    if len(eval_dates)<1200: raise RuntimeError(f'too few evaluation dates {len(eval_dates)}')
    bars=BarStore(raw); store=build_feature_store(features)
    trials=[]
    for i,p in enumerate(candidate_pool(),1):
        engine=PortfolioEngine(INITIAL)
        result=engine.run(eval_dates,bars,ResearchStrategy(store,p))
        m=metrics(result); qualified=(m['max_dd']>DD_GATE and m['positive_years']>=3 and m['completed_trades']>=20)
        row={'trial':i,'params':asdict(p),'qualified':bool(qualified),**m}
        trials.append(row)
        print(json.dumps({'trial':i,'family':p.family,'cagr':m['cagr'],'dd':m['max_dd'],'trades':m['completed_trades'],'qualified':qualified}),flush=True)
    qualified=[x for x in trials if x['qualified']]
    # Contract objective: DD gate/robustness first, then maximum CAGR.
    qualified.sort(key=lambda x:(x['cagr'],-abs(x['max_dd'])),reverse=True)
    best=qualified[0] if qualified else None
    summary={
      'status':'PASS' if best else 'NO_QUALIFIED_STRATEGY',
      'initial_capital':INITIAL,'evaluation':'2021-2025','warmup':'2020',
      'dd_gate':'strictly below 20%','trial_count':len(trials),
      'qualified_count':len(qualified),'best':best,
      'note':'Research-stage full-sample search only. Not final until walk-forward, neighboring-parameter stability and stress validation pass.'
    }
    (OUT/'search_trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'search_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'qualified':len(qualified),'best':None if best is None else {'cagr':best['cagr'],'dd':best['max_dd']}},ensure_ascii=False))

if __name__=='__main__': main()
