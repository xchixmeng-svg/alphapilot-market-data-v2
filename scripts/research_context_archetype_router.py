from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SRC=Path('research_out_peer_persistence')
CTXDIR=Path('research_out_continuous_context')
OUT=Path('research_out_context_archetype'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; MAX_EXPOSURE=.95; ADV_CAP=.02
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
MIN_PRIOR_DAYS=60; K=40; MIN_TRADES=12

# Inputs are generated causally by the upstream research jobs.
base=pd.read_csv(SRC/'orders_base_peer.csv',dtype={'code':str})
ctx=pd.read_csv(CTXDIR/'daily_continuous_context.csv')
if base.empty: raise RuntimeError('base peer orders missing')
base['code']=base.code.astype(str).str.zfill(4)
for c in ['signal_date','entry_date','exit_date']: base[c]=base[c].astype(int)
ctx['date']=ctx.date.astype(int)

# Use the same continuous market representation, with no semantic bull/bear labels.
Z=[c for c in ctx.columns if c.endswith('_z')]
ctx=ctx[ctx.get('context_ready',False).astype(bool)].copy().sort_values('date')
ctx_vec={int(r.date):np.array([getattr(r,c) for c in Z],dtype=float) for r in ctx.itertuples(index=False)}
ctx_dates=sorted(ctx_vec)

# Structural archetypes. These are not threshold sweeps and are defined before 2025 evaluation.
# base: broad peer rotation; persistent: sustained peer leadership; flow_persistent: sustained leadership + known historical edge.
def tag_arch(r):
    if int(getattr(r,'persist5',0))>=3 and int(getattr(r,'persist10',0))>=5 and int(getattr(r,'hist_n',0))>=20 and float(getattr(r,'hist_pf',0))>=1.10 and float(getattr(r,'hist_wr',0))>=0.48:
        return 'flow_persistent'
    if int(getattr(r,'persist5',0))>=3 and int(getattr(r,'persist10',0))>=5:
        return 'persistent'
    return 'broad_peer'
base['archetype']=base.apply(tag_arch,axis=1)
base['event_return']=(base.exit_price.astype(float)*(1-FEE-TAX))/(base.entry_price.astype(float)*(1+FEE))-1

# Candidate quality here remains market/liquidity quality only. No fundamental claim.
# Historical archetype outcomes are eligible only after their exits are already known.
def dist(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<max(6,int(.7*len(a))): return np.inf
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def choose_arch(d):
    if d not in ctx_vec: return ('broad_peer','context_missing')
    prior=[x for x in ctx_dates if x<d]
    if len(prior)<MIN_PRIOR_DAYS: return ('broad_peer','warmup')
    dv=sorted((dist(ctx_vec[d],ctx_vec[x]),x) for x in prior)
    neigh=[x for dd,x in dv if np.isfinite(dd)][:K]
    hist=base[(base.signal_date.isin(neigh))&(base.exit_date<d)].copy()
    rows=[]
    for a,q in hist.groupby('archetype'):
        if len(q)<MIN_TRADES: continue
        r=q.event_return.astype(float)
        pos=float(r[r>0].sum()); neg=float(-r[r<0].sum()); pf=pos/neg if neg>0 else (99. if pos>0 else 0.)
        wr=float((r>0).mean()); mean=float(r.mean()); med=float(r.median()); q10=float(r.quantile(.10))
        # Shrunk, downside-aware score. No target forcing and no 2025 tuning.
        shrink=len(r)/(len(r)+20.0)
        edge=shrink*(5.0*mean+1.4*med+1.2*(wr-.5)+.12*min(pf,3)+1.2*min(q10,0))
        rows.append((edge,a,len(r),wr,pf,mean))
    if not rows: return ('broad_peer','no_history')
    rows.sort(reverse=True)
    best=rows[0]
    # Every market can still route to an archetype; do not use CASH as a primary gate.
    return (best[1],f'n={best[2]},wr={best[3]:.3f},pf={best[4]:.3f},mean={best[5]:.4f}')

sel=[]
for d in sorted(base.signal_date.unique()):
    a,reason=choose_arch(int(d)); sel.append({'signal_date':int(d),'selected_archetype':a,'selection_reason':reason})
selector=pd.DataFrame(sel)
selector.to_csv(OUT/'daily_archetype_selection.csv',index=False)
chosen=base.merge(selector,on='signal_date',how='left')
chosen=chosen[chosen.archetype==chosen.selected_archetype].copy()

# Within the selected archetype, screen names only after context routing.
# For flow/persistent archetypes emphasize peer leadership; broad peer keeps diversified ranking.
chosen['route_rank_score']=chosen.candidate_score.astype(float)
chosen.loc[chosen.selected_archetype=='persistent','route_rank_score'] += .10*chosen.loc[chosen.selected_archetype=='persistent','lead_score'].astype(float)
chosen.loc[chosen.selected_archetype=='flow_persistent','route_rank_score'] += .12*chosen.loc[chosen.selected_archetype=='flow_persistent','lead_score'].astype(float) + .03*chosen.loc[chosen.selected_archetype=='flow_persistent','hist_pf'].clip(upper=3).astype(float)
# separate market/liquidity quality pass, not business fundamentals
chosen=chosen[(chosen.quality!='loose') | (chosen.vol20.astype(float)>0)].copy()
chosen=chosen.sort_values(['signal_date','route_rank_score'],ascending=[True,False])
chosen=chosen.groupby('signal_date',as_index=False,group_keys=False).head(6)
chosen.to_csv(OUT/'routed_orders.csv',index=False)

px=pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz',dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4); px=px.sort_values(['code','date'])
px_idx={(int(r.date),r.code):r for r in px.itertuples(index=False)}
calendar=sorted(int(x) for x in px[(px.date>=START)&(px.date<=END)].date.unique())

def tick(p):
    if p<10:return .01
    if p<50:return .05
    if p<100:return .1
    if p<500:return .5
    if p<1000:return 1.0
    return 5.0
def floor_tick(p):
    t=tick(p); return math.floor((p+1e-12)/t)*t

def pf_of(r):
    r=pd.Series(r,dtype=float); pos=float(r[r>0].sum()); neg=float(-r[r<0].sum())
    return pos/neg if neg>0 else (99. if pos>0 else 0.)
def mark(d,c,field,fallback):
    r=px_idx.get((int(d),c)); v=getattr(r,field,np.nan) if r else np.nan
    return float(v) if np.isfinite(v) and v>0 else fallback

def simulate(src,start,end,label):
    days=[d for d in calendar if start<=d<=end]
    em=defaultdict(list)
    for r in src[(src.entry_date>=start)&(src.entry_date<=end)].itertuples(index=False): em[int(r.entry_date)].append(r)
    cash=INIT; pos={}; trades=[]; navrows=[]; min_cash=INIT
    for d in days:
        for c in list(pos):
            p=pos[c]
            if p['exit_date']!=d: continue
            proceeds=p['shares']*p['exit_price']*(1-FEE-TAX); cash+=proceeds
            trades.append({**p,'exit_fill_date':d,'realized_return':proceeds/p['cost']-1}); del pos[c]
        open_value=sum(p['shares']*mark(d,c,'open',p['entry_price']) for c,p in pos.items())
        nav_open=cash+open_value
        for r in sorted(em.get(d,[]),key=lambda x:getattr(x,'route_rank_score',0),reverse=True):
            if r.code in pos or len(pos)>=4: continue
            budget=min(nav_open*.24,max(0.,nav_open*MAX_EXPOSURE-open_value),cash)
            adv=int(max(0,math.floor(float(r.vol20)*ADV_CAP)))
            shares=min(int(budget//(float(r.entry_price)*(1+FEE))),adv)
            if shares<=0: continue
            cost=shares*float(r.entry_price)*(1+FEE)
            if cost>cash+1e-8: continue
            cash-=cost; min_cash=min(min_cash,cash); open_value+=shares*float(r.entry_price)
            pos[r.code]={'code':r.code,'signal_date':int(r.signal_date),'entry_date':int(r.entry_date),'exit_date':int(r.exit_date),'entry_price':float(r.entry_price),'exit_price':float(r.exit_price),'shares':shares,'cost':cost,'archetype':r.archetype,'playbook':r.playbook}
        mv=sum(p['shares']*mark(d,c,'close',p['entry_price']) for c,p in pos.items()); nav=cash+mv
        navrows.append({'date':d,'cash':cash,'market_value':mv,'nav':nav,'positions':len(pos)})
        if cash < -1e-6: raise AssertionError(('negative cash',d,cash))
    last=days[-1]
    for c in list(pos):
        p=pos[c]; xp=floor_tick(mark(last,c,'close',p['entry_price'])*.995); proceeds=p['shares']*xp*(1-FEE-TAX); cash+=proceeds
        trades.append({**p,'exit_fill_date':last,'realized_return':proceeds/p['cost']-1,'forced_end':True}); del pos[c]
    td=pd.DataFrame(trades); nd=pd.DataFrame(navrows)
    rr=td.realized_return.astype(float) if not td.empty else pd.Series(dtype=float)
    years=max((pd.Timestamp(str(end))-pd.Timestamp(str(start))).days/365.25,1/252)
    mdd=float((nd.nav/nd.nav.cummax()-1).min()) if not nd.empty else 0.
    return {'label':label,'start':start,'end':end,'start_nav':INIT,'end_nav':float(cash),'return':float(cash/INIT-1),'cagr':float((cash/INIT)**(1/years)-1),'max_dd':mdd,'trades':int(len(td)),'wins':int((rr>0).sum()),'win_rate':float((rr>0).mean()) if len(rr) else 0.,'pf':pf_of(rr),'min_cash':float(min_cash)},td,nd

rows=[]; yearly=[]
for a,b,lab in [(START,DEV_END,'dev'),(HOLDOUT_START,END,'holdout_2025'),(START,END,'full')]:
    s,td,nd=simulate(chosen,a,b,lab); rows.append(s); td.to_csv(OUT/f'trades_{lab}.csv',index=False); nd.to_csv(OUT/f'nav_{lab}.csv',index=False)
for y in [2023,2024,2025]:
    a=max(START,y*10000+101); b=min(END,y*10000+1231)
    if a<=b:
        s,_,_=simulate(chosen,a,b,str(y)); s['year']=y; yearly.append(s)
summary=pd.DataFrame(rows); years=pd.DataFrame(yearly)
summary.to_csv(OUT/'portfolio_summary.csv',index=False); years.to_csv(OUT/'yearly_summary.csv',index=False)
h=summary[summary.label=='holdout_2025'].iloc[0]
stable=bool(h['return']>0 and h['pf']>1.10 and h['trades']>=25 and h['win_rate']>.50)
decision={'architecture':'causal continuous context -> archetype -> stock screen -> market/liquidity quality -> T+1 shared capital','semantic_market_gate':False,'holdout_optimized':False,'fundamental_filter_available':False,'minute_stage_allowed':stable,'primary_target_met':bool(h.cagr>=.50 and h.win_rate>=.70 and h.trades>=25),'holdout':h.to_dict(),'formal_r10_modified':False}
json.dump(decision,open(OUT/'decision.json','w'),indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
print(summary.to_string(index=False)); print(years.to_string(index=False)); print(json.dumps(decision,indent=2,default=str))
