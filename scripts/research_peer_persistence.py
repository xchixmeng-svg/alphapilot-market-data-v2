from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path('research_out_causal_peer_rotation')
OUT=Path('research_out_peer_persistence'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; MAX_EXPOSURE=.95; ADV_CAP=.02
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231


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

orders=pd.read_csv(BASE/'orders.csv',dtype={'code':str})
routes=pd.read_csv(BASE/'routes.csv')
if orders.empty: raise RuntimeError('base peer rotation produced no orders')
orders['code']=orders.code.astype(str).str.zfill(4)
for c in ['signal_date','entry_date','exit_date']:
    orders[c]=orders[c].astype(int)
routes['date']=routes.date.astype(int)

px=pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz',dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4); px=px.sort_values(['code','date'])
px_idx={(int(r.date),r.code):r for r in px.itertuples(index=False)}
calendar=sorted(int(x) for x in px[(px.date>=START)&(px.date<=END)].date.unique())

# Route persistence is measured only from previously observed route selections inside the
# current frozen monthly peer map. Group IDs are intentionally NOT linked across refits.
routes=routes.sort_values(['date','group']).reset_index(drop=True)
by_day={int(d):x.copy() for d,x in routes.groupby('date')}
cal_pos={d:i for i,d in enumerate(calendar)}

def route_features(d,g):
    i=cal_pos.get(int(d),-1)
    if i<0:return {'persist5':0,'persist10':0,'lead_med5':np.nan,'pf_med5':np.nan,'wr_med5':np.nan}
    month=str(int(d))[:6]
    prev5=[x for x in calendar[max(0,i-5):i] if str(x)[:6]==month]
    prev10=[x for x in calendar[max(0,i-10):i] if str(x)[:6]==month]
    def rows(ds):
        z=[]
        for dd in ds:
            q=by_day.get(dd)
            if q is None:continue
            q=q[q.group==g]
            if not q.empty:z.append(q.iloc[0])
        return z
    a5=rows(prev5); a10=rows(prev10)
    return {
        'persist5':len(a5), 'persist10':len(a10),
        'lead_med5':float(np.median([x.lead_score for x in a5])) if a5 else np.nan,
        'pf_med5':float(np.median([x.hist_pf for x in a5])) if a5 else np.nan,
        'wr_med5':float(np.median([x.hist_wr for x in a5])) if a5 else np.nan,
    }

feat=[]
for r in orders.itertuples(index=False):
    f=route_features(r.signal_date,r.group)
    feat.append(f)
for k in feat[0]: orders[k]=[x[k] for x in feat]

# Pre-registered architecture variants. These are structural hypotheses, not a threshold sweep:
# 1) persistence: avoid one-day peer leadership flashes.
# 2) persistent_flow: also require the route's already-completed historical playbook edge to remain positive.
# 3) concentrated_leader: allocate only to the strongest persistent route each signal day.
variants={}
variants['base_peer']=orders.copy()
variants['persistence']=orders[(orders.persist5>=3)&(orders.persist10>=5)].copy()
variants['persistent_flow']=orders[(orders.persist5>=3)&(orders.persist10>=5)&(orders.hist_n>=20)&(orders.hist_pf>=1.10)&(orders.hist_wr>=0.48)].copy()

z=variants['persistent_flow'].copy()
if not z.empty:
    z['route_conviction']=z.lead_score + .08*np.minimum(z.hist_pf,3) + .20*z.hist_wr
    keep=[]
    for d,x in z.groupby('signal_date'):
        best=x.sort_values(['route_conviction','candidate_score'],ascending=False).iloc[0]
        best_group=best.group
        keep.extend(x[x.group==best_group].sort_values('candidate_score',ascending=False).head(2).index.tolist())
    z=z.loc[sorted(set(keep))].copy()
variants['concentrated_leader']=z

for name,z in variants.items():
    z.to_csv(OUT/f'orders_{name}.csv',index=False)


def mark(d,c,field,fallback):
    r=px_idx.get((int(d),c)); v=getattr(r,field,np.nan) if r else np.nan
    return float(v) if np.isfinite(v) and v>0 else fallback


def simulate(src,start,end,label,max_pos=4,slot=.24):
    days=[d for d in calendar if start<=d<=end]
    entry_map=defaultdict(list)
    for r in src[(src.entry_date>=start)&(src.entry_date<=end)].itertuples(index=False):
        entry_map[int(r.entry_date)].append(r)
    cash=INIT; pos={}; trades=[]; navrows=[]; min_cash=INIT
    for d in days:
        for c in list(pos):
            p=pos[c]
            if p['exit_date']!=d:continue
            proceeds=p['shares']*p['exit_price']*(1-FEE-TAX); cash+=proceeds
            trades.append({**p,'exit_fill_date':d,'realized_return':proceeds/p['cost']-1,'proceeds':proceeds}); del pos[c]
        open_value=sum(p['shares']*mark(d,c,'open',p['entry_price']) for c,p in pos.items())
        nav_open=cash+open_value
        for r in sorted(entry_map.get(d,[]),key=lambda x:(getattr(x,'lead_score',0),getattr(x,'candidate_score',0)),reverse=True):
            if r.code in pos or len(pos)>=max_pos:continue
            budget=min(nav_open*slot,max(0.,nav_open*MAX_EXPOSURE-open_value),cash)
            adv_shares=int(max(0,math.floor(float(r.vol20)*ADV_CAP)))
            shares=min(int(budget//(float(r.entry_price)*(1+FEE))),adv_shares)
            if shares<=0:continue
            cost=shares*float(r.entry_price)*(1+FEE)
            if cost>cash+1e-8:continue
            cash-=cost; min_cash=min(min_cash,cash); open_value+=shares*float(r.entry_price)
            pos[r.code]={'code':r.code,'group':r.group,'playbook':r.playbook,'quality':r.quality,
                         'signal_date':int(r.signal_date),'entry_date':int(r.entry_date),'exit_date':int(r.exit_date),
                         'entry_price':float(r.entry_price),'exit_price':float(r.exit_price),'shares':shares,'cost':cost,
                         'hist_n':int(r.hist_n),'hist_wr':float(r.hist_wr),'hist_pf':float(r.hist_pf),
                         'persist5':int(r.persist5),'persist10':int(r.persist10)}
        mv=sum(p['shares']*mark(d,c,'close',p['entry_price']) for c,p in pos.items()); nav=cash+mv
        navrows.append({'date':d,'cash':cash,'market_value':mv,'nav':nav,'positions':len(pos),'exposure':mv/nav if nav>0 else 0})
        if cash < -1e-6: raise AssertionError(('negative cash',d,cash))
    last=days[-1]
    for c in list(pos):
        p=pos[c]; xp=floor_tick(mark(last,c,'close',p['entry_price'])*.995)
        proceeds=p['shares']*xp*(1-FEE-TAX); cash+=proceeds
        trades.append({**p,'exit_fill_date':last,'realized_return':proceeds/p['cost']-1,'proceeds':proceeds,'forced_end':True}); del pos[c]
    td=pd.DataFrame(trades); nd=pd.DataFrame(navrows)
    end_nav=float(cash); years=max((pd.Timestamp(str(end))-pd.Timestamp(str(start))).days/365.25,1/252)
    rr=td.realized_return.astype(float) if not td.empty else pd.Series(dtype=float)
    mdd=float((nd.nav/nd.nav.cummax()-1).min()) if not nd.empty else 0.
    s={'label':label,'start':start,'end':end,'start_nav':INIT,'end_nav':end_nav,'return':end_nav/INIT-1,
       'cagr':(end_nav/INIT)**(1/years)-1,'max_dd':mdd,'trades':int(len(td)),'wins':int((rr>0).sum()),
       'win_rate':float((rr>0).mean()) if len(rr) else 0.,'pf':pf_of(rr),'avg_trade':float(rr.mean()) if len(rr) else 0.,
       'min_cash':float(min_cash)}
    return s,td,nd

rows=[]; yearly=[]
for name,src in variants.items():
    for a,b,lab in [(START,DEV_END,'dev'),(HOLDOUT_START,END,'holdout'),(START,END,'full')]:
        max_pos=2 if name=='concentrated_leader' else 4
        slot=.45 if name=='concentrated_leader' else .24
        s,td,nd=simulate(src,a,b,f'{name}_{lab}',max_pos=max_pos,slot=slot); s['variant']=name; s['segment']=lab; rows.append(s)
        td.to_csv(OUT/f'trades_{name}_{lab}.csv',index=False); nd.to_csv(OUT/f'nav_{name}_{lab}.csv',index=False)
    for y in [2023,2024,2025]:
        a=max(START,y*10000+101); b=min(END,y*10000+1231)
        if a>b:continue
        s,_,_=simulate(src,a,b,f'{name}_{y}',max_pos=(2 if name=='concentrated_leader' else 4),slot=(.45 if name=='concentrated_leader' else .24))
        s['variant']=name; s['year']=y; yearly.append(s)

summary=pd.DataFrame(rows); summary.to_csv(OUT/'portfolio_summary.csv',index=False)
pd.DataFrame(yearly).to_csv(OUT/'yearly_summary.csv',index=False)

# Holdout is evidence only. No variant is modified after seeing this table.
h=summary[summary.segment=='holdout'].copy()
h['stable_positive']=(h['return']>0)&(h.pf>1.10)&(h.trades>=25)&(h.win_rate>0.50)
# Primary target is reported but never forced through small samples.
h['primary_target']=(h.cagr>=.50)&(h.win_rate>=.70)&(h.trades>=25)
rank=h.sort_values(['stable_positive','cagr','pf','win_rate'],ascending=False)
best=rank.iloc[0].to_dict() if len(rank) else {}
decision={'architecture':'causal_peer_persistence','holdout_untouched':True,'minute_gate_open':bool(best.get('stable_positive',False)),
          'primary_target_met':bool(best.get('primary_target',False)),'best_holdout':best,
          'notes':['peer groups fitted on prior 120 trading days and frozen monthly','persistence uses only prior route observations within current frozen peer map','quality remains market/liquidity quality, not business fundamentals','no minute optimization performed in this batch']}
json.dump(decision,open(OUT/'decision.json','w'),indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
print(summary.to_string(index=False)); print(pd.DataFrame(yearly).to_string(index=False)); print(json.dumps(decision,indent=2,default=str))
