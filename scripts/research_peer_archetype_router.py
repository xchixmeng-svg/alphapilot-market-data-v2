from __future__ import annotations
import math, json, runpy
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('research_out_peer_archetype_router'); OUT.mkdir(exist_ok=True)
PERSIST=Path('research_out_peer_persistence')
INIT=1_300_000.0; FEE=.000855; TAX=.003; MAX_EXPOSURE=.95; ADV_CAP=.02
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
ARCH=['base_peer','persistent_flow','concentrated_leader']

# Rebuild the causal peer layer to expose frozen peer maps/group snapshots. This script
# never alters the peer fit: prior 120 trading days only, monthly freeze.
ns=runpy.run_path('scripts/research_causal_peer_rotation.py')
group_snapshot=ns['group_snapshot']; calendar=ns['calendar']; px=ns['px']
px_idx={(int(r.date),r.code):r for r in px.itertuples(index=False)}

variants={}
for a in ARCH:
    q=pd.read_csv(PERSIST/f'orders_{a}.csv',dtype={'code':str})
    q['code']=q.code.astype(str).str.zfill(4)
    for c in ['signal_date','entry_date','exit_date']: q[c]=q[c].astype(int)
    q['realized_return_net']=q.exit_price*(1-FEE-TAX)/(q.entry_price*(1+FEE))-1
    variants[a]=q

# Continuous peer/flow context, all known at T close. No bull/bear labels.
ctx=[]; top_hist=[]
for d in calendar:
    s=group_snapshot(d)
    if s.empty:
        ctx.append({'date':d,'route_count':0,'top_lead':0.,'lead_gap':0.,'top_breadth':0.,'top_flow_breadth':0.,'top_flow_med':0.,'lead_conc':0.,'top_persist5':0.,'top_persist10':0.})
        top_hist.append((d,None)); continue
    s=s.sort_values('lead_score',ascending=False).reset_index(drop=True)
    top=s.iloc[0]; gid=top.group; month=str(d)[:6]
    prev5=[g for dd,g in top_hist[-5:] if str(dd)[:6]==month]
    prev10=[g for dd,g in top_hist[-10:] if str(dd)[:6]==month]
    lead_gap=float(top.lead_score-(s.lead_score.iloc[1] if len(s)>1 else 0.))
    lead_conc=float(top.lead_score/max(float(s.lead_score.sum()),1e-9))
    ctx.append({'date':d,'route_count':len(s),'top_lead':float(top.lead_score),'lead_gap':lead_gap,
                'top_breadth':float(top.breadth),'top_flow_breadth':float(top.flow_breadth),'top_flow_med':float(np.nan_to_num(top.flow_med)),
                'lead_conc':lead_conc,'top_persist5':sum(g==gid for g in prev5),'top_persist10':sum(g==gid for g in prev10)})
    top_hist.append((d,gid))
ctx=pd.DataFrame(ctx).set_index('date')
FEATURES=['route_count','top_lead','lead_gap','top_breadth','top_flow_breadth','top_flow_med','lead_conc','top_persist5','top_persist10']

# Monthly walk-forward archetype router. Standardization and historical outcome bank are
# fitted ONLY on information available before the month's first session, then frozen.
month_first={}
for d in calendar: month_first.setdefault(str(d)[:6],d)
month_choice=[]
for month,fit_date in month_first.items():
    hist_ctx=ctx.loc[ctx.index<fit_date].copy()
    if len(hist_ctx)<80:
        month_choice.append({'month':month,'fit_date':fit_date,'archetype':'base_peer','reason':'warmup'}); continue
    mu=hist_ctx[FEATURES].tail(252).mean(); sd=hist_ctx[FEATURES].tail(252).std().replace(0,1).fillna(1)
    cand=ctx.loc[[d for d in calendar if str(d)[:6]==month],FEATURES]
    # Month representative uses first day's context only; decision is frozen for whole month.
    x=((cand.iloc[0]-mu)/sd).fillna(0).clip(-5,5)
    H=((hist_ctx[FEATURES]-mu)/sd).fillna(0).clip(-5,5)
    dist=((H-x)**2).mean(axis=1).pow(.5).sort_values()
    near_days=list(dist.head(45).index)
    scores=[]
    for a in ARCH:
        q=variants[a]
        # Only outcomes fully completed BEFORE fit_date are eligible.
        z=q[(q.signal_date.isin(near_days))&(q.exit_date<fit_date)].copy()
        r=z.realized_return_net.astype(float)
        if len(r)<12: continue
        pos=float(r[r>0].sum()); neg=float(-r[r<0].sum()); pf=pos/neg if neg>0 else (99. if pos>0 else 0.)
        wr=float((r>0).mean()); mean=float(r.mean()); med=float(r.median())
        # Structural ranking, not a threshold sweep. Negative edge is penalized but one archetype is always chosen.
        score=10*mean+2*med+0.35*(wr-.5)+0.08*min(pf,3)
        scores.append((score,a,len(r),mean,wr,pf))
    if not scores:
        pick=('base_peer',0,0,0,0)
        month_choice.append({'month':month,'fit_date':fit_date,'archetype':'base_peer','reason':'insufficient_history'})
    else:
        best=max(scores,key=lambda t:t[0])
        month_choice.append({'month':month,'fit_date':fit_date,'archetype':best[1],'score':best[0],'hist_n':best[2],'hist_mean':best[3],'hist_wr':best[4],'hist_pf':best[5],'reason':'past_similar_context'})
choices=pd.DataFrame(month_choice); choices.to_csv(OUT/'monthly_archetype_choices.csv',index=False)
choice_map=dict(zip(choices.month,choices.archetype))

# Required order is preserved: frozen current peer/flow context -> archetype -> pre-existing
# method-specific stock screen -> market/liquidity quality -> T+1 shared-capital execution.
selected=[]
for d in calendar:
    a=choice_map.get(str(d)[:6],'base_peer')
    q=variants[a]; z=q[q.signal_date==d].copy()
    if z.empty: continue
    z['archetype']=a
    selected.append(z)
orders=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()
orders.to_csv(OUT/'selected_orders.csv',index=False)

def mark(d,c,field,fallback):
    r=px_idx.get((int(d),c)); v=getattr(r,field,np.nan) if r else np.nan
    return float(v) if np.isfinite(v) and v>0 else fallback

def tick(p):
    if p<10:return .01
    if p<50:return .05
    if p<100:return .1
    if p<500:return .5
    if p<1000:return 1.
    return 5.
def floor_tick(p):
    t=tick(p); return math.floor((p+1e-12)/t)*t

def pf_of(r):
    r=pd.Series(r,dtype=float); pos=float(r[r>0].sum()); neg=float(-r[r<0].sum())
    return pos/neg if neg>0 else (99. if pos>0 else 0.)

def simulate(start,end,label):
    days=[d for d in calendar if start<=d<=end]; em=defaultdict(list)
    if not orders.empty:
        for r in orders[(orders.entry_date>=start)&(orders.entry_date<=end)].itertuples(index=False): em[int(r.entry_date)].append(r)
    cash=INIT; pos={}; trades=[]; navrows=[]; min_cash=INIT
    for d in days:
        for c in list(pos):
            p=pos[c]
            if p['exit_date']!=d: continue
            proceeds=p['shares']*p['exit_price']*(1-FEE-TAX); cash+=proceeds
            trades.append({**p,'exit_fill_date':d,'realized_return':proceeds/p['cost']-1}); del pos[c]
        open_value=sum(p['shares']*mark(d,c,'open',p['entry_price']) for c,p in pos.items()); nav_open=cash+open_value
        for r in sorted(em.get(d,[]),key=lambda x:(getattr(x,'lead_score',0),getattr(x,'candidate_score',0)),reverse=True):
            if r.code in pos or len(pos)>=4: continue
            budget=min(nav_open*.24,max(0.,nav_open*MAX_EXPOSURE-open_value),cash)
            adv=int(max(0,math.floor(float(r.vol20)*ADV_CAP))); shares=min(int(budget//(float(r.entry_price)*(1+FEE))),adv)
            if shares<=0: continue
            cost=shares*float(r.entry_price)*(1+FEE)
            if cost>cash+1e-8: continue
            cash-=cost; min_cash=min(min_cash,cash); open_value+=shares*float(r.entry_price)
            pos[r.code]={'code':r.code,'archetype':r.archetype,'signal_date':int(r.signal_date),'entry_date':int(r.entry_date),'exit_date':int(r.exit_date),'entry_price':float(r.entry_price),'exit_price':float(r.exit_price),'shares':shares,'cost':cost}
        mv=sum(p['shares']*mark(d,c,'close',p['entry_price']) for c,p in pos.items()); nav=cash+mv
        navrows.append({'date':d,'cash':cash,'market_value':mv,'nav':nav,'positions':len(pos)})
        if cash < -1e-6: raise AssertionError(('negative cash',d,cash))
    last=days[-1]
    for c in list(pos):
        p=pos[c]; xp=floor_tick(mark(last,c,'close',p['entry_price'])*.995); proceeds=p['shares']*xp*(1-FEE-TAX); cash+=proceeds
        trades.append({**p,'exit_fill_date':last,'realized_return':proceeds/p['cost']-1,'forced_end':True}); del pos[c]
    td=pd.DataFrame(trades); nd=pd.DataFrame(navrows); rr=td.realized_return.astype(float) if not td.empty else pd.Series(dtype=float)
    years=max((pd.Timestamp(str(end))-pd.Timestamp(str(start))).days/365.25,1/252); end_nav=float(cash)
    return {'label':label,'start':start,'end':end,'start_nav':INIT,'end_nav':end_nav,'return':end_nav/INIT-1,'cagr':(end_nav/INIT)**(1/years)-1,
            'max_dd':float((nd.nav/nd.nav.cummax()-1).min()) if not nd.empty else 0.,'trades':len(td),'wins':int((rr>0).sum()),'win_rate':float((rr>0).mean()) if len(rr) else 0.,'pf':pf_of(rr),'min_cash':min_cash},td,nd

rows=[]
for a,b,l in [(START,DEV_END,'dev'),(HOLDOUT_START,END,'holdout_2025'),(START,END,'full')]:
    s,td,nd=simulate(a,b,l); rows.append(s); td.to_csv(OUT/f'trades_{l}.csv',index=False); nd.to_csv(OUT/f'nav_{l}.csv',index=False)
for y in [2023,2024,2025]:
    a=max(START,y*10000+101); b=min(END,y*10000+1231)
    if a<=b: rows.append(simulate(a,b,str(y))[0])
summary=pd.DataFrame(rows); summary.to_csv(OUT/'summary.csv',index=False)
h=summary[summary.label=='holdout_2025'].iloc[0]
decision={'architecture':'monthly_frozen_causal_peer_archetype_router','semantic_bull_bear_gate':False,'holdout_not_used_for_fit':True,
          'quality_is_fundamental':False,'minute_stage_allowed':bool(h['return']>0 and h.pf>1.10 and h.trades>=25 and h.win_rate>.50),
          'primary_target_met':bool(h.cagr>=.50 and h.win_rate>=.70 and h.trades>=25),'formal_r10_modified':False}
json.dump(decision,open(OUT/'decision.json','w'),indent=2)
print(choices.to_string(index=False)); print(summary.to_string(index=False)); print(json.dumps(decision,indent=2))
