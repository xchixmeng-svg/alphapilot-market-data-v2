from __future__ import annotations
import bisect, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BASE_EV=Path('research_out_continuous_context')
OUT=Path('research_out_causal_peer_rotation'); OUT.mkdir(exist_ok=True)
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
INIT=1_300_000.0; FEE=.000855; TAX=.003; MAX_POS=4; SLOT=.24; MAX_EXPOSURE=.95; ADV_CAP=.02
LOOKBACK=120; MIN_OBS=80; TOPK=4; MIN_GROUP=4; TOP_GROUPS=3; MIN_PB=14


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

px=pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz',dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4); px=px.sort_values(['code','date']).reset_index(drop=True)
inst=pd.read_parquet('formal_run/institutional_2020_2025.parquet'); inst['code']=inst.code.astype(str).str.zfill(4)
if pd.api.types.is_datetime64_any_dtype(inst.date): inst['date']=inst.date.dt.strftime('%Y%m%d').astype(int)
else: inst['date']=pd.to_datetime(inst.date).dt.strftime('%Y%m%d').astype(int)
inst=inst.sort_values(['code','date'])
px['amount']=px.close*px.volume
g=px.groupby('code',group_keys=False)
px['ret1']=g.aclose.pct_change(); px['ret20']=g.aclose.pct_change(20)
px['ma20']=g.aclose.transform(lambda s:s.rolling(20,min_periods=20).mean())
px['amount20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean())
px['vol20']=g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean())
gi=inst.groupby('code',group_keys=False)
inst['inst5']=gi.foreign_net.transform(lambda s:s.rolling(5,min_periods=5).sum())+gi.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
inst['inst20']=gi.foreign_net.transform(lambda s:s.rolling(20,min_periods=20).sum())+gi.trust_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
px=px.merge(inst[['date','code','inst5','inst20']],on=['date','code'],how='left'); px[['inst5','inst20']]=px[['inst5','inst20']].fillna(0.)
px['flow_ratio']=px.inst20*px.close/(px.amount20*20).replace(0,np.nan)
valid=px.code.str.fullmatch(r'[1-9]\d{3}') & (px.amount20>=50_000_000)
st=px[valid].copy()

# Candidate events were generated without peer/sector routing and use only T-close signals -> T+1 fills.
ev=pd.read_csv(BASE_EV/'candidate_events.csv',dtype={'code':str}); ev['code']=ev.code.astype(str).str.zfill(4)
liq=px[['date','code','vol20']].rename(columns={'date':'signal_date'})
ev=ev.merge(liq,on=['signal_date','code'],how='left'); ev=ev[np.isfinite(ev.vol20)&(ev.vol20>0)].copy()

# Corporate-action crossing exclusion, same conservative research convention as prior router.
def as_bool(x):
    if isinstance(x,str): return x.strip().lower() in {'1','true','t','yes','y'}
    return bool(x) if pd.notna(x) else False
action_rows=px[px.is_official_event.map(as_bool)][['code','date']] if 'is_official_event' in px.columns else pd.DataFrame(columns=['code','date'])
action_dates={c:sorted(q.date.astype(int).tolist()) for c,q in action_rows.groupby('code')}
def crosses_action(code,start,end):
    a=action_dates.get(code,[]); i=bisect.bisect_left(a,int(start)); return i<len(a) and a[i]<=int(end)
ev=ev[[not crosses_action(r.code,r.entry_date,r.exit_date) for r in ev.itertuples(index=False)]].copy()

calendar=sorted(int(x) for x in px[(px.date>=START)&(px.date<=END)].date.unique())
all_dates=sorted(int(x) for x in px.date.unique()); date_pos={d:i for i,d in enumerate(all_dates)}

# Fit peer graph only at month boundaries from PRIOR observations; freeze mapping through month.
month_first={}
for d in calendar:
    k=str(d)[:6]; month_first.setdefault(k,d)
fit_dates=sorted(month_first.values())

def make_peer_map(fit_date):
    j=date_pos.get(fit_date)
    if j is None or j<LOOKBACK:return {},{}
    hist_dates=all_dates[max(0,j-LOOKBACK):j]  # excludes fit_date
    q=st[st.date.isin(hist_dates)].copy()
    coverage=q.groupby('code').date.nunique(); codes=coverage[coverage>=MIN_OBS].index.tolist()
    if len(codes)<40:return {},{}
    q=q[q.code.isin(codes)]
    ret=q.pivot(index='date',columns='code',values='ret1').reindex(hist_dates)
    flow=q.pivot(index='date',columns='code',values='flow_ratio').reindex(hist_dates)
    # Standardize each stock using only this historical fit window.
    rz=(ret-ret.mean())/ret.std().replace(0,np.nan); fz=(flow-flow.mean())/flow.std().replace(0,np.nan)
    # Similarity blends price and institutional-flow co-movement. No semantic industry labels.
    rc=rz.corr(min_periods=MIN_OBS); fc=fz.corr(min_periods=max(40,MIN_OBS//2))
    sim=.75*rc.fillna(0)+.25*fc.fillna(0); np.fill_diagonal(sim.values,-np.inf)
    nbr={c:set(sim[c].nlargest(TOPK).index) for c in sim.columns}
    adj={c:set() for c in sim.columns}
    for c in sim.columns:
        for n in nbr[c]:
            if c in nbr.get(n,set()) and sim.loc[c,n]>0: adj[c].add(n); adj[n].add(c)
    seen=set(); groups=[]
    for c in sorted(adj):
        if c in seen:continue
        stack=[c]; comp=[]; seen.add(c)
        while stack:
            x=stack.pop(); comp.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); stack.append(y)
        if len(comp)>=MIN_GROUP: groups.append(sorted(comp))
    code_to_group={}
    meta={}
    for i,comp in enumerate(groups):
        gid=f'P{i:03d}'
        for c in comp: code_to_group[c]=gid
        meta[gid]={'n':len(comp),'codes':comp}
    return code_to_group,meta

peer_maps={}; peer_meta={}
for fd in fit_dates:
    m=str(fd)[:6]; peer_maps[m],peer_meta[m]=make_peer_map(fd)

def month_key(d): return str(int(d))[:6]

def group_snapshot(d):
    mp=peer_maps.get(month_key(d),{})
    q=st[st.date==d].copy(); q['group']=q.code.map(mp); q=q[q.group.notna()]
    if q.empty:return pd.DataFrame()
    rows=[]
    for gid,x in q.groupby('group'):
        if len(x)<MIN_GROUP:continue
        # All inputs known at T close. Leadership combines breadth, persistence, relative strength and flow.
        rows.append({'group':gid,'n':len(x),'r20':float(x.ret20.median()),'breadth':float((x.aclose>x.ma20).mean()),
                     'flow_breadth':float((x.inst5>0).mean()),'flow_med':float(x.flow_ratio.replace([np.inf,-np.inf],np.nan).median()),
                     'liq':float(x.amount20.sum())})
    z=pd.DataFrame(rows)
    if z.empty:return z
    for c in ['r20','breadth','flow_breadth','flow_med']:
        z[c+'_rank']=z[c].rank(pct=True)
    z['lead_score']=.35*z.r20_rank+.25*z.breadth_rank+.25*z.flow_breadth_rank+.15*z.flow_med_rank
    return z.sort_values('lead_score',ascending=False)

# Preserve required order: context/peer rotation -> favored group -> playbook -> stock screen -> quality tier.
def choose_routes(d):
    snap=group_snapshot(d)
    if snap.empty:return []
    favored=snap.head(TOP_GROUPS)
    mp=peer_maps.get(month_key(d),{})
    routes=[]
    for gr in favored.itertuples(index=False):
        codes=[c for c,gid in mp.items() if gid==gr.group]
        # Only trades fully completed before d may inform playbook choice.
        hist=ev[(ev.code.isin(codes))&(ev.exit_date<d)&(ev.signal_date<d)].copy()
        if hist.empty:continue
        # Stage 1: playbook selected using loose market/liquidity quality only.
        loose=hist[hist.quality=='loose']; pbs=[]
        for pb,x in loose.groupby('playbook'):
            if len(x)<MIN_PB:continue
            r=x.return_net.astype(float); pf=pf_of(r); mean=float(r.mean()); wr=float((r>0).mean()); med=float(r.median())
            if mean<=0 or pf<=1.05:continue
            score=2.0*wr+8.0*mean+1.0*med+.15*min(pf,3)
            pbs.append((score,pb,len(x),wr,pf,mean))
        if not pbs:continue
        score,pb,n,wr,pf,mean=max(pbs)
        # Stage 2: quality is explicitly market/liquidity quality, NOT fundamentals.
        same=hist[hist.playbook==pb]; qs=[]
        for qt,x in same.groupby('quality'):
            if len(x)<10:continue
            r=x.return_net.astype(float); qpf=pf_of(r); qmean=float(r.mean()); qwr=float((r>0).mean())
            if qmean<=0 or qpf<=1.02:continue
            qs.append((2*qwr+8*qmean+.15*min(qpf,3),qt,len(x),qwr,qpf,qmean))
        if not qs:continue
        qscore,qt,qn,qwr,qpf,qmean=max(qs)
        routes.append({'group':gr.group,'lead_score':float(gr.lead_score),'playbook':pb,'quality':qt,
                       'hist_n':int(n),'hist_wr':wr,'hist_pf':pf,'hist_mean':mean,'quality_n':int(qn),'quality_wr':qwr,'quality_pf':qpf})
    return routes

orders=[]; route_rows=[]
for d in calendar:
    routes=choose_routes(d); mp=peer_maps.get(month_key(d),{})
    for r in routes:
        route_rows.append({'date':d,**r})
        codes=[c for c,gid in mp.items() if gid==r['group']]
        q=ev[(ev.signal_date==d)&(ev.code.isin(codes))&(ev.playbook==r['playbook'])&(ev.quality==r['quality'])].copy()
        if q.empty:continue
        for x in q.sort_values('score',ascending=False).head(3).itertuples(index=False):
            orders.append({'signal_date':d,'entry_date':int(x.entry_date),'exit_date':int(x.exit_date),'code':x.code,'group':r['group'],
                           'playbook':r['playbook'],'quality':r['quality'],'candidate_score':float(x.score),'entry_price':float(x.entry_price),
                           'exit_price':float(x.exit_price),'vol20':float(x.vol20),'lead_score':r['lead_score'],'hist_n':r['hist_n'],'hist_wr':r['hist_wr'],'hist_pf':r['hist_pf']})
orders=pd.DataFrame(orders); pd.DataFrame(route_rows).to_csv(OUT/'routes.csv',index=False)
if orders.empty: orders=pd.DataFrame(columns=['signal_date','entry_date','exit_date','code','group','playbook','quality','candidate_score','entry_price','exit_price','vol20','lead_score','hist_n','hist_wr','hist_pf'])
orders.to_csv(OUT/'orders.csv',index=False)

px_idx={(int(r.date),r.code):r for r in px.itertuples(index=False)}
def mark(d,c,field,fallback):
    r=px_idx.get((int(d),c)); v=getattr(r,field,np.nan) if r else np.nan
    return float(v) if np.isfinite(v) and v>0 else fallback

def simulate(start,end,label):
    days=[d for d in calendar if start<=d<=end]; entry_map=defaultdict(list)
    for r in orders[(orders.entry_date>=start)&(orders.entry_date<=end)].itertuples(index=False): entry_map[int(r.entry_date)].append(r)
    cash=INIT; pos={}; trades=[]; navrows=[]; min_cash=INIT
    for d in days:
        for c in list(pos):
            p=pos[c]
            if p['exit_date']!=d:continue
            proceeds=p['shares']*p['exit_price']*(1-FEE-TAX); cash+=proceeds
            trades.append({**p,'exit_fill_date':d,'realized_return':proceeds/p['cost']-1,'proceeds':proceeds}); del pos[c]
        open_value=sum(p['shares']*mark(d,c,'open',p['entry_price']) for c,p in pos.items()); nav_open=cash+open_value
        exposure=open_value/nav_open if nav_open>0 else 0
        for r in sorted(entry_map.get(d,[]),key=lambda x:(x.lead_score,x.candidate_score),reverse=True):
            if r.code in pos or len(pos)>=MAX_POS:continue
            budget=min(nav_open*SLOT,max(0.,nav_open*MAX_EXPOSURE-open_value),cash)
            adv_shares=int(max(0,math.floor(r.vol20*ADV_CAP))); shares=min(int(budget//(r.entry_price*(1+FEE))),adv_shares)
            if shares<=0:continue
            cost=shares*r.entry_price*(1+FEE)
            if cost>cash+1e-8:continue
            cash-=cost; min_cash=min(min_cash,cash); open_value+=shares*r.entry_price; exposure=open_value/nav_open if nav_open>0 else 0
            pos[r.code]={'code':r.code,'group':r.group,'playbook':r.playbook,'quality':r.quality,'signal_date':r.signal_date,'entry_date':r.entry_date,
                         'exit_date':r.exit_date,'entry_price':r.entry_price,'exit_price':r.exit_price,'shares':shares,'cost':cost,'hist_n':r.hist_n,'hist_wr':r.hist_wr,'hist_pf':r.hist_pf}
        mv=sum(p['shares']*mark(d,c,'close',p['entry_price']) for c,p in pos.items()); nav=cash+mv
        navrows.append({'date':d,'cash':cash,'market_value':mv,'nav':nav,'positions':len(pos),'exposure':mv/nav if nav>0 else 0})
        if cash<-1e-6: raise AssertionError(('negative cash',d,cash))
    # liquidate remaining on final close conservatively
    last=days[-1]
    for c in list(pos):
        p=pos[c]; xp=floor_tick(mark(last,c,'close',p['entry_price'])*.995); proceeds=p['shares']*xp*(1-FEE-TAX); cash+=proceeds
        trades.append({**p,'exit_fill_date':last,'realized_return':proceeds/p['cost']-1,'proceeds':proceeds,'forced_end':True}); del pos[c]
    td=pd.DataFrame(trades); nd=pd.DataFrame(navrows)
    end_nav=float(cash); years=max((pd.Timestamp(str(end))-pd.Timestamp(str(start))).days/365.25,1/252)
    r=td.realized_return.astype(float) if not td.empty else pd.Series(dtype=float)
    peak=nd.nav.cummax(); mdd=float((nd.nav/peak-1).min()) if not nd.empty else 0.
    s={'label':label,'start':start,'end':end,'start_nav':INIT,'end_nav':end_nav,'return':end_nav/INIT-1,'cagr':(end_nav/INIT)**(1/years)-1,
       'trades':int(len(td)),'wins':int((r>0).sum()),'win_rate':float((r>0).mean()) if len(r) else 0.,'pf':pf_of(r),'max_dd':mdd,'min_cash':float(min_cash)}
    td.to_csv(OUT/f'trades_{label}.csv',index=False); nd.to_csv(OUT/f'nav_{label}.csv',index=False); return s

summ=[simulate(START,DEV_END,'dev'),simulate(HOLDOUT_START,END,'holdout_2025'),simulate(START,END,'full')]
pd.DataFrame(summ).to_csv(OUT/'portfolio_summary.csv',index=False)
# Explicit gate: no minute optimization unless daily peer-rotation layer is positive and nontrivial OOS.
h=[x for x in summ if x['label']=='holdout_2025'][0]
gate=bool(h['return']>0 and h['pf']>1.10 and h['trades']>=25 and h['win_rate']>0.50)
decision={'architecture':'causal_rolling_peer_rotation','fixed_sector_proxy_abandoned':True,'peer_fit':'monthly, prior 120 trading days only, mutual top-4 return/flow graph','fundamental_filter_available':False,
          'quality_semantics':'market/liquidity quality only; not business fundamentals','minute_gate_pass':gate,'target_cagr_50_wr_70_met':bool(h['cagr']>=.50 and h['win_rate']>=.70),
          'holdout':h,'formal_r10_modified':False}
json.dump(decision,open(OUT/'decision.json','w'),indent=2)
print(pd.DataFrame(summ).to_string(index=False)); print(json.dumps(decision,indent=2))
