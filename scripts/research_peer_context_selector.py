from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path('research_out_peer_persistence')
OUT=Path('research_out_peer_context_selector'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; MAX_EXPOSURE=.95; ADV_CAP=.02
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
VARIANTS=['base_peer','persistence','persistent_flow','concentrated_leader']
LOOKBACK_CTX=120; KNN_DAYS=40; MIN_HIST_ORDERS=18


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
px['ma60']=g.aclose.transform(lambda s:s.rolling(60,min_periods=60).mean())
px['amount20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean())
gi=inst.groupby('code',group_keys=False)
inst['inst5']=gi.foreign_net.transform(lambda s:s.rolling(5,min_periods=5).sum())+gi.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
inst['inst20']=gi.foreign_net.transform(lambda s:s.rolling(20,min_periods=20).sum())+gi.trust_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
px=px.merge(inst[['date','code','inst5','inst20']],on=['date','code'],how='left'); px[['inst5','inst20']]=px[['inst5','inst20']].fillna(0.)
valid=px.code.str.fullmatch(r'[1-9]\d{3}') & (px.amount20>=50_000_000)
st=px[valid].copy()
calendar=sorted(int(x) for x in px[(px.date>=START)&(px.date<=END)].date.unique())
cal_pos={d:i for i,d in enumerate(calendar)}

# Continuous T-close context. No bull/bear labels and no future observations.
ctx=[]
for d,x in st[(st.date>=START)&(st.date<=END)].groupby('date'):
    r20=x.ret20.replace([np.inf,-np.inf],np.nan)
    amt=x.amount20.replace([np.inf,-np.inf],np.nan)
    flow=(x.inst20*x.close/(x.amount20*20).replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    ctx.append({'date':int(d),'median_r20':float(r20.median()),'breadth20':float((x.aclose>x.ma20).mean()),
                'breadth60':float((x.aclose>x.ma60).mean()),'dispersion20':float(r20.std()),
                'flow_breadth':float((x.inst5>0).mean()),'flow_median':float(flow.median()),
                'turnover_conc':float((amt.nlargest(max(1,int(len(amt)*.1))).sum()/amt.sum()) if amt.sum()>0 else np.nan)})
ctx=pd.DataFrame(ctx).sort_values('date').reset_index(drop=True)
routes=pd.read_csv('research_out_causal_peer_rotation/routes.csv')
routes['date']=routes.date.astype(int)
route_agg=[]
for d,x in routes.groupby('date'):
    ls=x.lead_score.astype(float).sort_values(ascending=False).to_numpy()
    route_agg.append({'date':int(d),'active_routes':len(x),'lead_top':float(ls[0]) if len(ls) else np.nan,
                      'lead_gap':float(ls[0]-ls[1]) if len(ls)>1 else float(ls[0]) if len(ls) else np.nan,
                      'hist_pf_med':float(x.hist_pf.median()),'hist_wr_med':float(x.hist_wr.median())})
ctx=ctx.merge(pd.DataFrame(route_agg),on='date',how='left').sort_values('date').reset_index(drop=True)
features=['median_r20','breadth20','breadth60','dispersion20','flow_breadth','flow_median','turnover_conc','active_routes','lead_top','lead_gap','hist_pf_med','hist_wr_med']
# Causal trailing z-scores use T-1 and earlier only.
for c in features:
    s=ctx[c].astype(float)
    mu=s.shift(1).rolling(LOOKBACK_CTX,min_periods=40).mean(); sd=s.shift(1).rolling(LOOKBACK_CTX,min_periods=40).std().replace(0,np.nan)
    ctx[c+'_z']=(s-mu)/sd
zcols=[c+'_z' for c in features]
ctx_idx=ctx.set_index('date')
ctx.to_csv(OUT/'daily_context.csv',index=False)

orders={}
for v in VARIANTS:
    q=pd.read_csv(BASE/f'orders_{v}.csv',dtype={'code':str})
    if q.empty:
        q=pd.DataFrame(columns=['signal_date','entry_date','exit_date','code','group','playbook','quality','candidate_score','entry_price','exit_price','vol20','lead_score','hist_n','hist_wr','hist_pf','persist5','persist10'])
    q['code']=q.code.astype(str).str.zfill(4)
    for c in ['signal_date','entry_date','exit_date']:
        if c in q:q[c]=q[c].astype(int)
    if len(q): q['event_ret']=q.exit_price*(1-FEE-TAX)/(q.entry_price*(1+FEE))-1
    else: q['event_ret']=pd.Series(dtype=float)
    orders[v]=q

# Similar-context selector: on each day, nearest historical context days are drawn only from dates < T.
# Variant scores use only orders fully exited before T. 2025 is never used to tune constants.
def choose_variant(d):
    if d not in ctx_idx.index:return 'base_peer',{}
    cur=ctx_idx.loc[d,zcols].astype(float)
    if cur.isna().sum()>len(zcols)//2:return 'base_peer',{}
    hist=ctx[(ctx.date<d)].copy()
    hist=hist[hist[zcols].notna().sum(axis=1)>=len(zcols)//2]
    if hist.empty:return 'base_peer',{}
    arr=hist[zcols].astype(float)
    diff=(arr-cur).fillna(0.0)
    common=(arr.notna() & cur.notna()).sum(axis=1).clip(lower=1)
    hist['dist']=np.sqrt((diff*diff).sum(axis=1)/common)
    near=set(hist.nsmallest(KNN_DAYS,'dist').date.astype(int).tolist())
    scores=[]; detail={}
    for v,q in orders.items():
        h=q[(q.exit_date<d)&(q.signal_date.isin(near))].copy()
        n=len(h)
        if n<MIN_HIST_ORDERS:continue
        r=h.event_ret.astype(float); wr=float((r>0).mean()); pf=pf_of(r); mean=float(r.mean()); med=float(r.median())
        # Penalize fragile tiny-sample edges; all terms derive from completed past trades only.
        shrink=n/(n+25.0)
        score=shrink*(2.0*wr+8.0*mean+0.15*min(pf,3)+1.0*med)
        detail[v]={'n':n,'wr':wr,'pf':pf,'mean':mean,'median':med,'score':score}
        scores.append((score,v))
    if not scores:return 'base_peer',detail
    return max(scores)[1],detail

selected=[]; decision_rows=[]
for d in calendar:
    v,det=choose_variant(d)
    decision_rows.append({'date':d,'selected_variant':v,'detail_json':json.dumps(det,sort_keys=True)})
    q=orders[v]
    z=q[q.signal_date==d].copy()
    if z.empty:continue
    z['selected_variant']=v
    selected.append(z)
sel=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()
pd.DataFrame(decision_rows).to_csv(OUT/'selector_decisions.csv',index=False)
sel.to_csv(OUT/'selected_orders.csv',index=False)

px_idx={(int(r.date),r.code):r for r in px.itertuples(index=False)}
def mark(d,c,field,fallback):
    r=px_idx.get((int(d),c)); v=getattr(r,field,np.nan) if r else np.nan
    return float(v) if np.isfinite(v) and v>0 else fallback

def simulate(src,start,end,label):
    days=[d for d in calendar if start<=d<=end]; entry_map=defaultdict(list)
    if not src.empty:
        for r in src[(src.entry_date>=start)&(src.entry_date<=end)].itertuples(index=False): entry_map[int(r.entry_date)].append(r)
    cash=INIT; pos={}; trades=[]; navrows=[]; min_cash=INIT
    for d in days:
        for c in list(pos):
            p=pos[c]
            if p['exit_date']!=d:continue
            proceeds=p['shares']*p['exit_price']*(1-FEE-TAX); cash+=proceeds
            trades.append({**p,'exit_fill_date':d,'realized_return':proceeds/p['cost']-1,'proceeds':proceeds}); del pos[c]
        open_value=sum(p['shares']*mark(d,c,'open',p['entry_price']) for c,p in pos.items()); nav_open=cash+open_value
        for r in sorted(entry_map.get(d,[]),key=lambda x:(getattr(x,'lead_score',0),getattr(x,'candidate_score',0)),reverse=True):
            if r.code in pos:continue
            max_pos=2 if getattr(r,'selected_variant','')=='concentrated_leader' else 4
            slot=.45 if max_pos==2 else .24
            if len(pos)>=max_pos:continue
            budget=min(nav_open*slot,max(0.,nav_open*MAX_EXPOSURE-open_value),cash)
            adv_shares=int(max(0,math.floor(float(r.vol20)*ADV_CAP)))
            shares=min(int(budget//(float(r.entry_price)*(1+FEE))),adv_shares)
            if shares<=0:continue
            cost=shares*float(r.entry_price)*(1+FEE)
            if cost>cash+1e-8:continue
            cash-=cost; min_cash=min(min_cash,cash); open_value+=shares*float(r.entry_price)
            pos[r.code]={'code':r.code,'group':r.group,'playbook':r.playbook,'quality':r.quality,'selected_variant':getattr(r,'selected_variant',''),
                         'signal_date':int(r.signal_date),'entry_date':int(r.entry_date),'exit_date':int(r.exit_date),
                         'entry_price':float(r.entry_price),'exit_price':float(r.exit_price),'shares':shares,'cost':cost}
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
       'win_rate':float((rr>0).mean()) if len(rr) else 0.,'pf':pf_of(rr),'avg_trade':float(rr.mean()) if len(rr) else 0.,'min_cash':float(min_cash)}
    return s,td,nd

rows=[]; yearly=[]
for a,b,lab in [(START,DEV_END,'dev'),(HOLDOUT_START,END,'holdout'),(START,END,'full')]:
    s,td,nd=simulate(sel,a,b,'context_selector_'+lab); s['segment']=lab; rows.append(s)
    td.to_csv(OUT/f'trades_{lab}.csv',index=False); nd.to_csv(OUT/f'nav_{lab}.csv',index=False)
for y in [2023,2024,2025]:
    a=max(START,y*10000+101); b=min(END,y*10000+1231)
    s,_,_=simulate(sel,a,b,f'context_selector_{y}'); s['year']=y; yearly.append(s)
summary=pd.DataFrame(rows); summary.to_csv(OUT/'portfolio_summary.csv',index=False)
pd.DataFrame(yearly).to_csv(OUT/'yearly_summary.csv',index=False)
h=summary[summary.segment=='holdout'].iloc[0].to_dict()
stable=bool(h['return']>0 and h['pf']>1.10 and h['trades']>=25 and h['win_rate']>0.50)
primary=bool(h['cagr']>=.50 and h['win_rate']>=.70 and h['trades']>=25)
decision={'architecture':'causal_peer_context_selector','holdout_untouched_or_rolling_oos':True,
          'minute_gate_open':stable,'primary_target_met':primary,'holdout':h,
          'notes':['no fixed bull/bear labels','context z-scores use T-1 and earlier trailing history only','archetype selection uses only fully exited past trades in nearest historical contexts','2025 not used to tune selector constants','quality remains market/liquidity quality, not business fundamentals','no minute optimization in this batch']}
json.dump(decision,open(OUT/'decision.json','w'),indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
print(summary.to_string(index=False)); print(pd.DataFrame(yearly).to_string(index=False)); print(json.dumps(decision,indent=2,default=str))
