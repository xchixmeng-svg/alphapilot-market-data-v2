from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path('research_out_continuous_context')
OUT=Path('research_out_context_cycle'); OUT.mkdir(exist_ok=True)
START=20230523; DEV_END=20241231; HOLDOUT_START=20250101; END=20251231
K=45; MIN_HIST=80; MIN_EVENTS=24

ctx=pd.read_csv(BASE/'daily_continuous_context.csv')
ev=pd.read_csv(BASE/'candidate_events.csv',dtype={'code':str})
ev['code']=ev.code.astype(str).str.zfill(4)
px=pd.read_csv('formal_run/ohlcv_causal_2020_2025.csv.gz',dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4)
px=px.sort_values(['code','date'])
px['r20']=px.groupby('code').aclose.pct_change(20)
px['r60']=px.groupby('code').aclose.pct_change(60)
px['amount']=px.close*px.volume

# Fixed ex-ante code-range proxies only; not hindsight sector assignments.
# Used because no point-in-time official industry map is present in the repository.
PROXIES={
 'financial_proxy': lambda s:s.str.match(r'^28\d\d$'),
 'electronics_proxy': lambda s:s.str.match(r'^(23|24)\d\d$'),
 'transport_proxy': lambda s:s.str.match(r'^26\d\d$'),
 'materials_proxy': lambda s:s.str.match(r'^20\d\d$'),
}
rows=[]
for d,q in px[(px.date>=20220101)&(px.date<=END)].groupby('date'):
    rec={'date':int(d)}
    for name,fn in PROXIES.items():
        z=q[fn(q.code)]
        rec[name+'_r20']=float(z.r20.median()) if len(z) else np.nan
        rec[name+'_r60']=float(z.r60.median()) if len(z) else np.nan
        rec[name+'_breadth20']=float((z.r20>0).mean()) if len(z) else np.nan
        rec[name+'_turn_share']=float(z.amount.sum()/q.amount.sum()) if q.amount.sum()>0 and len(z) else np.nan
    rows.append(rec)
cyc=pd.DataFrame(rows).sort_values('date')

# Causal archetype/sector-cycle persistence: only event outcomes whose exits were already known.
playbooks=sorted(ev.playbook.unique())
for pb in playbooks:
    vals=[]
    for d in cyc.date:
        h=ev[(ev.playbook==pb)&(ev.exit_date<d)&(ev.exit_date>=d-10000)]
        # last ~1 calendar year; tail by completion date for causal recency
        h=h.sort_values('exit_date').tail(120)
        vals.append({
          'mean':float(h.return_net.mean()) if len(h) else np.nan,
          'win':float((h.return_net>0).mean()) if len(h) else np.nan,
          'pf':float(h.loc[h.return_net>0,'return_net'].sum()/(-h.loc[h.return_net<0,'return_net'].sum())) if (h.return_net<0).any() else np.nan,
          'n':int(len(h))})
    cyc[pb+'_trail_mean']=[x['mean'] for x in vals]
    cyc[pb+'_trail_win']=[x['win'] for x in vals]
    cyc[pb+'_trail_pf']=[x['pf'] for x in vals]
    cyc[pb+'_trail_n']=[x['n'] for x in vals]

aug=ctx.merge(cyc,on='date',how='left').sort_values('date').reset_index(drop=True)
cycle_cols=[c for c in aug.columns if c.endswith(('_r20','_r60','_breadth20','_turn_share','_trail_mean','_trail_win','_trail_pf')) and c not in {'etf_r20','etf_r60','median_r20','breadth20'}]
# Transform to rolling z-scores using PRIOR observations only.
for c in cycle_cols:
    mu=aug[c].shift(1).rolling(252,min_periods=60).mean(); sd=aug[c].shift(1).rolling(252,min_periods=60).std().replace(0,np.nan)
    aug[c+'_cz']=((aug[c]-mu)/sd).clip(-5,5)
base_z=[c for c in aug.columns if c.endswith('_z') and not c.endswith('_cz')]
cycle_z=[c+'_cz' for c in cycle_cols]
aug.to_csv(OUT/'daily_context_cycle.csv',index=False)

# Event-level causal router comparison. This is a DAILY/CONTEXT gate only, not portfolio approval.
def vec(row,cols): return row[cols].to_numpy(float)
def dist(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    return np.sqrt(np.mean((a[m]-b[m])**2)) if m.sum()>=max(8,int(.7*len(a))) else np.inf

def evaluate(cols,label):
    dates=aug[(aug.date>=START)&(aug.date<=END)].date.tolist(); amap=aug.set_index('date')
    out=[]
    for d in dates:
        prior=[x for x in aug.date if x<d]
        if len(prior)<MIN_HIST: continue
        cur=vec(amap.loc[d],cols)
        ds=sorted((dist(cur,vec(amap.loc[x],cols)),int(x)) for x in prior)
        nei=[x for di,x in ds if np.isfinite(di)][:K]
        if len(nei)<20: continue
        hist=ev[(ev.signal_date.isin(nei))&(ev.exit_date<d)]
        stats=[]
        for (pb,qt),q in hist.groupby(['playbook','quality']):
            if len(q)<MIN_EVENTS: continue
            r=q.return_net.astype(float); loss=-r[r<0].sum(); pf=r[r>0].sum()/loss if loss>0 else 99
            stats.append((2.2*(r>0).mean()+7*r.mean()+1.5*r.median()+.15*min(pf,3)+1.5*min(r.quantile(.1),0),pb,qt,len(q),r.mean(),(r>0).mean(),pf))
        stats=[x for x in stats if x[4]>0 and x[6]>1.05]
        if not stats: continue
        _,pb,qt,n,hm,hw,hpf=max(stats,key=lambda x:(x[0],x[3]))
        today=ev[(ev.signal_date==d)&(ev.playbook==pb)&(ev.quality==qt)].sort_values('score',ascending=False).head(4)
        for r in today.itertuples(index=False): out.append({'router':label,'signal_date':d,'entry_date':int(r.entry_date),'exit_date':int(r.exit_date),'playbook':pb,'quality':qt,'code':r.code,'return_net':float(r.return_net),'hist_n':int(n),'hist_mean':float(hm),'hist_win':float(hw),'hist_pf':float(hpf)})
    return pd.DataFrame(out)

base=evaluate(base_z,'base_context')
augm=evaluate(base_z+cycle_z,'context_plus_cycle')
allr=pd.concat([base,augm],ignore_index=True); allr.to_csv(OUT/'router_event_results.csv',index=False)

def summary(q,label):
    if q.empty:return {'label':label,'n':0}
    r=q.return_net; loss=-r[r<0].sum(); pf=r[r>0].sum()/loss if loss>0 else 99
    return {'label':label,'n':int(len(q)),'win_rate':float((r>0).mean()),'mean_return':float(r.mean()),'median_return':float(r.median()),'pf':float(pf)}

summ=[]
for router,q in allr.groupby('router'):
    summ.append(summary(q,router+'_full'))
    summ.append(summary(q[q.signal_date<=DEV_END],router+'_dev'))
    summ.append(summary(q[q.signal_date>=HOLDOUT_START],router+'_holdout2025'))
    for y in (2024,2025):
        summ.append(summary(q[(q.signal_date//10000)==y],router+f'_{y}'))
sdf=pd.DataFrame(summ); sdf.to_csv(OUT/'cycle_gate_summary.csv',index=False)

def get(label):
    z=sdf[sdf.label==label]; return z.iloc[0].to_dict() if len(z) else {'label':label,'n':0}
bh=get('base_context_holdout2025'); ch=get('context_plus_cycle_holdout2025')
improved=bool(ch.get('n',0)>=20 and ch.get('mean_return',-9)>bh.get('mean_return',-9) and ch.get('pf',0)>bh.get('pf',0) and ch.get('win_rate',0)>=bh.get('win_rate',0))
decision={
 'status':'PASS',
 'stage':'daily_context_cycle_gate',
 'semantic_market_labels_used':False,
 'cycle_layer':cycle_cols,
 'macro_vintage_policy':'No revised macro series injected. Yield curve/FX/PMI/IP omitted until point-in-time release-vintage data are available; market-implied and fixed ex-ante sector proxies only.',
 'fixed_proxy_note':'28xx financial, 23xx/24xx electronics, 26xx transport, 20xx materials are ex-ante code-range proxies, not hindsight sector labels.',
 'base_holdout':bh,'cycle_holdout':ch,
 'cycle_improves_holdout':improved,
 'minute_stage_allowed':False,
 'reason':'Minute stage remains blocked until daily/context/cycle layer demonstrates stable positive OOS edge with portfolio-level accounting.',
 'formal_r10_untouched':True}
(OUT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
print(sdf.to_string(index=False)); print(json.dumps(decision,ensure_ascii=False,indent=2))