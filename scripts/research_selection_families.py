"""Research-only causal stock-selection family discovery.

Purpose: test genuinely distinct cross-sectional selection logics before spending
minute-data quota. Signals use T-close information only. Outcomes start from T+1
open and are diagnostic event returns, not yet a common-cash portfolio claim.
Promotion requires non-tiny dev + holdout evidence and neighboring holding windows.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('research_out_selection_families'); OUT.mkdir(exist_ok=True)
PX=Path('formal_run/ohlcv_causal_2020_2025.csv.gz')
INST=Path('data/history/2020-2025/institutional_2020_2025.parquet')
if not PX.exists(): raise FileNotFoundError(PX)
if not INST.exists(): raise FileNotFoundError(INST)
DEV_END=20231231; START=20210101; END=20251231
HOLDS=[10,20,30]
TOPK=[1,3,5]

px=pd.read_csv(PX,dtype={'code':str},low_memory=False)
px['code']=px.code.astype(str).str.zfill(4); px['date']=px.date.astype(int)
px=px.sort_values(['code','date']).reset_index(drop=True)
px['amt']=px.close*px.volume
valid=px.code.str.fullmatch(r'[1-9]\d{3}') & ~px.name.astype(str).str.contains('KY',case=False,na=False)
px['valid']=valid
g=px.groupby('code',group_keys=False)
for w in (5,10,20,40,60,120):
    px[f'r{w}']=g.aclose.transform(lambda s,w=w:s.pct_change(w))
for w in (20,60,120):
    px[f'ma{w}']=g.aclose.transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
px['avgamt20']=g.amt.transform(lambda s:s.rolling(20,min_periods=20).mean())
px['amt5']=g.amt.transform(lambda s:s.rolling(5,min_periods=5).mean())
px['amt_ratio']=px.amt5/px.avgamt20
px['prior_high20']=g.aclose.transform(lambda s:s.shift(1).rolling(20,min_periods=20).max())
px['prior_high60']=g.aclose.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max())
px['high20_break']=px.aclose/px.prior_high20-1
px['near60']=px.aclose/px.prior_high60
px['dist_ma20']=px.aclose/px.ma20-1
px['vol20']=g.causal_ret.transform(lambda s:s.rolling(20,min_periods=20).std())
px['vol60']=g.causal_ret.transform(lambda s:s.rolling(60,min_periods=60).std())
rng=(px.high-px.low).replace(0,np.nan)
px['clv']=((2*px.close-px.high-px.low)/rng).fillna(0).clip(-1,1)
px['clv_amt']=px.clv*px.amt
px['clvflow10']=g.clv_amt.transform(lambda s:s.rolling(10,min_periods=10).sum())/g.amt.transform(lambda s:s.rolling(10,min_periods=10).sum())

inst=pd.read_parquet(INST); inst['code']=inst.code.astype(str).str.zfill(4)
if pd.api.types.is_datetime64_any_dtype(inst.date): inst['date']=inst.date.dt.strftime('%Y%m%d').astype(int)
else: inst['date']=pd.to_datetime(inst.date).dt.strftime('%Y%m%d').astype(int)
inst=inst.sort_values(['code','date']); ig=inst.groupby('code',group_keys=False)
inst['f5']=ig.foreign_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
inst['f20']=ig.foreign_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
inst['t5']=ig.trust_net.transform(lambda s:s.rolling(5,min_periods=5).sum())
inst['t20']=ig.trust_net.transform(lambda s:s.rolling(20,min_periods=20).sum())
px=px.merge(inst[['date','code','f5','f20','t5','t20']],on=['date','code'],how='left')
for c in ['f5','f20','t5','t20']: px[c]=px[c].fillna(0)

# 0050 relative strength and market gate; all values known at T close.
bm=px[px.code=='0050'][['date','r20','r60','aclose','ma60','ma120']].drop_duplicates('date').rename(columns={'r20':'m20','r60':'m60','aclose':'mclose','ma60':'mma60','ma120':'mma120'})
px=px.merge(bm,on='date',how='left')
px['rs20']=px.r20-px.m20; px['rs60']=px.r60-px.m60
px['risk_on']=(px.mclose>px.mma60)&(px.mma60>px.mma120)&(px.m20>0)

# T+1 open and forward exits; raw execution prices are used for event-return diagnostics.
g=px.groupby('code',group_keys=False)
px['entry_open']=g.open.shift(-1)
for h in HOLDS: px[f'exit_close_{h}']=g.close.shift(-(h+1))

def pct_rank(s): return s.rank(pct=True,method='average')
rank_cols=['r10','r20','r40','r60','rs20','rs60','amt_ratio','f5','f20','t5','t20','clvflow10','vol20','near60']
for c in rank_cols: px[f'p_{c}']=px.groupby('date')[c].transform(pct_rank)

base=(px.valid & (px.date>=START)&(px.date<=END)&(px.avgamt20>=30_000_000)&(px.close>=10)&(px.close<=5000)&px.risk_on & (px.aclose>px.ma60)&(px.ma60>px.ma120))

# Distinct economically interpretable families. No grid search; three broad variants each.
def make_score(family,variant):
    q=px
    if family=='FLOW_BREAKOUT':
        score=.25*q.p_rs20+.20*q.p_r20+.20*q.p_f20+.10*q.p_t20+.15*q.p_amt_ratio+.10*q.p_clvflow10
        mask=base & (q.high20_break>=(-.01+.005*variant)) & (q.near60>=.92)
    elif family=='PERSISTENT_MOMENTUM':
        score=.18*q.p_r10+.22*q.p_r20+.20*q.p_r40+.15*q.p_r60+.15*q.p_rs60+.10*q.p_amt_ratio
        mask=base & (q.r10>0)&(q.r20>0)&(q.r40>0)&(q.r60>0) & (q.rs20>(-.01+.01*variant))
    elif family=='PULLBACK_RS_FLOW':
        pull=(1-(q.dist_ma20.abs()/(.10-.015*variant))).clip(0,1)
        score=.25*q.p_rs60+.20*q.p_rs20+.15*q.p_r60+.15*pull+.15*q.p_f20+.10*q.p_t20
        mask=base & q.dist_ma20.between(-.04+.01*variant,.08-.01*variant) & (q.r60>0)
    elif family=='INSTITUTIONAL_ACCEL':
        facc=q.p_f5-q.p_f20; tacc=q.p_t5-q.p_t20
        score=.25*q.p_rs20+.18*q.p_r20+.22*q.p_f5+.13*q.p_t5+.12*q.p_amt_ratio+.10*q.p_clvflow10 + .08*facc.clip(-1,1)+.05*tacc.clip(-1,1)
        mask=base & (q.f5>0)&(q.f20>0)&(q.p_f5>=.65+.05*variant)
    elif family=='LOWVOL_RS_TREND':
        score=.28*q.p_rs60+.22*q.p_rs20+.16*q.p_r20+.14*q.p_r60+.12*(1-q.p_vol20)+.08*q.p_amt_ratio
        mask=base & (q.p_vol20<=.65-.08*variant)&(q.rs60>0)
    elif family=='FRESH_HIGH_ABSORB':
        score=.25*q.p_rs20+.18*q.p_r20+.17*q.p_amt_ratio+.15*q.p_clvflow10+.15*q.p_f5+.10*q.p_t5
        mask=base & q.near60.ge(.94+.015*variant) & q.high20_break.ge(-.015+.0075*variant) & q.clvflow10.gt(0)
    else: raise ValueError(family)
    return score,mask

families=['FLOW_BREAKOUT','PERSISTENT_MOMENTUM','PULLBACK_RS_FLOW','INSTITUTIONAL_ACCEL','LOWVOL_RS_TREND','FRESH_HIGH_ABSORB']
rows=[]
for fam in families:
  for v in range(3):
    score,mask=make_score(fam,v)
    z=px.loc[mask,['date','code','entry_open']+[f'exit_close_{h}' for h in HOLDS]].copy(); z['score']=score[mask]
    z=z.replace([np.inf,-np.inf],np.nan).dropna(subset=['entry_open','score']); z=z[z.entry_open>0]
    z['rank']=z.groupby('date').score.rank(method='first',ascending=False)
    for k in TOPK:
      a=z[z['rank']<=k].copy()
      # Sample weekly signal days only to reduce overlapping pseudo-replication.
      dates=sorted(a.date.unique()); chosen=set(dates[::5]); a=a[a.date.isin(chosen)]
      for h in HOLDS:
        b=a.dropna(subset=[f'exit_close_{h}']).copy()
        # Realistic round trip costs: buy fee + sell fee + 0.3% tax.
        b['ret']=(b[f'exit_close_{h}']*(1-.000855-.003))/(b.entry_open*(1+.000855))-1
        for split,sg in [('dev',b[b.date<=DEV_END]),('holdout',b[b.date>DEV_END]),('all',b)]:
          if len(sg)==0: continue
          rows.append({'family':fam,'variant':v,'topk':k,'hold':h,'split':split,'n':len(sg),'wins':int((sg.ret>0).sum()),'win_rate':float((sg.ret>0).mean()),'mean_return':float(sg.ret.mean()),'median_return':float(sg.ret.median()),'worst_return':float(sg.ret.min()),'p10_return':float(sg.ret.quantile(.10))})

s=pd.DataFrame(rows); s.to_csv(OUT/'selection_family_summary.csv',index=False)
prom=[]
for (fam,v,k),q in s.groupby(['family','variant','topk']):
    good=[]
    for h in HOLDS:
        d=q[(q.hold==h)&(q.split=='dev')]; o=q[(q.hold==h)&(q.split=='holdout')]
        ok=(len(d)==1 and len(o)==1 and d.iloc[0].n>=30 and o.iloc[0].n>=30 and d.iloc[0].win_rate>=.65 and o.iloc[0].win_rate>=.65 and d.iloc[0].mean_return>0 and o.iloc[0].mean_return>0)
        if ok: good.append(h)
    stable=any(abs(a-b)<=10 for a in good for b in good if a!=b)
    prom.append({'family':fam,'variant':int(v),'topk':int(k),'good_holds':good,'neighbor_stable':stable,'promote_to_portfolio':len(good)>=2 and stable})
pd.DataFrame(prom).to_csv(OUT/'promotion.csv',index=False)
# Rank holdout evidence without pretending event study is a portfolio CAGR.
h=s[s.split=='holdout'].copy(); h['quality']=h.win_rate.clip(0,1)*np.maximum(h.mean_return,0)*np.log1p(h.n)
h=h.sort_values(['quality','n'],ascending=False); h.head(50).to_csv(OUT/'holdout_leaderboard.csv',index=False)
payload={'status':'PASS','families':families,'variants_each':3,'topk':TOPK,'holds':HOLDS,'dev_end':DEV_END,'tested_cells':int(len(s)),'promotions':[x for x in prom if x['promote_to_portfolio']],'best_holdout':h.head(10).to_dict('records'),'note':'Diagnostic selection event study only. No CAGR claim until full common-cash T+1 portfolio simulation.'}
(OUT/'decision.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2)); print(h.head(25).to_string(index=False))
