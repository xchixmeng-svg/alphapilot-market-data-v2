#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
P2=ROOT/'calm_phase2_output'
P3=ROOT/'calm_phase3_output'
OUT=ROOT/'adaptive_router_stage4'; OUT.mkdir(exist_ok=True)
STATES=['TREND_EXPANSION','ROTATION_DISPERSION','RISK_OFF','CHOP_FALSE_BREAKOUT','NEUTRAL_MIXED','WARMUP_UNKNOWN']

p=pd.read_csv(P2/'daily_cross_section_panel.csv').sort_values('date').reset_index(drop=True)
p['year']=p.date.astype(str).str[:4].astype(int)
# Smooth only past/current market structure; no forward columns enter state assignment.
for c in ['breadth_ma60','breadth_r20_pos','median_r20','break60_rate','cross_section_r20_std']:
    p['s_'+c]=p[c].ewm(span=5,adjust=False,min_periods=5).mean()

def causal_pct(s,window=252,minp=126):
    a=s.to_numpy(float); out=np.full(len(a),np.nan)
    for i,v in enumerate(a):
        lo=max(0,i-window); hist=a[lo:i]; hist=hist[np.isfinite(hist)]
        if len(hist)>=minp and np.isfinite(v): out[i]=(hist<=v).mean()
    return out

for c in ['s_breadth_ma60','s_breadth_r20_pos','s_median_r20','s_break60_rate','s_cross_section_r20_std']:
    p['pct_'+c]=causal_pct(p[c])

raw=[]
for _,r in p.iterrows():
    b=r['pct_s_breadth_ma60']; bp=r['pct_s_breadth_r20_pos']; m=r['pct_s_median_r20']; br=r['pct_s_break60_rate']; ds=r['pct_s_cross_section_r20_std']
    if any(pd.isna(x) for x in [b,bp,m,br,ds]): raw.append('WARMUP_UNKNOWN'); continue
    trend=(b>=.70 and bp>=.65 and m>=.65 and br>=.60)
    risk=(b<=.25 and bp<=.30 and m<=.35)
    rotation=(ds>=.70 and b>=.45 and bp>=.45 and m>=.45)
    chop=(br<=.30 and .35<=m<=.65 and b<=.65)
    if risk: s='RISK_OFF'
    elif trend: s='TREND_EXPANSION'
    elif rotation: s='ROTATION_DISPERSION'
    elif chop: s='CHOP_FALSE_BREAKOUT'
    else: s='NEUTRAL_MIXED'
    raw.append(s)
p['candidate_state']=raw

# Causal hysteresis. Ordinary transitions need 3 consecutive candidate days and >=5 days in state.
# Risk-off can override after 2 consecutive days; exiting risk-off needs 3 non-risk candidate days.
final=[]; cur='WARMUP_UNKNOWN'; age=0; pending=None; pending_n=0
for cand in p.candidate_state:
    if cand=='WARMUP_UNKNOWN':
        final.append(cur); age+=1; continue
    if cur=='WARMUP_UNKNOWN':
        cur=cand; age=1; pending=None; pending_n=0; final.append(cur); continue
    if cand==cur:
        age+=1; pending=None; pending_n=0; final.append(cur); continue
    if pending==cand: pending_n+=1
    else: pending=cand; pending_n=1
    need=2 if cand=='RISK_OFF' else 3
    if cur=='RISK_OFF' and cand!='RISK_OFF': need=3
    min_age=0 if cand=='RISK_OFF' else 5
    if pending_n>=need and age>=min_age:
        cur=cand; age=1; pending=None; pending_n=0
    else: age+=1
    final.append(cur)
p['state']=final
p['switch']=(p.state!=p.state.shift(1)) & p.state.ne('WARMUP_UNKNOWN')
p[['date','year','state','candidate_state','switch','breadth_ma60','breadth_r20_pos','median_r20','break60_rate','cross_section_r20_std']].to_csv(OUT/'detector_v2_daily_states.csv',index=False)

# Duration / switching audit.
dur=[]; start=0
for i in range(1,len(p)+1):
    if i==len(p) or p.state.iloc[i]!=p.state.iloc[start]:
        dur.append({'state':p.state.iloc[start],'start_date':int(p.date.iloc[start]),'end_date':int(p.date.iloc[i-1]),'days':i-start})
        start=i
dur=pd.DataFrame(dur); dur.to_csv(OUT/'detector_v2_durations.csv',index=False)
state_days=p[p.year.between(2016,2020)].groupby('state').agg(days=('date','size'),switches=('switch','sum')).reset_index()
state_days=state_days.merge(dur.groupby('state').days.agg(['mean','median']).reset_index(),on='state',how='left')
state_days.to_csv(OUT/'detector_v2_state_stability_2016_2020.csv',index=False)

# Link exact R10 completed trades to signal-date state via filled buy orders; do not alter any R10 order.
def attach(folder,label):
    t=pd.read_csv(folder/'r10max_formal_trades.csv',dtype={'code':str}); o=pd.read_csv(folder/'r10max_formal_orders.csv',dtype={'code':str})
    t.code=t.code.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4); o.code=o.code.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)
    b=o[(o.side=='BUY')&(o.status=='FILLED')][['code','strategy','fill_date','signal_date']].copy()
    b.fill_date=pd.to_numeric(b.fill_date,errors='coerce').astype('Int64'); b.signal_date=pd.to_numeric(b.signal_date,errors='coerce').astype('Int64')
    t.entry_date=pd.to_numeric(t.entry_date,errors='coerce').astype('Int64')
    m=t.merge(b,left_on=['code','strategy','entry_date'],right_on=['code','strategy','fill_date'],how='left')
    m=m.merge(p[['date','state']],left_on='signal_date',right_on='date',how='left'); m['period']=label
    return m

def summ(m):
    z=[]
    for s,g in m.dropna(subset=['state']).groupby('state'):
        gp=g.loc[g.pnl>0,'pnl'].sum(); gl=-g.loc[g.pnl<0,'pnl'].sum()
        z.append({'state':s,'trades':len(g),'win_rate':(g.pnl>0).mean(),'avg_return':g['return'].mean(),'net_pnl':g.pnl.sum(),'pf':gp/gl if gl>0 else np.nan})
    return pd.DataFrame(z)
pre=attach(P3/'phase3_old','2016-2020'); hold=attach(P3/'phase3_new','2021-2025')
pre_s=summ(pre); hold_s=summ(hold); pre_s.to_csv(OUT/'r10_edge_v2_2016_2020.csv',index=False); hold_s.to_csv(OUT/'r10_edge_v2_holdout_audit.csv',index=False)

# Future columns only diagnose whether causal states describe distinct opportunity structures.
diag=p[p.year.between(2016,2020)].groupby('state').agg(days=('date','size'),top_mom_fwd20=('top10pct_r20_fwd20_mean','mean'),breakout_fwd20=('break60_fwd20_mean','mean'),breakout_win=('break60_fwd20_win','mean')).reset_index()
diag.to_csv(OUT/'detector_v2_forward_diagnostic_pre2021.csv',index=False)
manifest={'design':'causal-relative-percentile-detector-v2','market_refetch':False,'formal_r10_modified':False,'state_assignment_uses_forward_data':False,'selection_uses_2021_2025':False,'feature_reference':'rolling previous 252 trading days only; minimum 126','smoothing':'5-day EWM current/past only','hysteresis':'3 consecutive candidate days + 5-day minimum ordinary state age; risk-off override after 2 consecutive days','holdout':'2021-2025 audit only, never used to set rules'}
(OUT/'stage4_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ADAPTIVE_ROUTER_STAGE4_COMPLETE')
print('\nSTABILITY'); print(state_days.to_string(index=False))
print('\nPRE2021_R10_EDGE'); print(pre_s.to_string(index=False))
print('\nHOLDOUT_AUDIT_NO_RETUNE'); print(hold_s.to_string(index=False))
