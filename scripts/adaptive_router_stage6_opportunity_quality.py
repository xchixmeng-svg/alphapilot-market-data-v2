#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
P2=ROOT/'calm_phase2_output'; P3=ROOT/'calm_phase3_output'; OUT=ROOT/'adaptive_router_stage6'; OUT.mkdir(exist_ok=True)
p=pd.read_csv(P2/'daily_cross_section_panel.csv').sort_values('date').reset_index(drop=True)
p['year']=p.date.astype(str).str[:4].astype(int)
# These diagnostic labels from old dates become observable only after their horizon has elapsed.
# shift(20) means today's feature uses the fully realized 20D outcome of signals from 20 trading days ago.
p['real_break20']=p.break60_fwd20_mean.shift(20)
p['real_breakwin20']=p.break60_fwd20_win.shift(20)
p['real_topmom20']=p.top10pct_r20_fwd20_mean.shift(20)
p['real_topmom10']=p.top10pct_r20_fwd10_mean.shift(10)
for c in ['real_break20','real_breakwin20','real_topmom20','real_topmom10','breadth_ma60','breadth_r20_pos','median_r20','break60_rate']:
    p['s_'+c]=p[c].ewm(span=5,adjust=False,min_periods=5).mean()

def cpct(s,window=252,minp=126):
    a=s.to_numpy(float); out=np.full(len(a),np.nan)
    for i,v in enumerate(a):
        h=a[max(0,i-window):i]; h=h[np.isfinite(h)]
        if len(h)>=minp and np.isfinite(v): out[i]=(h<=v).mean()
    return out
for c in ['s_real_break20','s_real_breakwin20','s_real_topmom20','s_real_topmom10','s_breadth_ma60','s_breadth_r20_pos','s_median_r20','s_break60_rate']:
    p['pct_'+c]=cpct(p[c])
qcols=['pct_s_real_break20','pct_s_real_breakwin20','pct_s_real_topmom20','pct_s_real_topmom10']
scols=['pct_s_breadth_ma60','pct_s_breadth_r20_pos','pct_s_median_r20','pct_s_break60_rate']
p['quality_score']=p[qcols].mean(axis=1,skipna=False); p['structure_score']=p[scols].mean(axis=1,skipna=False)
raw=[]
for _,r in p.iterrows():
    q=r.quality_score; s=r.structure_score
    if pd.isna(q) or pd.isna(s): st='WARMUP_UNKNOWN'
    elif q>=.70 and s>=.50: st='EDGE_EXPANSION'
    elif q>=.60 and s<.50: st='EDGE_RECOVERY'
    elif q<=.30: st='EDGE_DECAY'
    elif s>=.70 and q<.60: st='STRUCTURE_WITHOUT_EDGE'
    else: st='MIXED_EDGE'
    raw.append(st)
p['candidate_state']=raw
# Hysteresis: 3 consecutive observations to change quality state; decay can override after 2.
cur='WARMUP_UNKNOWN'; age=0; pend=None; n=0; final=[]
for cand in p.candidate_state:
    if cand=='WARMUP_UNKNOWN': final.append(cur); age+=1; continue
    if cur=='WARMUP_UNKNOWN': cur=cand; age=1; pend=None; n=0; final.append(cur); continue
    if cand==cur: age+=1; pend=None; n=0; final.append(cur); continue
    if pend==cand: n+=1
    else: pend=cand; n=1
    need=2 if cand=='EDGE_DECAY' else 3
    if n>=need and (age>=5 or cand=='EDGE_DECAY'):
        cur=cand; age=1; pend=None; n=0
    else: age+=1
    final.append(cur)
p['quality_state']=final
p['switch']=(p.quality_state!=p.quality_state.shift(1))&p.quality_state.ne('WARMUP_UNKNOWN')
p[['date','year','quality_state','candidate_state','quality_score','structure_score','real_break20','real_breakwin20','real_topmom20','real_topmom10','switch']].to_csv(OUT/'detector_v3_daily_quality.csv',index=False)

# Audit exact R10 transaction edge under V3 quality state. State definition never reads R10 outcome.
def load(folder,period):
    t=pd.read_csv(folder/'r10max_formal_trades.csv',dtype={'code':str}); o=pd.read_csv(folder/'r10max_formal_orders.csv',dtype={'code':str})
    for x in [t,o]: x.code=x.code.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)
    t.entry_date=pd.to_numeric(t.entry_date,errors='coerce').astype('Int64')
    b=o[(o.side=='BUY')&(o.status=='FILLED')][['code','strategy','fill_date','signal_date']].copy(); b.fill_date=pd.to_numeric(b.fill_date,errors='coerce').astype('Int64'); b.signal_date=pd.to_numeric(b.signal_date,errors='coerce').astype('Int64')
    m=t.merge(b,left_on=['code','strategy','entry_date'],right_on=['code','strategy','fill_date'],how='left')
    m=m.merge(p[['date','quality_state','quality_score','structure_score']],left_on='signal_date',right_on='date',how='left'); m['period']=period
    return m

def summ(m):
    rows=[]
    for st,g in m.dropna(subset=['quality_state']).groupby('quality_state'):
        gp=g.loc[g.pnl>0,'pnl'].sum(); gl=-g.loc[g.pnl<0,'pnl'].sum()
        rows.append({'quality_state':st,'trades':len(g),'win_rate':(g.pnl>0).mean(),'avg_return':g['return'].mean(),'net_pnl':g.pnl.sum(),'pf':gp/gl if gl>0 else np.nan,'avg_quality_score':g.quality_score.mean(),'avg_structure_score':g.structure_score.mean()})
    return pd.DataFrame(rows)
pre=load(P3/'phase3_old','2016-2020'); hold=load(P3/'phase3_new','2021-2025'); a=summ(pre); b=summ(hold)
a.to_csv(OUT/'r10_edge_by_quality_pre2021.csv',index=False); b.to_csv(OUT/'r10_edge_by_quality_holdout.csv',index=False)
# Year-by-state stability for pre-2021 only.
y=[]
pre['year']=pre.signal_date.astype(str).str[:4].astype(int)
for (yr,st),g in pre.groupby(['year','quality_state']):
    gp=g.loc[g.pnl>0,'pnl'].sum(); gl=-g.loc[g.pnl<0,'pnl'].sum()
    y.append({'year':yr,'quality_state':st,'trades':len(g),'avg_return':g['return'].mean(),'net_pnl':g.pnl.sum(),'pf':gp/gl if gl>0 else np.nan})
pd.DataFrame(y).to_csv(OUT/'pre2021_year_state_stability.csv',index=False)
manifest={'design':'causal-realized-opportunity-quality-detector-v3','market_refetch':False,'formal_r10_modified':False,'future_leakage':False,'quality_feature_rule':'20D forward diagnostics shifted +20 trading days; 10D diagnostics shifted +10, therefore fully realized before use','state_definition_uses_r10_outcome':False,'uses_2021_2025_to_define_thresholds':False,'thresholds':'generic rolling-percentile bands only','holdout_use':'audit only'}
(OUT/'stage6_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ADAPTIVE_ROUTER_STAGE6_COMPLETE')
print('\nPRE2021'); print(a.to_string(index=False)); print('\nHOLDOUT_NO_RETUNE'); print(b.to_string(index=False))
