#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
S1=ROOT/'adaptive_router_stage1'
P2=ROOT/'calm_phase2_output'
P3=ROOT/'calm_phase3_output'
OUT=ROOT/'adaptive_router_stage2'; OUT.mkdir(exist_ok=True)

states=pd.read_csv(S1/'stage1_daily_states.csv')
diag=pd.read_csv(S1/'stage1_state_diagnostics_2016_2020.csv')
phase2=pd.read_csv(P2/'daily_cross_section_panel.csv')
old_trades=pd.read_csv(P3/'phase3_old'/'r10max_formal_trades.csv')
old_orders=pd.read_csv(P3/'phase3_old'/'r10max_formal_orders.csv')
new_trades=pd.read_csv(P3/'phase3_new'/'r10max_formal_trades.csv')
new_orders=pd.read_csv(P3/'phase3_new'/'r10max_formal_orders.csv')

states['date']=pd.to_numeric(states.date,errors='coerce').astype('Int64')
states['year']=states['date'].astype(str).str[:4].astype(int)

def attach_state(trades,orders,label):
    t=trades.copy(); o=orders.copy()
    for x in (t,o): x['code']=x['code'].astype(str).str.replace(r'\\.0$','',regex=True).str.zfill(4)
    t['entry_date']=pd.to_numeric(t.entry_date,errors='coerce').astype('Int64')
    o['fill_date']=pd.to_numeric(o.fill_date,errors='coerce').astype('Int64')
    o['signal_date']=pd.to_numeric(o.signal_date,errors='coerce').astype('Int64')
    b=o[(o.side=='BUY') & (o.status=='FILLED')][['code','strategy','fill_date','signal_date']].copy()
    m=t.merge(b,left_on=['code','strategy','entry_date'],right_on=['code','strategy','fill_date'],how='left')
    m=m.merge(states[['date','state']],left_on='signal_date',right_on='date',how='left')
    m['period']=label
    return m

def summarize(m):
    rows=[]
    for state,g in m.dropna(subset=['state']).groupby('state'):
        gp=g.loc[g.pnl>0,'pnl'].sum(); gl=-g.loc[g.pnl<0,'pnl'].sum()
        rows.append({'state':state,'trades':int(len(g)),'wins':int((g.pnl>0).sum()),
                     'win_rate':float((g.pnl>0).mean()),'avg_return':float(g['return'].mean()),
                     'median_return':float(g['return'].median()),'net_pnl':float(g.pnl.sum()),
                     'pf':float(gp/gl) if gl>0 else None})
    return pd.DataFrame(rows)

pre=attach_state(old_trades,old_orders,'2016-2020')
hold=attach_state(new_trades,new_orders,'2021-2025')
pre_s=summarize(pre); hold_s=summarize(hold)
pre_s.to_csv(OUT/'r10_state_edge_2016_2020.csv',index=False)
hold_s.to_csv(OUT/'r10_state_audit_2021_2025.csv',index=False)

# Stability by discovery year, using only pre-2021 data for candidate selection.
pre['year']=pre.entry_date.astype(str).str[:4].astype(int)
yearly=[]
for (y,state),g in pre.groupby(['year','state']):
    gp=g.loc[g.pnl>0,'pnl'].sum(); gl=-g.loc[g.pnl<0,'pnl'].sum()
    yearly.append({'year':int(y),'state':state,'trades':int(len(g)),'avg_return':float(g['return'].mean()),
                   'net_pnl':float(g.pnl.sum()),'pf':float(gp/gl) if gl>0 else None})
pd.DataFrame(yearly).to_csv(OUT/'r10_state_edge_by_year_2016_2020.csv',index=False)

# Candidate strategy-family hypotheses. Forward columns are diagnostic only; they never define state.
d=diag.set_index('state').to_dict('index'); r=pre_s.set_index('state').to_dict('index')
all_states=['TREND_EXPANSION','ROTATION_DISPERSION','RISK_OFF','CHOP_FALSE_BREAKOUT','NEUTRAL_MIXED']
rows=[]
for state in all_states:
    rd=r.get(state,{}); dd=d.get(state,{})
    pf=rd.get('pf',np.nan); n=int(rd.get('trades',0) or 0)
    mom=float(dd.get('avg_top_mom_fwd20',np.nan)); br=float(dd.get('avg_breakout_fwd20',np.nan))
    if state=='ROTATION_DISPERSION' and n>=10 and pd.notna(pf) and pf>1.25:
        primary='R10_LOCKED'; challenger='ROTATION_SPECIALIST'
    elif state=='TREND_EXPANSION':
        primary='TREND_BREAKOUT_TOURNAMENT'; challenger='R10_LOCKED'
    elif state=='RISK_OFF':
        primary='CASH_DEFENSIVE'; challenger='OVERSOLD_REBOUND_TEST_ONLY'
    elif state=='CHOP_FALSE_BREAKOUT':
        primary='CASH_OR_MEAN_REVERSION'; challenger='MEAN_REVERSION_TEST_ONLY'
    else:
        primary='SELECTIVE_NEUTRAL_TOURNAMENT'; challenger='CASH'
    rows.append({'state':state,'pre2021_r10_trades':n,'pre2021_r10_pf':pf,
                 'pre2021_r10_avg_return':rd.get('avg_return',np.nan),
                 'proxy_top_momentum_fwd20':mom,'proxy_breakout_fwd20':br,
                 'primary_candidate':primary,'challenger':challenger})
cand=pd.DataFrame(rows)
cand.to_csv(OUT/'stage2_candidate_strategy_map_pre2021_only.csv',index=False)

manifest={
 'design':'adaptive-router-stage2-state-edge-audit',
 'inputs':'validated cached Stage1 + Phase2 + Phase3 artifacts only',
 'network_market_refetch':False,
 'formal_r10_modified':False,
 'candidate_map_uses_2021_2025':False,
 'holdout_2021_2025_use':'audit_only_no_retuning',
 'next_stage':'transaction-level tournament of independent strategy families on 2016-2020; preserve exact R10 as external locked candidate'
}
(OUT/'stage2_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ADAPTIVE_ROUTER_STAGE2_COMPLETE')
print('\nPRE2021_R10_BY_STATE')
print(pre_s.sort_values('net_pnl',ascending=False).to_string(index=False))
print('\nCANDIDATE_MAP_PRE2021_ONLY')
print(cand.to_string(index=False))
print('\nHOLDOUT_AUDIT_NO_RETUNE')
print(hold_s.sort_values('net_pnl',ascending=False).to_string(index=False))
