#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
INP=ROOT/'calm_phase2_output'/'daily_cross_section_panel.csv'
OUT=ROOT/'adaptive_router_output'; OUT.mkdir(exist_ok=True)

if not INP.exists():
    raise FileNotFoundError(f'missing cached Phase2 panel: {INP}')

df=pd.read_csv(INP).sort_values('date').reset_index(drop=True)
# strictly past-only rolling transforms; no forward columns used for state assignment
features=['breadth_ma60','breadth_r20_pos','median_r20','cross_section_r20_std','break60_rate']
for c in features:
    s=pd.to_numeric(df[c],errors='coerce')
    med=s.shift(1).rolling(126,min_periods=63).median()
    mad=(s.shift(1)-med).abs().rolling(126,min_periods=63).median().replace(0,np.nan)
    df[c+'_z']=(s-med)/(1.4826*mad)

# Causal state taxonomy. This is a discovery baseline, not a final tuned classifier.
def classify(r):
    b=r.get('breadth_ma60_z'); m=r.get('median_r20_z'); d=r.get('cross_section_r20_std_z'); br=r.get('break60_rate_z')
    vals=[b,m,d,br]
    if any(pd.isna(x) for x in vals): return 'WARMUP_UNKNOWN'
    if b>=0.75 and m>=0.50 and br>=0.50: return 'TREND_EXPANSION'
    if b>=-0.25 and m>=-0.25 and d>=0.50: return 'ROTATION_DISPERSION'
    if b<=-0.75 and m<=-0.50: return 'RISK_OFF'
    if br<=-0.50 and abs(m)<=0.50: return 'CHOP_FALSE_BREAKOUT'
    return 'NEUTRAL_MIXED'

df['state']=df.apply(classify,axis=1)
df['year']=df['date'].astype(str).str[:4].astype(int)

# Forward columns are used only AFTER state assignment for diagnostics, never to define the state.
metrics=[]
for state,g in df[df.year.between(2016,2020)].groupby('state'):
    metrics.append({
        'state':state,'days':int(len(g)),
        'avg_top_mom_fwd10':float(g['top10pct_r20_fwd10_mean'].mean()),
        'avg_top_mom_fwd20':float(g['top10pct_r20_fwd20_mean'].mean()),
        'avg_breakout_fwd20':float(g['break60_fwd20_mean'].mean()),
        'breakout_win':float(g['break60_fwd20_win'].mean()),
        'avg_breadth':float(g['breadth_ma60'].mean()),
        'avg_dispersion':float(g['cross_section_r20_std'].mean())
    })
summary=pd.DataFrame(metrics).sort_values('avg_top_mom_fwd20',ascending=False)
summary.to_csv(OUT/'stage1_state_diagnostics_2016_2020.csv',index=False)
df[['date','year','state']+[c+'_z' for c in features]].to_csv(OUT/'stage1_daily_states.csv',index=False)

# Holdout audit: distribution only, no threshold retuning on 2021-2025.
hold=df[df.year.between(2021,2025)].groupby('state').size().rename('days').reset_index()
hold.to_csv(OUT/'stage1_holdout_state_distribution_2021_2025.csv',index=False)

meta={
 'design':'adaptive-router-stage1-causal-state-discovery',
 'data_source':'cached calm_phase2_output/daily_cross_section_panel.csv',
 'network_fetch':False,
 'formal_r10_modified':False,
 'training_diagnostic_window':'2016-2020',
 'holdout_distribution_only':'2021-2025',
 'state_assignment_uses_forward_data':False,
 'states':['TREND_EXPANSION','ROTATION_DISPERSION','RISK_OFF','CHOP_FALSE_BREAKOUT','NEUTRAL_MIXED','WARMUP_UNKNOWN']
}
(OUT/'stage1_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('ADAPTIVE_ROUTER_STAGE1_COMPLETE')
print(summary.to_string(index=False))
print(hold.to_string(index=False))
