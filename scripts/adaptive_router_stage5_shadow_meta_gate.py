#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
P3=ROOT/'calm_phase3_output'
S4=ROOT/'adaptive_router_stage4'
OUT=ROOT/'adaptive_router_stage5'; OUT.mkdir(exist_ok=True)

states=pd.read_csv(S4/'detector_v2_daily_states.csv')[['date','state']]
states['date']=pd.to_numeric(states.date,errors='coerce').astype('Int64')

def load(folder,period):
    t=pd.read_csv(folder/'r10max_formal_trades.csv',dtype={'code':str})
    o=pd.read_csv(folder/'r10max_formal_orders.csv',dtype={'code':str})
    for x in [t,o]: x['code']=x.code.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)
    t['entry_date']=pd.to_numeric(t.entry_date,errors='coerce').astype('Int64'); t['exit_date']=pd.to_numeric(t.exit_date,errors='coerce').astype('Int64')
    b=o[(o.side=='BUY')&(o.status=='FILLED')][['code','strategy','fill_date','signal_date']].copy()
    b['fill_date']=pd.to_numeric(b.fill_date,errors='coerce').astype('Int64'); b['signal_date']=pd.to_numeric(b.signal_date,errors='coerce').astype('Int64')
    m=t.merge(b,left_on=['code','strategy','entry_date'],right_on=['code','strategy','fill_date'],how='left')
    m=m.merge(states,left_on='signal_date',right_on='date',how='left'); m['period']=period
    return m

old=load(P3/'phase3_old','2016-2020'); new=load(P3/'phase3_new','2021-2025')
alltr=pd.concat([old,new],ignore_index=True).sort_values(['signal_date','entry_date','code']).reset_index(drop=True)

# Shadow engine is always running: even if actual router would allocate cash elsewhere,
# completed hypothetical R10 trades remain observable for future meta decisions.
features=[]
for i,r in alltr.iterrows():
    past=alltr[(alltr.exit_date<r.signal_date)&(alltr.strategy==r.strategy)].sort_values('exit_date')
    rec=past.tail(8); rec4=past.tail(4)
    gp=rec.loc[rec.pnl>0,'pnl'].sum(); gl=-rec.loc[rec.pnl<0,'pnl'].sum()
    pf8=float(gp/gl) if gl>0 else (10.0 if gp>0 else np.nan)
    features.append({'shadow_n':len(past),'shadow_n8':len(rec),'shadow_pf8':pf8,
                     'shadow_avg8':float(rec['return'].mean()) if len(rec) else np.nan,
                     'shadow_win8':float((rec.pnl>0).mean()) if len(rec) else np.nan,
                     'shadow_avg4':float(rec4['return'].mean()) if len(rec4) else np.nan})
f=pd.DataFrame(features); alltr=pd.concat([alltr.reset_index(drop=True),f],axis=1)

# Fixed policy family; thresholds are economic guardrails, not optimized against holdout.
def decide(r,policy):
    st=r.state; n=r.shadow_n8; pf=r.shadow_pf8; a8=r.shadow_avg8; a4=r.shadow_avg4
    good=(n>=6 and pd.notna(pf) and pf>=1.05 and pd.notna(a8) and a8>0)
    strong=(n>=6 and pd.notna(pf) and pf>=1.40 and pd.notna(a8) and a8>=0.025)
    weak=(n>=6 and ((pd.notna(pf) and pf<0.80) or (pd.notna(a8) and a8<-0.02)))
    if policy=='ALWAYS_R10': return True
    if policy=='MARKET_ONLY': return st in ['TREND_EXPANSION','ROTATION_DISPERSION']
    if policy=='SHADOW_EDGE': return good if n>=6 else st=='TREND_EXPANSION'
    if policy=='DEFENSIVE_HYBRID':
        if st=='TREND_EXPANSION': return not weak
        if st=='ROTATION_DISPERSION': return good
        return False
    if policy=='FULL_HYBRID':
        if st=='TREND_EXPANSION': return not weak
        if st=='ROTATION_DISPERSION': return good
        if st=='NEUTRAL_MIXED': return strong
        if st in ['RISK_OFF','CHOP_FALSE_BREAKOUT']: return strong and pd.notna(a4) and a4>0
        return False
    raise ValueError(policy)

policies=['ALWAYS_R10','MARKET_ONLY','SHADOW_EDGE','DEFENSIVE_HYBRID','FULL_HYBRID']
rows=[]; detail=[]
for pol in policies:
    alltr[pol]=alltr.apply(lambda r:decide(r,pol),axis=1)
    for period,g0 in alltr.groupby('period'):
        for selected,g in [('selected',g0[g0[pol]]),('rejected_shadow',g0[~g0[pol]])]:
            gp=g.loc[g.pnl>0,'pnl'].sum(); gl=-g.loc[g.pnl<0,'pnl'].sum()
            rows.append({'policy':pol,'period':period,'bucket':selected,'trades':len(g),'coverage':len(g)/len(g0) if len(g0) else np.nan,
                         'win_rate':(g.pnl>0).mean() if len(g) else np.nan,'avg_return':g['return'].mean() if len(g) else np.nan,
                         'net_pnl_original_sizing':g.pnl.sum(),'pf':gp/gl if gl>0 else np.nan})
    for _,r in alltr.iterrows(): detail.append({'policy':pol,'period':r.period,'signal_date':r.signal_date,'code':r.code,'strategy':r.strategy,'state':r.state,'selected':bool(r[pol]),'shadow_n8':r.shadow_n8,'shadow_pf8':r.shadow_pf8,'shadow_avg8':r.shadow_avg8,'shadow_avg4':r.shadow_avg4,'trade_return':r['return'],'pnl':r.pnl})
summary=pd.DataFrame(rows); summary.to_csv(OUT/'shadow_meta_policy_audit.csv',index=False); pd.DataFrame(detail).to_csv(OUT/'shadow_meta_trade_decisions.csv',index=False)

# Strict pre-2021 scorecard and completely separate holdout audit. No policy is chosen from holdout.
pre=summary[(summary.period=='2016-2020')&(summary.bucket=='selected')].copy()
hold=summary[(summary.period=='2021-2025')&(summary.bucket=='selected')].copy()
pre.to_csv(OUT/'policy_scorecard_pre2021.csv',index=False); hold.to_csv(OUT/'policy_holdout_audit_no_retune.csv',index=False)
manifest={'design':'causal-shadow-performance-meta-gate','market_refetch':False,'formal_r10_modified':False,'shadow_engine':'always-on hypothetical R10; only trades exited before current signal_date feed current gate','uses_current_or_future_trade_outcome':False,'policy_thresholds_tuned_on_2021_2025':False,'holdout_use':'audit only','important_limit':'trade-stream discrimination audit only; not yet a shared-capital router CAGR backtest','next':'if a fixed meta policy is stable pre/holdout, implement external shared-capital router and hard R10 invariance semantics'}
(OUT/'stage5_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ADAPTIVE_ROUTER_STAGE5_COMPLETE')
print('\nPRE2021 SELECTED'); print(pre.to_string(index=False))
print('\nHOLDOUT SELECTED NO RETUNE'); print(hold.to_string(index=False))
