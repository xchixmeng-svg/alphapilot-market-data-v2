#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v16_friction_prescreen as v16
import validate_v12_champion_portfolio_v13 as v13
import research_v18_context_adaptive_stock_setups as v18
import research_v28_anti_failure_sponsor_frontier as v28

ROOT=Path(__file__).resolve().parent.parent
INITIAL=1_300_000.0
DEV_START=20230523; DEV_END=20241231
VAL_START=20250101; VAL_END=20251231

HYPOTHESES={
 'context_rank_balanced_leader':(20,3),
 'context_rank_dispersion_leader':(20,3),
 'context_rank_lowvol_compounder':(30,3),
 'context_rank_flow_persistence':(25,3),
 'context_rank_residual_reacceleration':(20,3),
 'context_rank_post_breakout_absorption':(25,3),
 'context_rank_two_slot_conviction':(20,2),
 'context_rank_quality_floor':(20,3),
 'context_rank_anti_crowded':(20,3),
 'context_rank_flow_then_price':(20,3),
 'context_rank_price_then_flow':(20,3),
 'context_rank_patient_hold':(35,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return v28.lag(d,col,n)
def priors(d): return v28.priors(d)

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    l3dist=lag(d,'dist_ma20',3); l5dist=lag(d,'dist_ma20',5)
    l5f=pct(lag(d,'flow5_pr',5),d.date); l5a=pct(lag(d,'flow_accel_pr',5),d.date)
    l5r20=pct(lag(d,'r20_pr',5),d.date); l10v=lag(d,'vol20_pr',10)

    liquid=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.56)
    quality=liquid&(d.vol20_pr<=.74)
    sponsor=(d.flow_accel_pr>=.56)&(d.flow20_pr>=.48)
    leader=(r20p>=.58)&(r60p>=.46)
    breakout=(d.r5_pr>=.50)&(d.dist_ma20.between(-.025,.12))&(d.aclose>=d.ma60*.98)
    notcrowd=~((d.flow5_pr>.94)&(d.r20_pr>.94))
    context=.30*p['flow_breadth']+.25*p['breadth_up']+.25*p['dispersion']+.20*p['calm']

    if name=='context_rank_balanced_leader':
        mask=sponsor&leader&breakout&quality&notcrowd
        score=.14*d.flow_accel_pr+.09*d.flow20_pr+.24*r20p+.11*r60p+.13*d.amount20_pr+.08*(1-d.vol20_pr)+.21*context
    elif name=='context_rank_dispersion_leader':
        mask=sponsor&leader&breakout&quality&notcrowd
        score=.13*d.flow_accel_pr+.08*d.flow20_pr+.25*r20p+.10*r60p+.12*d.amount20_pr+.07*(1-d.vol20_pr)+.15*p['dispersion']+.10*p['flow_breadth']
    elif name=='context_rank_lowvol_compounder':
        mask=sponsor&(r20p>=.56)&(r60p>=.50)&(d.amount20_pr>=.60)&(d.vol20_pr<=.50)&(d.dist_ma20.between(-.05,.08))&notcrowd
        score=.12*d.flow_accel_pr+.09*d.flow20_pr+.21*r20p+.14*r60p+.13*d.amount20_pr+.14*(1-d.vol20_pr)+.17*context
    elif name=='context_rank_flow_persistence':
        mask=leader&breakout&quality&(d.flow5_pr>=.52)&(l5f>=.50)&(l5a>=.48)&notcrowd
        score=.10*d.flow_accel_pr+.08*d.flow20_pr+.09*d.flow5_pr+.09*l5f+.08*l5a+.22*r20p+.10*r60p+.09*d.amount20_pr+.15*context
    elif name=='context_rank_residual_reacceleration':
        mask=sponsor&breakout&quality&(l5r20.between(.40,.72))&(r20p>=.64)&(r5p>=.54)&notcrowd
        score=.12*d.flow_accel_pr+.08*d.flow20_pr+.29*r20p+.14*r5p+.08*(1-l5r20)+.10*d.amount20_pr+.07*(1-d.vol20_pr)+.12*context
    elif name=='context_rank_post_breakout_absorption':
        prior_up=l5dist.between(.01,.12); now_absorb=d.dist_ma20.between(-.035,.045)
        mask=sponsor&(r20p>=.58)&prior_up&now_absorb&quality&(r5p>=.46)&notcrowd
        score=.11*d.flow_accel_pr+.08*d.flow20_pr+.24*r20p+.10*r60p+.10*r5p+.11*d.amount20_pr+.09*(1-d.vol20_pr)+.17*context
    elif name=='context_rank_two_slot_conviction':
        mask=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.52)&(r20p>=.62)&(r60p>=.50)&breakout&(d.amount20_pr>=.68)&(d.vol20_pr<=.66)&notcrowd
        score=.15*d.flow_accel_pr+.10*d.flow20_pr+.29*r20p+.13*r60p+.11*d.amount20_pr+.08*(1-d.vol20_pr)+.14*context
    elif name=='context_rank_quality_floor':
        mask=sponsor&leader&breakout&(d.amount20_pr>=.72)&(d.vol20_pr<=.62)&notcrowd
        score=.12*d.flow_accel_pr+.08*d.flow20_pr+.24*r20p+.11*r60p+.17*d.amount20_pr+.10*(1-d.vol20_pr)+.18*context
    elif name=='context_rank_anti_crowded':
        uncrowded=(d.flow5_pr<=.86)&(d.r20_pr<=.88)&(d.dist_ma20<=.085)
        mask=sponsor&leader&breakout&quality&uncrowded
        score=.12*d.flow_accel_pr+.08*d.flow20_pr+.25*r20p+.10*r60p+.12*d.amount20_pr+.08*(1-d.flow5_pr)+.07*(1-d.vol20_pr)+.18*context
    elif name=='context_rank_flow_then_price':
        mask=(l5a>=.56)&(l5f>=.52)&leader&breakout&quality&(d.flow_accel_pr>=.52)&notcrowd
        score=.12*l5a+.10*l5f+.11*d.flow_accel_pr+.23*r20p+.11*r60p+.10*d.amount20_pr+.07*(1-d.vol20_pr)+.16*context
    elif name=='context_rank_price_then_flow':
        prior_price=(l5r20>=.58)&(l3dist>=-.02)
        mask=prior_price&sponsor&leader&quality&(d.flow5_pr>=.54)&notcrowd
        score=.10*l5r20+.12*d.flow_accel_pr+.09*d.flow20_pr+.10*d.flow5_pr+.22*r20p+.10*r60p+.10*d.amount20_pr+.17*context
    else: # patient hold
        mask=sponsor&leader&breakout&quality&(d.flow5_pr>=.50)&notcrowd
        score=.12*d.flow_accel_pr+.09*d.flow20_pr+.23*r20p+.13*r60p+.11*d.amount20_pr+.08*(1-d.vol20_pr)+.24*context

    raw=d[(mask&quality).fillna(False)].copy()
    if raw.empty:return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sig=(raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False])
            .groupby('signal_date',as_index=False).head(slots))
    return sig,hold,slots

def window_metrics(nav,trades,start,end):
    z=nav[(nav.date>=start)&(nav.date<=end)].copy(); tt=[x for x in trades if start<=int(x['exit_date'])<=end]
    if z.empty:return {'end_nav':None,'total_return':None,'cagr':None,'max_dd':None,'trades':0,'win_rate':0.0,'pf':0.0}
    n=z.nav.astype(float); ret=float(n.iloc[-1]/n.iloc[0]-1); dd=float((n/n.cummax()-1).min()); rr=pd.Series([x['return'] for x in tt],dtype=float)
    return {'end_nav':float(n.iloc[-1]),'total_return':ret,'cagr':ret,'max_dd':dd,'trades':int(len(rr)),'win_rate':float((rr>0).mean()) if len(rr) else 0.0,'pf':float(v13.pf(rr)) if len(rr) else 0.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); a=ap.parse_args(); name=a.hypothesis
    out=ROOT/'v33_context_rank_out'/name; out.mkdir(parents=True,exist_ok=True)
    px,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index()
    d=d.merge(m,on='date',how='left').merge(v18.build_context(d),on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty: raise RuntimeError('no signals '+name)
    schedule=v16.simulate_fast(name,sig,hold,v16.build_price_index(px)); schedule.to_csv(out/'signal_schedule.csv',index=False)
    navd,trd,_=v13.simulate_portfolio(px,schedule,slots,DEV_START,DEV_END,INITIAL)
    navreset,trreset,_=v13.simulate_portfolio(px,schedule,slots,VAL_START,VAL_END,INITIAL)
    navf,trf,corp=v13.simulate_portfolio(px,schedule,slots,DEV_START,VAL_END,INITIAL)
    yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(VAL_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    md=v13.metrics(navd,trd,yd); mr=v13.metrics(navreset,trreset,1.0); mf=v13.metrics(navf,trf,yf); rolling=window_metrics(navf,trf,VAL_START,VAL_END)
    navd.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trd).to_csv(out/'dev_trades.csv',index=False); navreset.to_csv(out/'fresh_reset_2025_nav.csv',index=False); pd.DataFrame(trreset).to_csv(out/'fresh_reset_2025_trades.csv',index=False); navf.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trf).to_csv(out/'full_trades.csv',index=False)
    if not corp.empty: corp.to_csv(out/'corporate_actions.csv',index=False)
    audit={'version':'v33-context-rank-frontier','hypothesis':name,'hold_days':hold,'slots':slots,'selection_uses_2025':False,'meta_research_has_seen_2025':True,'development_period':[DEV_START,DEV_END],'validation_period':[VAL_START,VAL_END],'validation_note':'2025 has been repeatedly observed across prior research batches; treat as reused validation, not untouched final holdout. Final untouched evidence must come from future forward data.','architecture':'causal context soft-rank -> playbook-specific stock screen -> separate liquidity/volatility quality -> T+1 execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','dev':md,'fresh_reset_2025':mr,'rolling_continuous_2025':rolling,'full':mf,'year_returns':v28.yearly(navf),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
