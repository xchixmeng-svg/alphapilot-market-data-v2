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

# V34 moves away from V33's mostly static context-rank variants.  Each family encodes
# a different causal *transition path* using only T-close and lagged observations.
HYPOTHESES={
 'transition_flow_reacceleration':(20,3),
 'transition_price_reacceleration':(20,3),
 'transition_breakout_absorption':(25,3),
 'transition_quiet_base_release':(25,3),
 'transition_sponsor_after_price':(20,3),
 'transition_price_after_sponsor':(20,3),
 'transition_uncrowded_leadership':(25,3),
 'transition_liquidity_expansion':(20,3),
 'transition_lowvol_sponsor':(30,3),
 'transition_multihorizon_confirmation':(25,3),
 'transition_two_slot_conviction':(25,2),
 'transition_failed_breakout_recovery':(30,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return v28.lag(d,col,n)
def priors(d): return v28.priors(d)

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    l3_r5=pct(lag(d,'r5_pr',3),d.date)
    l5_r20=pct(lag(d,'r20_pr',5),d.date)
    l5_flow5=pct(lag(d,'flow5_pr',5),d.date)
    l5_acc=pct(lag(d,'flow_accel_pr',5),d.date)
    l10_acc=pct(lag(d,'flow_accel_pr',10),d.date)
    l5_dist=lag(d,'dist_ma20',5)
    l10_dist=lag(d,'dist_ma20',10)
    l5_vol=lag(d,'vol20_pr',5)
    l5_amt=lag(d,'amount20_pr',5)

    liquid=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.52)
    quality=liquid&(d.vol20_pr<=.82)
    sponsor=(d.flow_accel_pr>=.56)&(d.flow20_pr>=.48)
    leader=(r20p>=.58)&(r60p>=.46)
    not_ext=d.dist_ma20.between(-.08,.13)
    not_crowded=~((d.flow5_pr>=.93)&(d.r20_pr>=.93))
    # Causal market-trend prior. mkt_r20_z uses rolling history shifted by one
    # session for its standardization, while the current T-close return itself is
    # available at decision time. This was accidentally referenced as p['trend']
    # even though v27/v28 priors do not define that key.
    trend=pd.Series(v18.sigmoid(d.mkt_r20_z).to_numpy(),index=d.index)
    context=.28*p['flow_breadth']+.24*p['breadth_up']+.20*p['dispersion']+.16*p['calm']+.12*trend

    if name=='transition_flow_reacceleration':
        path=(l10_acc>=.50)&(l5_acc<=.58)&(d.flow_accel_pr>=.66)
        mask=path&(d.flow20_pr>=.52)&(r20p>=.56)&quality&not_ext&not_crowded
        score=.24*d.flow_accel_pr+.12*d.flow20_pr+.10*(1-l5_acc)+.22*r20p+.10*r60p+.10*d.amount20_pr+.12*context
    elif name=='transition_price_reacceleration':
        path=(l5_r20.between(.38,.66))&(r20p>=.68)&(r5p>=.56)
        mask=path&sponsor&quality&not_ext&not_crowded
        score=.13*d.flow_accel_pr+.09*d.flow20_pr+.28*r20p+.14*r5p+.09*(1-l5_r20)+.10*d.amount20_pr+.17*context
    elif name=='transition_breakout_absorption':
        path=l5_dist.between(.035,.14)&d.dist_ma20.between(-.025,.045)&(d.vol20_pr<=.62)
        mask=path&sponsor&(r20p>=.56)&(r5p>=.44)&liquid&not_crowded
        score=.12*d.flow_accel_pr+.09*d.flow20_pr+.23*r20p+.10*r5p+.13*(1-d.vol20_pr)+.12*d.amount20_pr+.21*context
    elif name=='transition_quiet_base_release':
        path=(l10_dist.abs()<=.045)&(l5_dist.abs()<=.055)&(d.dist_ma20.between(.015,.10))
        mask=path&(d.flow_accel_pr>=.60)&(r5p>=.58)&(r20p>=.54)&quality&not_crowded
        score=.17*d.flow_accel_pr+.08*d.flow20_pr+.19*r5p+.20*r20p+.10*d.amount20_pr+.09*(1-d.vol20_pr)+.17*context
    elif name=='transition_sponsor_after_price':
        path=(l5_r20>=.58)&(l5_acc<=.58)&(d.flow_accel_pr>=.66)
        mask=path&leader&quality&not_ext&not_crowded
        score=.20*d.flow_accel_pr+.08*d.flow20_pr+.09*l5_r20+.24*r20p+.10*r60p+.10*d.amount20_pr+.19*context
    elif name=='transition_price_after_sponsor':
        path=(l5_acc>=.62)&(l5_flow5>=.54)&(l5_r20<=.62)&(r20p>=.66)&(r5p>=.54)
        mask=path&(d.flow_accel_pr>=.52)&quality&not_ext&not_crowded
        score=.11*l5_acc+.08*l5_flow5+.10*d.flow_accel_pr+.28*r20p+.13*r5p+.09*d.amount20_pr+.11*context
    elif name=='transition_uncrowded_leadership':
        uncrowded=(d.flow5_pr<=.84)&(d.r20_pr<=.88)&(d.dist_ma20<=.085)
        mask=sponsor&leader&quality&uncrowded&(l5_r20>=.50)
        score=.11*d.flow_accel_pr+.08*d.flow20_pr+.27*r20p+.12*r60p+.10*d.amount20_pr+.11*(1-d.flow5_pr)+.07*(1-d.vol20_pr)+.14*context
    elif name=='transition_liquidity_expansion':
        path=(l5_amt<=.62)&(d.amount20_pr>=.78)
        mask=path&sponsor&(r20p>=.60)&(r5p>=.50)&(d.vol20_pr<=.78)&not_ext&not_crowded
        score=.13*d.flow_accel_pr+.08*d.flow20_pr+.24*r20p+.10*r5p+.20*d.amount20_pr+.08*(1-d.vol20_pr)+.17*context
    elif name=='transition_lowvol_sponsor':
        path=(l5_vol>=.40)&(d.vol20_pr<=.48)
        mask=path&sponsor&(r20p>=.55)&(r60p>=.48)&(d.amount20_pr>=.58)&d.dist_ma20.between(-.05,.075)&not_crowded
        score=.13*d.flow_accel_pr+.09*d.flow20_pr+.23*r20p+.12*r60p+.13*(1-d.vol20_pr)+.11*d.amount20_pr+.19*context
    elif name=='transition_multihorizon_confirmation':
        path=(r5p>=.54)&(r20p>=.62)&(r60p>=.54)&(l5_r20>=.48)
        mask=path&sponsor&quality&not_ext&not_crowded
        score=.11*d.flow_accel_pr+.07*d.flow20_pr+.13*r5p+.26*r20p+.15*r60p+.09*d.amount20_pr+.19*context
    elif name=='transition_two_slot_conviction':
        path=((l5_acc>=.60)&(r20p>=.64))|((l5_r20>=.60)&(d.flow_accel_pr>=.64))
        mask=path&sponsor&(r60p>=.50)&(d.amount20_pr>=.66)&(d.vol20_pr<=.68)&not_ext&not_crowded
        score=.16*d.flow_accel_pr+.09*d.flow20_pr+.28*r20p+.13*r60p+.12*d.amount20_pr+.08*(1-d.vol20_pr)+.14*context
    else:
        failed=l10_dist.between(.03,.14)&l5_dist.between(-.065,.015)
        recovered=d.dist_ma20.between(-.005,.075)&(r5p>=.58)
        mask=failed&recovered&sponsor&(r20p>=.54)&quality&not_crowded
        score=.12*d.flow_accel_pr+.09*d.flow20_pr+.20*r5p+.22*r20p+.10*d.amount20_pr+.09*(1-d.vol20_pr)+.18*context

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
    out=ROOT/'v34_transition_path_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    navd.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trd).to_csv(out/'dev_trades.csv',index=False)
    navreset.to_csv(out/'fresh_reset_2025_nav.csv',index=False); pd.DataFrame(trreset).to_csv(out/'fresh_reset_2025_trades.csv',index=False)
    navf.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trf).to_csv(out/'full_trades.csv',index=False)
    if not corp.empty: corp.to_csv(out/'corporate_actions.csv',index=False)
    audit={'version':'v34-transition-path-frontier','hypothesis':name,'hold_days':hold,'slots':slots,'selection_uses_2025':False,'meta_research_has_seen_2025':True,
      'development_period':[DEV_START,DEV_END],'validation_period':[VAL_START,VAL_END],
      'validation_note':'2025 has been repeatedly observed across prior research batches; use as reused validation, not untouched final holdout. Future forward data remains the final untouched evidence.',
      'architecture':'causal market/flow context -> transition-path playbook -> stock screen -> separate liquidity/volatility quality -> T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed causal hold then open -0.5% adverse','integer_shares':True,'shared_capital':True,'fees_tax':'v13 realistic Taiwan model','corporate_actions':'official effective-date rules'},
      'dev':md,'fresh_reset_2025':mr,'rolling_continuous_2025':rolling,'full':mf,'year_returns':v28.yearly(navf),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
