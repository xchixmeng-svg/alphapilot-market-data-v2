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

ROOT=Path(__file__).resolve().parent.parent
INITIAL=1_300_000.0
DEV_START=20230523; DEV_END=20241231
BLIND_START=20250101; BLIND_END=20251231

# V27 is seeded from V26's two surviving structures: dispersion leadership and soft context.
# Arms are intentionally structural: different causal context use, sponsorship timing, leadership phase,
# anti-failure filters, concentration, and realization path. No 2025 information is used for selection.
HYPOTHESES={
 'dispersion_phase_transition':(20,3),
 'dispersion_calm_leader':(20,3),
 'breadth_disagreement_leader':(20,3),
 'capital_concentration_breakout':(20,3),
 'flow_absorption_breakout':(25,3),
 'sponsor_accel_then_leadership':(20,3),
 'leadership_then_sponsor_accel':(20,3),
 'breakout_failure_avoidance':(20,3),
 'cross_horizon_dispersion_leader':(25,3),
 'soft_context_two_slot':(20,2),
 'soft_context_fast_realization':(12,3),
 'soft_context_patient_compound':(35,2),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))
def lag(d,col,n): return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def priors(d):
    return {
      'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'dispersion':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
      'breadth_up':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    lag3_r5=pct(lag(d,'r5_pr',3),d.date)
    lag5_r20=pct(lag(d,'r20',5)-lag(d,'mkt_r20',5),d.date)
    lag5_flow=pct(lag(d,'flow5_pr',5),d.date); lag5_acc=pct(lag(d,'flow_accel_pr',5),d.date)
    lag5_dist=lag(d,'dist_ma20',5)
    disp_lag=lag(pd.DataFrame({'code':d.code,'date':d.date,'x':p['dispersion']}).rename(columns={'x':'disp'}),'disp',5)
    flowbreadth_lag=lag(pd.DataFrame({'code':d.code,'date':d.date,'x':p['flow_breadth']}).rename(columns={'x':'fb'}),'fb',5)

    base=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.50)&(d.close>=10)&d.amount20.notna()
    quality=(d.amount20_pr>=.35)&(d.vol20_pr<=.97)
    leadership=(r20p>=.62)&(d.r5_pr>=.53)&(d.aclose>=d.ma60*.98)&(d.dist_ma20.between(-.01,.15))
    core=base&leadership

    if name=='dispersion_phase_transition':
        mask=core&(p['dispersion']>=.56)&(p['dispersion']>=disp_lag)&(r5p>=.54)
        score=.20*d.flow_accel_pr+.13*d.flow20_pr+.25*r20p+.12*r5p+.14*p['dispersion']+.08*p['flow_breadth']+.08*d.amount20_pr
    elif name=='dispersion_calm_leader':
        mask=core&(p['dispersion']>=.55)&(p['calm']>=.48)&(d.vol20_pr<=.82)
        score=.19*d.flow_accel_pr+.13*d.flow20_pr+.24*r20p+.11*r5p+.13*p['dispersion']+.10*p['calm']+.10*d.amount20_pr
    elif name=='breadth_disagreement_leader':
        mask=core&(p['dispersion']>=.55)&(p['flow_breadth']<=.62)&(r5p>=.56)
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.27*r20p+.13*r5p+.12*p['dispersion']+.08*(1-p['flow_breadth'])+.06*d.amount20_pr
    elif name=='capital_concentration_breakout':
        mask=core&(d.flow5_pr>=.62)&(d.flow_accel_pr>=.68)&(d.amount20_pr>=.45)&(p['dispersion']>=.52)
        score=.24*d.flow_accel_pr+.18*d.flow5_pr+.13*d.flow20_pr+.23*r20p+.10*p['dispersion']+.12*d.amount20_pr
    elif name=='flow_absorption_breakout':
        mask=base&(r20p>=.60)&(d.flow5_pr>=.58)&(lag5_dist>=-.03)&(lag5_dist<=.04)&(d.dist_ma20.between(0,.10))&(p['calm']>=.42)
        score=.22*d.flow_accel_pr+.16*d.flow5_pr+.14*d.flow20_pr+.24*r20p+.10*p['calm']+.14*d.amount20_pr
    elif name=='sponsor_accel_then_leadership':
        mask=core&(lag5_acc>=.58)&(d.flow_accel_pr>=lag5_acc)&(r5p>=.55)&(r20p>=lag5_r20)
        score=.22*d.flow_accel_pr+.14*lag5_acc+.13*d.flow20_pr+.25*r20p+.12*r5p+.08*p['dispersion']+.06*d.amount20_pr
    elif name=='leadership_then_sponsor_accel':
        mask=core&(lag5_r20>=.58)&(r20p>=.62)&(d.flow_accel_pr>=.66)&(d.flow5_pr>=.54)
        score=.21*d.flow_accel_pr+.14*d.flow5_pr+.13*d.flow20_pr+.18*lag5_r20+.20*r20p+.08*p['dispersion']+.06*d.amount20_pr
    elif name=='breakout_failure_avoidance':
        overextended=(d.dist_ma20>.12)|(d.r5_pr>.95)
        crowded=(d.flow5_pr>.94)&(d.r20_pr>.94)
        mask=core&(~overextended)&(~crowded)&(p['dispersion']>=.50)&(d.vol20_pr<=.88)
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.26*r20p+.11*r5p+.11*p['dispersion']+.10*d.amount20_pr+.08*(1-d.vol20_pr)
    elif name=='cross_horizon_dispersion_leader':
        mask=core&(r60p.between(.52,.88))&(r5p>=.54)&(p['dispersion']>=.54)
        score=.19*d.flow_accel_pr+.13*d.flow20_pr+.15*r5p+.25*r20p+.12*r60p+.10*p['dispersion']+.06*d.amount20_pr
    elif name=='soft_context_two_slot':
        mask=core&(p['dispersion']>=.50)&(p['flow_breadth']>=.45)
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.24*r20p+.10*r5p+.10*d.amount20_pr+.10*p['flow_breadth']+.07*p['breadth_up']+.05*p['dispersion']
    elif name=='soft_context_fast_realization':
        mask=core&(r5p>=.58)&(d.flow5_pr>=.56)&(p['dispersion']>=.50)
        score=.21*d.flow_accel_pr+.14*d.flow20_pr+.22*r20p+.17*r5p+.10*d.flow5_pr+.09*p['dispersion']+.07*d.amount20_pr
    else: # soft_context_patient_compound
        mask=core&(r60p>=.55)&(d.flow20_pr>=.55)&(d.vol20_pr<=.82)&(p['calm']>=.48)
        score=.18*d.flow_accel_pr+.15*d.flow20_pr+.22*r20p+.14*r60p+.09*d.amount20_pr+.08*(1-d.vol20_pr)+.08*p['flow_breadth']+.06*p['dispersion']

    raw=d[(mask&quality).fillna(False)].copy()
    if raw.empty:return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sig=(raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False])
           .groupby('signal_date',as_index=False).head(slots))
    return sig,hold,slots

def yearly(nav):
    out={}
    if nav.empty:return out
    n=nav.set_index('date').nav.astype(float)
    for y in (2023,2024,2025):
        lo=max(y*10000+101,DEV_START) if y==2023 else y*10000+101
        z=n[(n.index>=lo)&(n.index<=y*10000+1231)]
        out[str(y)]=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else None
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); args=ap.parse_args(); name=args.hypothesis
    out=ROOT/'v27_dispersion_context_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily()
    d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index()
    c=v18.build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty:raise RuntimeError(f'no signals for {name}')
    bycode=v16.build_price_index(px_all); schedule=v16.simulate_fast(name,sig,hold,bycode)
    schedule.to_csv(out/'signal_schedule.csv',index=False)
    navdev,trdev,_=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL)
    nav25,tr25,_=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL)
    navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
    yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    yf=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    md=v13.metrics(navdev,trdev,yd); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,yf)
    navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False)
    nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False)
    navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
    if not corpfull.empty:corpfull.to_csv(out/'corporate_actions.csv',index=False)
    audit={'version':'v27-dispersion-context-frontier','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'causal context -> dispersion/flow-conditioned repricing playbook -> stock screen -> separate market/liquidity quality -> T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
