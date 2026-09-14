#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v16_friction_prescreen as v16
import validate_v12_champion_portfolio_v13 as v13
import research_v18_context_adaptive_stock_setups as v18

ROOT=Path(__file__).resolve().parent.parent
INITIAL=1_300_000.0
DEV_START=20230523; DEV_END=20241231
BLIND_START=20250101; BLIND_END=20251231
HYPOTHESES={
 'pullback_acceptance_flow_floor':(25,3),
 'quiet_sponsorship_breakout':(25,3),
 'short_long_flow_agreement':(20,3),
 'residual_acceleration_uncrowded':(20,3),
 'sponsor_reversal_from_weakness':(25,3),
 'liquidity_shock_absorption':(20,3),
 'leader_reclaim_after_shakeout':(25,3),
 'multi_horizon_strength_not_extended':(30,3),
 'dispersion_opportunity_leader':(20,3),
 'broad_flow_independent_strength':(20,3),
 'calm_context_early_repricing':(25,3),
 'liquid_sponsor_compounder':(35,2),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6); return 1/(1+np.exp(-x))
def lag(d,col,n): return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def context(d):
    return {
      'flow':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'breadth':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
      'disp':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; c=context(d)
    res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    l2d=lag(d,'dist_ma20',2); l3d=lag(d,'dist_ma20',3); l5d=lag(d,'dist_ma20',5); l10d=lag(d,'dist_ma20',10)
    l3r5=pct(lag(d,'r5_pr',3),d.date); l5r20=pct(lag(d,'r20_pr',5),d.date); l10r20=pct(lag(d,'r20_pr',10),d.date)
    l3fa=pct(lag(d,'flow_accel_pr',3),d.date); l5fa=pct(lag(d,'flow_accel_pr',5),d.date); l10fa=pct(lag(d,'flow_accel_pr',10),d.date)
    l5f5=pct(lag(d,'flow5_pr',5),d.date); l10f5=pct(lag(d,'flow5_pr',10),d.date); l5f20=pct(lag(d,'flow20_pr',5),d.date)
    l5vol=pct(lag(d,'vol20_pr',5),d.date); l10vol=pct(lag(d,'vol20_pr',10),d.date); l5amt=pct(lag(d,'amount20_pr',5),d.date)
    base=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.aclose>=d.ma60*.94)
    liquid=(d.amount20_pr>=.50)&(d.vol20_pr<=.88)
    sponsor=(d.flow_accel_pr>=.58)&(d.flow20_pr>=.50)
    not_ext=d.dist_ma20.between(-.07,.13)

    if name=='pullback_acceptance_flow_floor':
        prior=(l10r20>=.62)&(l5fa>=.53); pull=(l3d.between(-.055,.015)); accept=(d.dist_ma20.between(-.005,.07))&(r5p>=.56); floor=(d.flow20_pr>=.54)&(l5f20>=.50)
        mask=base&prior&pull&accept&floor&liquid
        score=.18*l10r20+.12*l5fa+.23*r20p+.14*r5p+.13*d.flow20_pr+.10*d.amount20_pr+.10*c['breadth']
    elif name=='quiet_sponsorship_breakout':
        quiet=(l10vol<=.48)&(l10d.abs()<=.05)&(l5d.abs()<=.045); sponsor_build=(l10fa>=.50)&(l5fa>=.56)&(d.flow_accel_pr>=.62); breaknow=(r5p>=.60)&(r20p>=.58)&(d.dist_ma20.between(.005,.09))
        mask=base&quiet&sponsor_build&breaknow&liquid
        score=.18*l10fa+.16*l5fa+.16*d.flow_accel_pr+.21*r20p+.13*r5p+.09*d.amount20_pr+.07*c['calm']
    elif name=='short_long_flow_agreement':
        flowagree=(d.flow5_pr>=.61)&(d.flow20_pr>=.57)&(l5f20>=.53); strength=(r5p>=.56)&(r20p>=.63)&(r60p>=.54)
        mask=base&flowagree&strength&not_ext&liquid
        score=.15*d.flow5_pr+.14*d.flow20_pr+.10*l5f20+.26*r20p+.12*r60p+.10*r5p+.08*d.amount20_pr+.05*c['flow']
    elif name=='residual_acceleration_uncrowded':
        accel=(r5p>=.64)&(r20p>=.62)&(l5r20<=.62); uncrowded=(r20p<=.84)&(d.dist_ma20.between(-.025,.09))&(d.vol20_pr<=.78)
        mask=base&accel&uncrowded&sponsor&liquid
        score=.28*r20p+.18*r5p+.12*(r20p-l5r20).clip(lower=0)+.14*d.flow_accel_pr+.09*d.flow20_pr+.10*d.amount20_pr+.09*(1-d.vol20_pr)
    elif name=='sponsor_reversal_from_weakness':
        weakflow=(l10fa<=.48)&(l10f5<=.50); reversal=(l3fa>=.55)&(d.flow_accel_pr>=.66)&(d.flow5_pr>=.60); priceconfirm=(r5p>=.59)&(r20p>=.55)&(d.dist_ma20.between(-.035,.075))
        mask=base&weakflow&reversal&priceconfirm&liquid
        score=.12*(1-l10fa)+.10*(1-l10f5)+.22*d.flow_accel_pr+.13*d.flow5_pr+.20*r20p+.13*r5p+.10*d.amount20_pr
    elif name=='liquidity_shock_absorption':
        shock=(d.amount20_pr>=.78)&(l5amt<=.62); absorbed=(d.vol20_pr<=.72)&(d.dist_ma20.between(-.015,.085)); support=(d.flow20_pr>=.53)&(r20p>=.60)&(r5p>=.54)
        mask=base&shock&absorbed&support
        score=.20*d.amount20_pr+.11*(1-l5amt)+.13*(1-d.vol20_pr)+.14*d.flow20_pr+.25*r20p+.10*r5p+.07*c['flow']
    elif name=='leader_reclaim_after_shakeout':
        leader=(l10r20>=.68)&(r60p>=.58); shake=(l5d>=.02)&(l3d<=-.015); reclaim=(d.dist_ma20>=.005)&(r5p>=.61); flowhold=(l5f20>=.50)&(d.flow20_pr>=.52)
        mask=base&leader&shake&reclaim&flowhold&liquid&not_ext
        score=.16*l10r20+.15*r60p+.22*r20p+.16*r5p+.11*d.flow20_pr+.10*d.amount20_pr+.10*c['calm']
    elif name=='multi_horizon_strength_not_extended':
        consensus=(r5p>=.57)&(r20p>=.66)&(r60p>=.62); controlled=(d.dist_ma20.between(-.025,.085))&(d.vol20_pr<=.75)
        mask=base&consensus&controlled&(d.flow20_pr>=.51)&(d.flow_accel_pr>=.52)&liquid
        score=.16*r5p+.33*r20p+.18*r60p+.10*d.flow20_pr+.08*d.flow_accel_pr+.08*d.amount20_pr+.07*(1-d.vol20_pr)
    elif name=='dispersion_opportunity_leader':
        stock=(r20p>=.70)&(r5p>=.55)&(r60p>=.52); mask=base&stock&sponsor&not_ext&liquid
        score=.32*r20p+.13*r5p+.11*r60p+.13*d.flow_accel_pr+.08*d.flow20_pr+.08*d.amount20_pr+.15*c['disp']
    elif name=='broad_flow_independent_strength':
        independent=(r20p>=.69)&(r5p>=.57)&(d.flow20_pr>=.53); mask=base&independent&not_ext&liquid
        score=.34*r20p+.16*r5p+.11*d.flow20_pr+.10*d.flow_accel_pr+.09*d.amount20_pr+.08*(1-d.vol20_pr)+.06*c['flow']+.06*(1-c['flow'])
    elif name=='calm_context_early_repricing':
        early=(l5r20<=.58)&(r20p>=.61)&(r5p>=.61); ignition=(l5fa<=.57)&(d.flow_accel_pr>=.63); controlled=(d.dist_ma20.between(-.025,.08))&(d.vol20_pr<=.72)
        mask=base&early&ignition&controlled&(d.flow5_pr>=.55)&liquid
        score=.22*r20p+.16*r5p+.19*d.flow_accel_pr+.09*d.flow5_pr+.10*d.amount20_pr+.10*(1-d.vol20_pr)+.14*c['calm']
    else:
        longflow=(l10fa>=.54)&(l5f20>=.55)&(d.flow20_pr>=.57); longstrength=(r60p>=.60)&(r20p.between(.62,.84)); q=(d.amount20_pr>=.70)&(d.vol20_pr<=.72)&not_ext
        mask=base&longflow&longstrength&q&(d.flow_accel_pr>=.55)&(r5p>=.50)
        score=.12*l10fa+.11*l5f20+.13*d.flow20_pr+.18*r60p+.24*r20p+.08*r5p+.09*d.amount20_pr+.05*c['calm']

    raw=d[mask.fillna(False)].copy()
    if raw.empty:return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots),hold,slots

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
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); a=ap.parse_args(); name=a.hypothesis
    out=ROOT/'v32_sponsor_state_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index(); c=v18.build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty: raise RuntimeError(f'no signals for {name}')
    bycode=v16.build_price_index(px_all); schedule=v16.simulate_fast(name,sig,hold,bycode); schedule.to_csv(out/'signal_schedule.csv',index=False)
    navdev,trdev,_=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL)
    nav25,tr25,_=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL)
    navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
    yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    md=v13.metrics(navdev,trdev,yd); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,yf)
    navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False); nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False); navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
    if not corpfull.empty: corpfull.to_csv(out/'corporate_actions.csv',index=False)
    audit={'version':'v32-sponsor-state-structures','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal market/flow context soft prior -> stock-level sponsor/repricing playbook -> screen -> separate liquidity quality -> T+1 shared-capital execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
