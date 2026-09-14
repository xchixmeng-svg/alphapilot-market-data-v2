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

# V25 starts from the V24 Pareto winner but tests genuinely different causal
# interpretations of breakout quality. 2025 is never used for hypothesis choice.
HYPOTHESES={
 'breakout_retest_recovery':(25,3),
 'breakout_quiet_absorption':(25,3),
 'breakout_sponsor_persistence':(25,3),
 'breakout_residual_acceleration':(20,3),
 'breakout_early_stage':(25,3),
 'breakout_liquidity_quality':(20,3),
 'breakout_compression_release':(30,2),
 'breakout_soft_breadth_tailwind':(20,3),
 'breakout_relative_strength_confirm':(20,3),
 'breakout_flow_then_price':(25,3),
 'breakout_delayed_confirmation':(20,3),
 'liquidity_sponsor_breakout':(25,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))
def lag(d,col,n):
    return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def priors(d):
    return {
      'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'breadth_up':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
      'dispersion':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    lag5_flow=pct(lag(d,'flow5_pr',5),d.date)
    lag5_acc=pct(lag(d,'flow_accel_pr',5),d.date)
    lag5_r20=pct(lag(d,'r20_pr',5),d.date)
    lag3_dist=lag(d,'dist_ma20',3)
    lag3_r5=pct(lag(d,'r5_pr',3),d.date)
    lag10_vol=pct(lag(d,'vol20_pr',10),d.date)
    base=(d.flow_accel_pr>=.62)&(d.flow20_pr>=.52)&(d.close>=10)&d.amount20.notna()
    quality=(d.amount20_pr>=.35)&(d.vol20_pr<=.97)
    breakout=(r20p>=.62)&(d.r5_pr>=.54)&(d.dist_ma20.between(0,.15))&(d.aclose>=d.ma60*.98)
    not_ext=d.dist_ma20.between(-.08,.16)

    if name=='breakout_retest_recovery':
        # Prior short pullback/near-MA20 state, followed by renewed positive distance and sponsorship.
        mask=base&(r20p>=.62)&(lag3_dist.between(-.045,.035))&(d.dist_ma20.between(.005,.11))&(d.r5_pr>=.58)&(d.flow5_pr>=.54)
        score=.22*d.flow_accel_pr+.16*d.flow20_pr+.24*r20p+.14*d.r5_pr+.10*d.flow5_pr+.14*p['breadth_up']
    elif name=='breakout_quiet_absorption':
        # Sponsorship with muted volatility before the price expansion; proxy for absorption.
        mask=base&breakout&(d.vol20_pr<=.58)&(lag10_vol<=.62)
        score=.22*d.flow_accel_pr+.15*d.flow20_pr+.24*r20p+.16*(1-d.vol20_pr)+.11*d.amount20_pr+.12*p['calm']
    elif name=='breakout_sponsor_persistence':
        # Current breakout only after sponsorship was already visible five sessions earlier.
        mask=breakout&(d.flow_accel_pr>=.60)&(d.flow20_pr>=.56)&(lag5_flow>=.52)&(lag5_acc>=.50)&quality
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.14*lag5_flow+.12*lag5_acc+.24*r20p+.10*d.r5_pr+.06*p['flow_breadth']
    elif name=='breakout_residual_acceleration':
        # Residual leadership accelerates from a merely-good prior rank into a current leader.
        mask=base&breakout&(lag5_r20.between(.45,.75))&(r20p>=.68)&(r5p>=.58)
        score=.20*d.flow_accel_pr+.13*d.flow20_pr+.29*r20p+.16*r5p+.10*(1-lag5_r20)+.12*d.amount20_pr
    elif name=='breakout_early_stage':
        # Avoid mature 60D winners: seek first repricing leg, not late-stage momentum.
        mask=base&breakout&(r60p.between(.40,.72))&(d.r60_pr<=.82)&not_ext
        score=.22*d.flow_accel_pr+.15*d.flow20_pr+.27*r20p+.12*d.r5_pr+.12*(1-r60p)+.12*d.amount20_pr
    elif name=='breakout_liquidity_quality':
        # Separate quality pass: high tradability plus sponsorship and residual leadership.
        mask=base&breakout&(d.amount20_pr>=.72)&(d.vol20_pr<=.78)
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.25*r20p+.19*d.amount20_pr+.12*(1-d.vol20_pr)+.10*p['flow_breadth']
    elif name=='breakout_compression_release':
        # Low-volatility/base compression before a sponsored breakout.
        prior_near=(lag3_dist.abs()<=.045)&(lag3_r5<=.68)
        mask=base&breakout&prior_near&(d.vol20_pr<=.60)&(d.flow5_pr>=.52)
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.25*r20p+.12*d.r5_pr+.13*(1-d.vol20_pr)+.08*d.flow5_pr+.08*p['calm']
    elif name=='breakout_soft_breadth_tailwind':
        # Market breadth only influences ranking; it is not a hard market gate.
        mask=base&breakout&not_ext
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.24*r20p+.11*d.r5_pr+.10*d.amount20_pr+.11*p['flow_breadth']+.10*p['breadth_up']
    elif name=='breakout_relative_strength_confirm':
        # Cross-horizon residual leadership confirmation without requiring long-run crowding.
        mask=base&breakout&(r5p>=.56)&(r20p>=.64)&(r60p>=.50)&(r60p<=.84)
        score=.18*d.flow_accel_pr+.13*d.flow20_pr+.18*r5p+.27*r20p+.12*r60p+.12*d.amount20_pr
    elif name=='breakout_flow_then_price':
        # Sponsorship leads; price breakout is delayed confirmation at T close.
        mask=(lag5_acc>=.58)&(lag5_flow>=.55)&breakout&(d.flow_accel_pr>=.55)&quality
        score=.16*lag5_acc+.14*lag5_flow+.20*d.flow_accel_pr+.24*r20p+.14*d.r5_pr+.12*d.amount20_pr
    elif name=='breakout_delayed_confirmation':
        # Require a prior positive move but current acceleration; still execute only T+1 after current T close.
        mask=base&breakout&(lag3_r5>=.52)&(d.r5_pr>=lag3_r5)&(d.flow5_pr>=.55)&not_ext
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.25*r20p+.15*d.r5_pr+.10*lag3_r5+.08*d.flow5_pr+.08*d.amount20_pr
    else: # liquidity_sponsor_breakout
        # Blend the two V24 Pareto families: liquid sponsorship first, then breakout confirmation.
        mask=(d.amount20_pr>=.70)&(d.flow_accel_pr>=.60)&(d.flow20_pr>=.54)&breakout&(d.vol20_pr<=.86)
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.24*r20p+.17*d.amount20_pr+.11*d.r5_pr+.08*(1-d.vol20_pr)+.06*p['flow_breadth']

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
    out=ROOT/'v25_breakout_quality_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    audit={'version':'v25-breakout-quality-structures','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'causal context -> breakout/repricing playbook -> stock screen -> separate market/liquidity quality -> T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__':main()
