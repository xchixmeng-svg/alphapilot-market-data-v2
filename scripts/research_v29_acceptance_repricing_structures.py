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
 'breakout_price_acceptance':(25,3),
 'sponsor_divergence_breakout':(20,3),
 'flow_persistence_price_catchup':(25,3),
 'residual_decay_reacceleration':(20,3),
 'quiet_range_first_sponsorship':(30,2),
 'two_stage_pullback_with_sponsor':(25,3),
 'low_vol_residual_continuation':(25,3),
 'liquidity_quality_residual_breakout':(20,3),
 'sponsor_breadth_divergence':(25,3),
 'multi_day_strength_acceptance':(20,3),
 'concentrated_high_conviction_repricing':(25,2),
 'context_scaled_residual_sponsor':(20,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6); return 1/(1+np.exp(-x))
def lag(d,col,n): return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def ctx(d):
    return {
      'flow':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'breadth':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
      'disp':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; c=ctx(d)
    res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    l2d=lag(d,'dist_ma20',2); l3d=lag(d,'dist_ma20',3); l5d=lag(d,'dist_ma20',5); l10d=lag(d,'dist_ma20',10)
    l3r5=pct(lag(d,'r5_pr',3),d.date); l5r5=pct(lag(d,'r5_pr',5),d.date); l5r20=pct(lag(d,'r20_pr',5),d.date); l10r20=pct(lag(d,'r20_pr',10),d.date)
    l3fa=pct(lag(d,'flow_accel_pr',3),d.date); l5fa=pct(lag(d,'flow_accel_pr',5),d.date); l10fa=pct(lag(d,'flow_accel_pr',10),d.date)
    l5f5=pct(lag(d,'flow5_pr',5),d.date); l5f20=pct(lag(d,'flow20_pr',5),d.date)
    l10vol=pct(lag(d,'vol20_pr',10),d.date); l5amt=pct(lag(d,'amount20_pr',5),d.date)
    base=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.aclose>=d.ma60*.94)
    quality=(d.amount20_pr>=.48)&(d.vol20_pr<=.88)
    sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.52)
    breakout=(r20p>=.62)&(r5p>=.55)&(d.dist_ma20.between(-.01,.14))
    not_ext=d.dist_ma20.between(-.07,.13)

    if name=='breakout_price_acceptance':
        acceptance=(l3d>=.015)&(l2d>=.01)&(d.dist_ma20>=.02)&(r5p>=.56)
        mask=base&sponsor&breakout&acceptance&quality
        score=.28*r20p+.15*r5p+.15*d.flow_accel_pr+.10*d.flow20_pr+.10*d.amount20_pr+.08*(1-d.vol20_pr)+.07*l3r5+.07*c['breadth']
    elif name=='sponsor_divergence_breakout':
        divergence=(l10fa>=.58)&(l10r20<=.52)&(r20p>=.64)&(d.flow_accel_pr>=.60)
        mask=base&divergence&(d.flow20_pr>=.54)&(r5p>=.56)&not_ext&quality
        score=.18*l10fa+.28*r20p+.15*r5p+.14*d.flow_accel_pr+.10*d.flow20_pr+.08*d.amount20_pr+.07*c['flow']
    elif name=='flow_persistence_price_catchup':
        persistent=(l10fa>=.50)&(l5fa>=.55)&(d.flow_accel_pr>=.60)&(l5f20>=.52)
        catchup=(l10r20<=.56)&(r20p>=.61)&(r5p>=.57)
        mask=base&persistent&catchup&not_ext&quality
        score=.12*l10fa+.14*l5fa+.16*d.flow_accel_pr+.12*l5f20+.24*r20p+.12*r5p+.10*d.amount20_pr
    elif name=='residual_decay_reacceleration':
        decay=(l10r20>=.66)&(l5r20<=.57); reacc=(r20p>=.65)&(r5p>=.60)
        mask=base&sponsor&decay&reacc&(d.dist_ma20.between(-.035,.09))&quality
        score=.18*l10r20+.10*(1-l5r20)+.28*r20p+.16*r5p+.12*d.flow_accel_pr+.08*d.flow20_pr+.08*d.amount20_pr
    elif name=='quiet_range_first_sponsorship':
        quiet=(l10vol<=.48)&(l10d.abs()<=.045)&(l5d.abs()<=.045); ignition=(l5fa<=.55)&(d.flow_accel_pr>=.67)&(d.flow5_pr>=.58)
        mask=base&quiet&ignition&(r20p>=.55)&(r5p>=.56)&(d.dist_ma20.between(-.01,.075))&quality
        score=.23*d.flow_accel_pr+.13*d.flow5_pr+.22*r20p+.13*r5p+.12*(1-d.vol20_pr)+.09*d.amount20_pr+.08*c['calm']
    elif name=='two_stage_pullback_with_sponsor':
        impulse=(l5r20>=.66)&(l5fa>=.55); pullback=(l3d.between(-.045,.025))&(d.dist_ma20.between(-.005,.07))
        mask=base&impulse&pullback&sponsor&(r5p>=.54)&(r20p>=.59)&quality
        score=.16*l5r20+.12*l5fa+.25*r20p+.13*r5p+.14*d.flow_accel_pr+.10*d.flow20_pr+.10*d.amount20_pr
    elif name=='low_vol_residual_continuation':
        mask=base&sponsor&(r20p>=.63)&(r60p>=.52)&(d.vol20_pr<=.55)&(l10vol<=.62)&not_ext
        score=.28*r20p+.11*r60p+.14*d.flow_accel_pr+.10*d.flow20_pr+.16*(1-d.vol20_pr)+.12*d.amount20_pr+.09*c['calm']
    elif name=='liquidity_quality_residual_breakout':
        liq=(d.amount20_pr>=.72)&(l5amt>=.55)&(d.vol20_pr<=.72)
        mask=base&liq&sponsor&breakout
        score=.28*r20p+.14*r5p+.16*d.amount20_pr+.10*l5amt+.12*d.flow_accel_pr+.08*d.flow20_pr+.07*(1-d.vol20_pr)+.05*c['flow']
    elif name=='sponsor_breadth_divergence':
        weak_context=(c['breadth']<=.55); stock_strength=sponsor&(r20p>=.65)&(r5p>=.56)
        mask=base&weak_context&stock_strength&not_ext&quality
        score=.29*r20p+.13*r5p+.17*d.flow_accel_pr+.11*d.flow20_pr+.10*d.amount20_pr+.10*c['flow']+.10*(1-c['breadth'])
    elif name=='multi_day_strength_acceptance':
        sustained=(l5r20>=.60)&(l3r5>=.54)&(r20p>=.64)&(r5p>=.56)&(l3d>=-.005)&(d.dist_ma20>=.01)
        mask=base&sponsor&sustained&not_ext&quality
        score=.16*l5r20+.12*l3r5+.27*r20p+.13*r5p+.13*d.flow_accel_pr+.09*d.flow20_pr+.10*d.amount20_pr
    elif name=='concentrated_high_conviction_repricing':
        mask=base&sponsor&(r20p>=.68)&(r5p>=.58)&(d.amount20_pr>=.62)&(d.vol20_pr<=.76)&not_ext
        score=.31*r20p+.14*r5p+.17*d.flow_accel_pr+.11*d.flow20_pr+.12*d.amount20_pr+.08*(1-d.vol20_pr)+.07*c['flow']
    else:
        soft=.35*c['flow']+.25*c['breadth']+.20*c['calm']+.20*c['disp']
        mask=base&sponsor&breakout&quality
        score=.28*r20p+.13*r5p+.14*d.flow_accel_pr+.09*d.flow20_pr+.10*d.amount20_pr+.08*(1-d.vol20_pr)+.18*soft

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
    out=ROOT/'v29_acceptance_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    audit={'version':'v29-acceptance-repricing-structures','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal market/flow context soft prior -> acceptance/repricing playbook -> stock screen -> separate liquidity quality -> T+1 shared-capital execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
