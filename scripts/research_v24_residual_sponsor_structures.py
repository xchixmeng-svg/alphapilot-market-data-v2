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

# V24 structurally extends the strongest V23 evidence family: sponsor acceleration + residual momentum.
# These are not tiny threshold sweeps; each changes the causal setup interpretation or portfolio path.
HYPOTHESES={
 'residual_sponsor_persistence':(25,3),
 'residual_sponsor_breakout':(20,3),
 'residual_sponsor_pullback_reload':(30,2),
 'residual_sponsor_low_vol':(30,2),
 'residual_sponsor_liquidity':(20,3),
 'residual_sponsor_multihorizon':(25,3),
 'residual_sponsor_breadth_confirm':(20,3),
 'residual_sponsor_dispersion_leader':(20,3),
 'residual_sponsor_anti_crowd':(25,3),
 'residual_sponsor_flow_leads_price':(25,3),
 'residual_sponsor_price_leads_flow':(20,3),
 'residual_sponsor_concentrated_slots':(25,2),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))

def priors(d):
    return {
      'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'dispersion':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
      'breadth_up':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
      'trend':pd.Series(sigmoid(d.mkt_r20_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    base=(d.flow_accel_pr>=.62)&(d.flow20_pr>=.52)&(d.close>=10)&d.amount20.notna()
    quality=(d.amount20_pr>=.35)&(d.vol20_pr<=.97)
    not_ext=d.dist_ma20.between(-.10,.16)

    if name=='residual_sponsor_persistence':
        mask=base&(d.flow5_pr>=.56)&(d.flow20_pr>=.58)&(r20p>=.60)&(r60p>=.52)&not_ext
        score=.24*d.flow_accel_pr+.18*d.flow5_pr+.18*d.flow20_pr+.22*r20p+.10*r60p+.08*p['flow_breadth']
    elif name=='residual_sponsor_breakout':
        mask=base&(r20p>=.64)&(d.r5_pr>=.55)&(d.dist_ma20.between(.00,.14))&(d.aclose>=d.ma60*.98)
        score=.24*d.flow_accel_pr+.16*d.flow20_pr+.26*r20p+.12*d.r5_pr+.12*d.amount20_pr+.10*p['breadth_up']
    elif name=='residual_sponsor_pullback_reload':
        mask=base&(r60p>=.62)&(d.dist_ma20.between(-.08,.025))&(d.r5_pr>=.38)&(d.flow5_pr>=.48)
        score=.24*d.flow_accel_pr+.16*d.flow20_pr+.24*r60p+.12*d.r5_pr+.10*(1-d.vol20_pr)+.14*p['flow_breadth']
    elif name=='residual_sponsor_low_vol':
        mask=base&(r20p>=.58)&(d.vol20_pr<=.48)&(d.dist_ma20.between(-.05,.10))
        score=.24*d.flow_accel_pr+.18*d.flow20_pr+.22*r20p+.16*(1-d.vol20_pr)+.10*d.amount20_pr+.10*p['calm']
    elif name=='residual_sponsor_liquidity':
        mask=base&(d.amount20_pr>=.70)&(r20p>=.60)&not_ext
        score=.22*d.flow_accel_pr+.16*d.flow20_pr+.24*r20p+.18*d.amount20_pr+.10*(1-d.vol20_pr)+.10*p['flow_breadth']
    elif name=='residual_sponsor_multihorizon':
        mask=base&(r5p>=.52)&(r20p>=.60)&(r60p>=.54)&not_ext
        score=.20*d.flow_accel_pr+.14*d.flow20_pr+.14*r5p+.24*r20p+.16*r60p+.12*p['flow_breadth']
    elif name=='residual_sponsor_breadth_confirm':
        mask=base&(r20p>=.60)&(d.r20_pr>=.52)&not_ext
        score=.22*d.flow_accel_pr+.16*d.flow20_pr+.22*r20p+.10*d.amount20_pr+.16*p['flow_breadth']+.14*p['breadth_up']
    elif name=='residual_sponsor_dispersion_leader':
        mask=base&(r20p>=.62)&(d.r20_pr>=.52)&not_ext
        score=.22*d.flow_accel_pr+.15*d.flow20_pr+.25*r20p+.10*d.amount20_pr+.16*p['dispersion']+.12*p['flow_breadth']
    elif name=='residual_sponsor_anti_crowd':
        crowded=(d.r20_pr>=.90)&(d.flow5_pr>=.90)
        mask=base&(~crowded)&(r20p>=.58)&(d.r20_pr.between(.48,.86))&(d.flow5_pr<=.88)&not_ext
        score=.24*d.flow_accel_pr+.16*d.flow20_pr+.24*r20p+.12*(1-d.flow5_pr)+.10*(1-d.vol20_pr)+.14*p['flow_breadth']
    elif name=='residual_sponsor_flow_leads_price':
        mask=base&(d.flow5_pr>=.62)&(d.flow20_pr>=.60)&(r20p.between(.48,.72))&(d.dist_ma20.between(-.06,.08))
        score=.28*d.flow_accel_pr+.22*d.flow5_pr+.18*d.flow20_pr+.14*r20p+.08*d.amount20_pr+.10*p['flow_breadth']
    elif name=='residual_sponsor_price_leads_flow':
        mask=base&(r5p>=.58)&(r20p>=.64)&(d.flow5_pr>=.48)&not_ext
        score=.20*d.flow_accel_pr+.12*d.flow20_pr+.18*r5p+.28*r20p+.12*d.r20_pr+.10*d.amount20_pr
    else: # residual_sponsor_concentrated_slots
        mask=base&(r20p>=.64)&(r60p>=.54)&(d.flow5_pr>=.54)&not_ext
        score=.24*d.flow_accel_pr+.18*d.flow20_pr+.26*r20p+.14*r60p+.10*d.amount20_pr+.08*p['flow_breadth']

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
    out=ROOT/'v24_residual_sponsor_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    audit={'version':'v24-residual-sponsor-structures','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'causal context -> residual sponsor playbook -> stock screen -> market/liquidity quality -> T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__':main()
