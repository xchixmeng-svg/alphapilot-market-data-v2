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
 'quiet_base_flow_ignition':(25,3),
 'failed_breakout_reclaim':(20,3),
 'sponsor_accumulation_before_price':(25,3),
 'price_absorption_after_flow_shock':(20,3),
 'relative_strength_reset_reentry':(25,3),
 'liquidity_expansion_with_sponsorship':(20,3),
 'low_vol_accumulation_release':(25,3),
 'short_horizon_flow_reversal':(15,3),
 'long_horizon_laggard_catchup':(25,3),
 'multi_speed_repricing':(20,3),
 'soft_context_breakout_confirmation':(20,3),
 'sponsor_quality_concentration':(25,2),
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
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; c=context(d)
    res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    l3d=lag(d,'dist_ma20',3); l5d=lag(d,'dist_ma20',5); l10d=lag(d,'dist_ma20',10)
    l3fa=pct(lag(d,'flow_accel_pr',3),d.date); l5fa=pct(lag(d,'flow_accel_pr',5),d.date); l10fa=pct(lag(d,'flow_accel_pr',10),d.date)
    l5f5=pct(lag(d,'flow5_pr',5),d.date); l5f20=pct(lag(d,'flow20_pr',5),d.date)
    l5r5=pct(lag(d,'r5_pr',5),d.date); l5r20=pct(lag(d,'r20_pr',5),d.date); l10r20=pct(lag(d,'r20_pr',10),d.date)
    l5amt=pct(lag(d,'amount20_pr',5),d.date); l10vol=pct(lag(d,'vol20_pr',10),d.date)
    base=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.aclose>=d.ma60*.94)
    quality=(d.amount20_pr>=.45)&(d.vol20_pr<=.90)
    not_ext=d.dist_ma20.between(-.08,.14)

    if name=='quiet_base_flow_ignition':
        setup=(l10vol<=.52)&(l5d.abs()<=.04)&(d.flow_accel_pr>=.66)&(d.flow5_pr>=.56)
        mask=base&setup&(r5p>=.56)&(r20p>=.56)&(d.dist_ma20.between(-.01,.08))
        score=.24*d.flow_accel_pr+.12*d.flow5_pr+.22*r20p+.12*r5p+.12*(1-d.vol20_pr)+.10*d.amount20_pr+.08*c['calm']
    elif name=='failed_breakout_reclaim':
        prior_fail=(l5d>=.07)&(l3d<=.025); reclaim=(d.dist_ma20>=.035)&(r5p>=.60)
        mask=base&prior_fail&reclaim&(d.flow20_pr>=.54)&(r20p>=.58)&quality
        score=.26*r20p+.16*r5p+.16*d.flow20_pr+.12*d.flow_accel_pr+.10*d.amount20_pr+.10*(1-d.vol20_pr)+.10*c['breadth']
    elif name=='sponsor_accumulation_before_price':
        sponsor_first=(l10fa>=.60)&(l5f20>=.56)&(l10r20<=.54)&(r20p>=.58)
        mask=base&sponsor_first&(d.flow20_pr>=.56)&(r5p>=.56)&not_ext
        score=.18*l10fa+.14*l5f20+.26*r20p+.12*r5p+.12*d.flow20_pr+.10*d.amount20_pr+.08*c['flow']
    elif name=='price_absorption_after_flow_shock':
        shock=(l5fa>=.72)&(l5f5>=.62); absorb=d.dist_ma20.between(-.025,.055)&(d.vol20_pr<=.68)
        mask=base&shock&absorb&(d.flow20_pr>=.50)&(r20p>=.58)&(r5p>=.52)
        score=.18*l5fa+.12*l5f5+.24*r20p+.12*r5p+.12*d.flow20_pr+.12*(1-d.vol20_pr)+.10*d.amount20_pr
    elif name=='relative_strength_reset_reentry':
        reset=(l5r20>=.72)&(l5r5<=.48)&(r5p>=.58)&(r20p>=.64)
        mask=base&reset&(d.flow20_pr>=.48)&(d.dist_ma20.between(-.035,.075))&quality
        score=.18*l5r20+.18*r5p+.28*r20p+.10*d.flow20_pr+.10*d.flow_accel_pr+.09*d.amount20_pr+.07*(1-d.vol20_pr)
    elif name=='liquidity_expansion_with_sponsorship':
        liq=(l5amt<=.55)&(d.amount20_pr>=.75); sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.54)
        mask=base&liq&sponsor&(r20p>=.58)&(r5p>=.54)&not_ext
        score=.20*d.amount20_pr+.12*(1-l5amt)+.18*d.flow_accel_pr+.12*d.flow20_pr+.24*r20p+.08*r5p+.06*c['flow']
    elif name=='low_vol_accumulation_release':
        accum=(l10vol<=.48)&(l10fa>=.54)&(d.flow20_pr>=.54); release=(r5p>=.60)&(r20p>=.60)&(d.dist_ma20.between(.0,.09))
        mask=base&accum&release&(d.vol20_pr<=.62)&quality
        score=.24*r20p+.14*r5p+.16*d.flow20_pr+.10*l10fa+.16*(1-d.vol20_pr)+.12*d.amount20_pr+.08*c['calm']
    elif name=='short_horizon_flow_reversal':
        reversal=(l3fa<=.38)&(d.flow_accel_pr>=.68)&(d.flow5_pr>=.60)
        mask=base&reversal&(r5p>=.58)&(r20p>=.52)&(d.dist_ma20.between(-.04,.08))&quality
        score=.24*d.flow_accel_pr+.14*d.flow5_pr+.18*r5p+.20*r20p+.10*d.amount20_pr+.08*(1-d.vol20_pr)+.06*c['breadth']
    elif name=='long_horizon_laggard_catchup':
        laggard=(r60p.between(.20,.52))&(l10r20<=.50)&(r20p>=.60)&(r5p>=.58)
        mask=base&laggard&(d.flow20_pr>=.54)&(d.flow_accel_pr>=.56)&not_ext
        score=.28*r20p+.14*r5p+.16*(1-r60p)+.14*d.flow20_pr+.10*d.flow_accel_pr+.10*d.amount20_pr+.08*c['flow']
    elif name=='multi_speed_repricing':
        accel=(l10r20<=.52)&(l5r20>=.56)&(r20p>=.66)&(r5p>=.60)
        mask=base&accel&(d.flow20_pr>=.50)&(d.flow_accel_pr>=.56)&not_ext&quality
        score=.30*r20p+.16*r5p+.10*(1-l10r20)+.10*l5r20+.12*d.flow20_pr+.10*d.flow_accel_pr+.07*d.amount20_pr+.05*c['breadth']
    elif name=='soft_context_breakout_confirmation':
        breakout=(r20p>=.64)&(r5p>=.56)&(d.dist_ma20.between(.0,.10)); soft=(.55*c['flow']+.45*c['breadth'])
        mask=base&breakout&(d.flow20_pr>=.50)&quality
        score=.25*r20p+.13*r5p+.14*d.flow20_pr+.10*d.flow_accel_pr+.12*d.amount20_pr+.08*(1-d.vol20_pr)+.18*soft
    else:
        sponsor=(d.flow20_pr>=.62)&(d.flow_accel_pr>=.60)&(l5f20>=.58); concentration=(d.amount20_pr>=.62)&(d.vol20_pr<=.72)
        mask=base&sponsor&concentration&(r20p>=.62)&(r5p>=.54)&not_ext
        score=.18*d.flow20_pr+.16*d.flow_accel_pr+.10*l5f20+.28*r20p+.10*r5p+.10*d.amount20_pr+.08*(1-d.vol20_pr)

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
    out=ROOT/'v28_state_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    audit={'version':'v28-state-repricing-structures','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal context soft prior -> state-transition/repricing playbook -> stock screen -> separate liquidity quality -> T+1 shared-capital execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
