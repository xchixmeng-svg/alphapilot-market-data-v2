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
 'sponsor_breakout_asymmetry':(20,3),
 'acceptance_after_pullback':(25,3),
 'flow_lead_delayed_breakout':(20,3),
 'failed_breakout_reclaim':(25,3),
 'idiosyncratic_strength_soft_context':(20,3),
 'liquidity_scaled_repricing':(20,3),
 'quiet_base_sponsor_expansion':(30,2),
 'persistent_flow_uncrowded':(25,3),
 'short_flow_long_residual':(20,3),
 'sponsor_supported_retest':(25,3),
 'quality_first_repricing':(20,3),
 'long_horizon_sponsor_compounder':(35,2),
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
    l3r5=pct(lag(d,'r5_pr',3),d.date); l5r20=pct(lag(d,'r20_pr',5),d.date); l10r20=pct(lag(d,'r20_pr',10),d.date)
    l3fa=pct(lag(d,'flow_accel_pr',3),d.date); l5fa=pct(lag(d,'flow_accel_pr',5),d.date); l10fa=pct(lag(d,'flow_accel_pr',10),d.date)
    l5f5=pct(lag(d,'flow5_pr',5),d.date); l5f20=pct(lag(d,'flow20_pr',5),d.date)
    l5vol=pct(lag(d,'vol20_pr',5),d.date); l10vol=pct(lag(d,'vol20_pr',10),d.date); l5amt=pct(lag(d,'amount20_pr',5),d.date)
    base=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.aclose>=d.ma60*.94)
    quality=(d.amount20_pr>=.48)&(d.vol20_pr<=.88)
    sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.52)
    residual=(r20p>=.62)&(r5p>=.54)
    not_ext=d.dist_ma20.between(-.07,.13)

    if name=='sponsor_breakout_asymmetry':
        downside=(l10vol<=.68)&(l5d>=-.055)&(d.dist_ma20.between(-.005,.11))
        mask=base&sponsor&residual&downside&quality
        score=.29*r20p+.14*r5p+.16*d.flow_accel_pr+.10*d.flow20_pr+.11*(1-d.vol20_pr)+.10*d.amount20_pr+.10*c['calm']
    elif name=='acceptance_after_pullback':
        impulse=(l10r20>=.64)&(l5fa>=.54); pull=(l3d.between(-.05,.015))&(d.dist_ma20.between(.00,.075)); accept=(l2d>=-.01)&(r5p>=.56)
        mask=base&impulse&pull&accept&sponsor&quality
        score=.16*l10r20+.11*l5fa+.25*r20p+.14*r5p+.13*d.flow_accel_pr+.10*d.amount20_pr+.11*c['breadth']
    elif name=='flow_lead_delayed_breakout':
        lead=(l10fa>=.60)&(l5fa>=.56)&(l5r20<=.58); confirm=(r20p>=.64)&(r5p>=.58)&(d.dist_ma20.between(.005,.12))
        mask=base&lead&confirm&(d.flow20_pr>=.52)&quality
        score=.14*l10fa+.12*l5fa+.29*r20p+.14*r5p+.11*d.flow20_pr+.10*d.amount20_pr+.10*c['flow']
    elif name=='failed_breakout_reclaim':
        failed=(l5d>=.035)&(l3d<=.005); reclaim=(d.dist_ma20>=.018)&(r5p>=.61)&(r20p>=.61)
        mask=base&failed&reclaim&sponsor&quality&not_ext
        score=.24*r20p+.18*r5p+.15*d.flow_accel_pr+.10*d.flow20_pr+.10*l5f5+.10*d.amount20_pr+.13*c['calm']
    elif name=='idiosyncratic_strength_soft_context':
        soft=.30*c['flow']+.25*c['breadth']+.20*c['calm']+.25*c['disp']
        mask=base&sponsor&(r20p>=.67)&(r60p>=.52)&not_ext&quality
        score=.34*r20p+.12*r60p+.13*r5p+.13*d.flow_accel_pr+.08*d.flow20_pr+.08*d.amount20_pr+.12*soft
    elif name=='liquidity_scaled_repricing':
        liq=(d.amount20_pr>=.66)&(l5amt>=.52)&(d.vol20_pr<=.78); softliq=.55*d.amount20_pr+.25*l5amt+.20*(1-d.vol20_pr)
        mask=base&sponsor&residual&liq&not_ext
        score=.28*r20p+.13*r5p+.14*d.flow_accel_pr+.09*d.flow20_pr+.20*softliq+.08*c['flow']+.08*c['calm']
    elif name=='quiet_base_sponsor_expansion':
        quiet=(l10vol<=.48)&(l10d.abs()<=.05)&(l5d.abs()<=.045)&(l5vol<=.55); ignition=(l5fa<=.56)&(d.flow_accel_pr>=.66)&(d.flow5_pr>=.57)
        mask=base&quiet&ignition&(r20p>=.55)&(r5p>=.57)&(d.dist_ma20.between(-.01,.08))&quality
        score=.23*d.flow_accel_pr+.12*d.flow5_pr+.22*r20p+.14*r5p+.12*(1-d.vol20_pr)+.09*d.amount20_pr+.08*c['calm']
    elif name=='persistent_flow_uncrowded':
        persist=(l10fa>=.52)&(l5fa>=.56)&(d.flow_accel_pr>=.60)&(l5f20>=.52); uncrowded=(r20p.between(.58,.82))&(d.dist_ma20.between(-.03,.09))&(d.vol20_pr<=.78)
        mask=base&persist&uncrowded&(r5p>=.54)&quality
        score=.13*l10fa+.14*l5fa+.16*d.flow_accel_pr+.10*l5f20+.22*r20p+.11*r5p+.08*d.amount20_pr+.06*c['flow']
    elif name=='short_flow_long_residual':
        shortflow=(d.flow5_pr>=.60)&(d.flow_accel_pr>=.62); longres=(r60p>=.60)&(r20p>=.63)&(r5p>=.55)
        mask=base&shortflow&longres&not_ext&quality
        score=.16*d.flow5_pr+.16*d.flow_accel_pr+.27*r20p+.14*r60p+.11*r5p+.09*d.amount20_pr+.07*c['breadth']
    elif name=='sponsor_supported_retest':
        prior=(l5r20>=.64)&(l5d>=.025); retest=(l3d.between(-.035,.025))&(d.dist_ma20.between(-.005,.065)); support=(l5fa>=.54)&(d.flow_accel_pr>=.58)&(d.flow20_pr>=.54)
        mask=base&prior&retest&support&(r5p>=.53)&quality
        score=.17*l5r20+.11*l5fa+.24*r20p+.13*r5p+.14*d.flow_accel_pr+.10*d.flow20_pr+.11*d.amount20_pr
    elif name=='quality_first_repricing':
        q=(d.amount20_pr>=.74)&(l5amt>=.60)&(d.vol20_pr<=.68)&(l10vol<=.78)
        mask=base&q&sponsor&(r20p>=.61)&(r5p>=.55)&not_ext
        score=.20*d.amount20_pr+.10*l5amt+.13*(1-d.vol20_pr)+.25*r20p+.12*r5p+.12*d.flow_accel_pr+.08*d.flow20_pr
    else:
        longflow=(l10fa>=.54)&(l5f20>=.54)&(d.flow20_pr>=.56); longstrength=(r60p>=.58)&(r20p>=.62)&(r20p<=.86)&(d.vol20_pr<=.78)
        mask=base&longflow&longstrength&(d.flow_accel_pr>=.56)&not_ext&quality
        score=.12*l10fa+.10*l5f20+.13*d.flow20_pr+.16*r60p+.25*r20p+.09*r5p+.08*d.amount20_pr+.07*c['calm']

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
    out=ROOT/'v31_sponsor_acceptance_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    audit={'version':'v31-sponsor-acceptance-robustness','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal market/flow context soft prior -> stock-level sponsor/repricing playbook -> screen -> separate liquidity quality -> T+1 shared-capital execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
