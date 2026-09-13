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
DEV_START=20230523
DEV_END=20241231
BLIND_START=20250101
BLIND_END=20251231

# Each hypothesis is a structural stock-level repricing playbook. Market context is
# observed first and used only as a continuous prior/ranking term; it never blocks trading.
HYPOTHESES={
 'flow_breakout_fast_repricing':(10,3),
 'flow_breakout_slow_repricing':(40,2),
 'dispersion_flow_confluence':(20,3),
 'dispersion_pullback_reaccel':(30,2),
 'calm_sponsored_breakout':(30,2),
 'breadth_expansion_sponsor':(20,3),
 'breadth_contraction_anticrowded':(30,2),
 'flow_breadth_acceleration':(20,3),
 'flow_concentration_leader':(30,2),
 'quiet_base_to_repricing':(40,2),
 'dual_horizon_residual_flow':(20,3),
 'two_stage_flow_price_fusion':(30,3),
}

def pct(s, by):
    return s.groupby(by).rank(pct=True)

def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))

def context_priors(d):
    # All context z-scores are causal: v18 standardizes against lagged rolling history.
    return {
      'dispersion': pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
      'calm': pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
      'breadth_up': pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
      'breadth_down': pd.Series(sigmoid(-d.breadth_change_z).to_numpy(),index=d.index),
      'flow_breadth': pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'flow_concentration': pd.Series(sigmoid(d.flow_spread_z).to_numpy(),index=d.index),
      'trend': pd.Series(sigmoid(d.mkt_r20_z).to_numpy(),index=d.index),
    }

def make_signals(d: pd.DataFrame, name: str):
    hold,slots=HYPOTHESES[name]
    pri=context_priors(d)
    res20=d.r20-d.mkt_r20
    res60=d.r60-d.mkt_r60
    res20_pr=pct(res20,d.date)
    res60_pr=pct(res60,d.date)
    base_dist=d.dist_ma20.abs().clip(0,1)
    liquid=(d.amount20_pr>=.35)&(d.close>=10)&(d.vol20_pr<=.97)&d.amount20.notna()

    # 1) infer continuous context above; 2) choose this playbook/archetype;
    # 3) screen names for the playbook; 4) quality filter below; 5) T+1 simulation later.
    if name=='flow_breakout_fast_repricing':
        mask=(d.flow_accel_pr>=.72)&(d.flow5_pr>=.58)&(d.r20_pr>=.68)&(d.aclose>d.ma20)
        score=.30*d.flow_accel_pr+.22*d.flow5_pr+.24*d.r20_pr+.12*d.amount20_pr+.12*pri['flow_breadth']
    elif name=='flow_breakout_slow_repricing':
        mask=(d.flow20_pr>=.68)&(d.flow5_pr>=.58)&(d.r20_pr>=.65)&(d.r60_pr>=.55)&(d.aclose>d.ma20)
        score=.26*d.flow20_pr+.20*d.flow5_pr+.23*d.r20_pr+.16*d.r60_pr+.15*pri['trend']
    elif name=='dispersion_flow_confluence':
        mask=(res20_pr>=.74)&(d.flow_accel_pr>=.66)&(d.flow5_pr>=.54)&(d.r20_pr>=.58)
        score=.30*res20_pr+.24*d.flow_accel_pr+.16*d.flow5_pr+.14*d.amount20_pr+.16*pri['dispersion']
    elif name=='dispersion_pullback_reaccel':
        mask=(res60_pr>=.68)&(d.dist_ma20.between(-.08,.025))&(d.flow_accel_pr>=.60)&(d.r5_pr>=.50)
        score=.26*res60_pr+.22*d.flow_accel_pr+.18*d.r5_pr+.16*(1-base_dist)+.18*pri['dispersion']
    elif name=='calm_sponsored_breakout':
        mask=(d.r20_pr>=.70)&(d.vol20_pr<=.50)&(d.flow20_pr>=.60)&(d.flow5_pr>=.52)
        score=.27*d.r20_pr+.21*(1-d.vol20_pr)+.21*d.flow20_pr+.13*d.flow5_pr+.18*pri['calm']
    elif name=='breadth_expansion_sponsor':
        mask=(d.r20_pr>=.68)&(d.r60_pr>=.55)&(d.flow5_pr>=.56)&(d.aclose>d.ma20)
        score=.26*d.r20_pr+.17*d.r60_pr+.20*d.flow5_pr+.14*d.amount20_pr+.23*pri['breadth_up']
    elif name=='breadth_contraction_anticrowded':
        crowd=(d.flow5_pr>=.90)&(d.r20_pr>=.90)
        mask=(res20_pr>=.70)&(res60_pr>=.58)&(d.vol20_pr<=.72)&(~crowd)
        score=.28*res20_pr+.19*res60_pr+.16*(1-d.vol20_pr)+.15*(1-(.5*d.flow5_pr+.5*d.r20_pr))+.22*pri['breadth_down']
    elif name=='flow_breadth_acceleration':
        mask=(d.flow_accel_pr>=.68)&(d.flow20_pr>=.58)&(d.r20_pr>=.58)&(d.aclose>d.ma60*.96)
        score=.27*d.flow_accel_pr+.20*d.flow20_pr+.18*d.r20_pr+.13*d.amount20_pr+.22*pri['flow_breadth']
    elif name=='flow_concentration_leader':
        mask=(d.r60_pr>=.68)&(d.r20_pr>=.62)&(d.flow5_pr>=.60)&(d.flow20_pr>=.55)
        score=.24*d.r60_pr+.20*d.r20_pr+.20*d.flow5_pr+.14*d.flow20_pr+.22*pri['flow_concentration']
    elif name=='quiet_base_to_repricing':
        mask=(d.vol20_pr<=.42)&(base_dist<=.065)&(d.flow20_pr>=.58)&(d.flow_accel_pr>=.58)&(d.r5_pr>=.45)
        score=.22*(1-d.vol20_pr)+.20*d.flow20_pr+.20*d.flow_accel_pr+.16*d.r5_pr+.12*d.amount20_pr+.10*pri['calm']
    elif name=='dual_horizon_residual_flow':
        mask=(res20_pr>=.68)&(res60_pr>=.58)&(d.flow5_pr>=.54)&(d.flow20_pr>=.52)&(d.aclose>d.ma20)
        score=.26*res20_pr+.20*res60_pr+.18*d.flow5_pr+.14*d.flow20_pr+.12*d.amount20_pr+.10*pri['dispersion']
    else:
        stage1=.34*d.flow_accel_pr+.24*d.flow20_pr+.18*res20_pr+.12*d.amount20_pr+.12*pri['flow_breadth']
        stage2=.30*d.r20_pr+.20*d.r60_pr+.18*d.flow5_pr+.14*(1-d.vol20_pr)+.18*pri['dispersion']
        s1pr=pct(stage1,d.date); s2pr=pct(stage2,d.date)
        mask=(s1pr>=.62)&(s2pr>=.58)&((d.flow_accel_pr>=.58)|(d.flow20_pr>=.62))
        score=.52*stage1+.48*stage2

    raw=d[(mask&liquid).fillna(False)].copy()
    if raw.empty:
        return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    # Separate market/liquidity quality only. No business-fundamental claim.
    q=(raw.amount20_pr>=.35)&(raw.vol20_pr<=.97)&(raw.close>=10)&raw.amount20.notna()
    raw=raw[q].copy()
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
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); args=ap.parse_args()
    name=args.hypothesis
    out=ROOT/'v20_stock_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily()
    d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index()
    c=v18.build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty: raise RuntimeError(f'no signals for {name}')
    bycode=v16.build_price_index(px_all)
    schedule=v16.simulate_fast(name,sig,hold,bycode)
    schedule.to_csv(out/'signal_schedule.csv',index=False)
    navdev,trdev,corpdev=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL)
    nav25,tr25,corp25=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL)
    navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
    years_dev=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    years_full=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    md=v13.metrics(navdev,trdev,years_dev); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,years_full)
    yr=yearly(navfull)
    navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False)
    nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False)
    navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
    if not corpfull.empty: corpfull.to_csv(out/'corporate_actions.csv',index=False)
    audit={
      'version':'v20-stock-repricing-structures','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'causal continuous context observed first; stock-level repricing playbook selected structurally; context is a soft prior only, never a bull/bear trading gate',
      'quality_layer':'market/liquidity quality only; repo search found no validated point-in-time revenue/EPS/financial-statement layer, so business fundamentals are not claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yr,
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
