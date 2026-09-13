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

# Structural stock-level repricing hypotheses. 2025 is never used for selection.
HYPOTHESES={
 'flow_then_breakout':(20,3),
 'breakout_with_sponsorship':(40,2),
 'leader_pullback_reaccel':(30,2),
 'quiet_base_infiltration':(40,2),
 'residual_strength_confirmed':(30,3),
 'dispersion_specialist':(20,3),
 'calm_breakout':(40,2),
 'flow_divergence_repair':(30,2),
 'breadth_expansion_leader':(30,3),
 'breadth_contraction_relative_strength':(40,2),
 'anticrowded_reentry':(30,2),
 'two_stage_repricing_ensemble':(30,3),
}

def pct(s, by):
    return s.groupby(by).rank(pct=True)

def make_signals(d: pd.DataFrame, name: str):
    hold,slots=HYPOTHESES[name]
    # all features are known at T close; context z-scores are lagged rolling statistics from v18.
    res20=d.r20-d.mkt_r20
    res60=d.r60-d.mkt_r60
    res20_pr=pct(res20,d.date)
    res60_pr=pct(res60,d.date)
    base_dist=(d.dist_ma20.abs()).clip(0,1)
    near_high=(d.r20_pr>=.65)&(d.aclose>d.ma20)
    liquid=(d.amount20_pr>=.35)&(d.close>=10)&(d.vol20_pr<=.97)&d.amount20.notna()

    if name=='flow_then_breakout':
        mask=(d.flow_accel_pr>=.75)&(d.flow5_pr>=.60)&near_high
        score=.38*d.flow_accel_pr+.24*d.flow5_pr+.24*d.r20_pr+.14*d.amount20_pr
    elif name=='breakout_with_sponsorship':
        mask=(d.r20_pr>=.78)&(d.r60_pr>=.62)&(d.flow20_pr>=.60)&(d.flow5_pr>=.55)
        score=.34*d.r20_pr+.22*d.r60_pr+.26*d.flow20_pr+.18*d.amount20_pr
    elif name=='leader_pullback_reaccel':
        mask=(d.r60_pr>=.72)&(d.dist_ma20.between(-.07,.02))&(d.flow_accel_pr>=.60)&(d.flow5_pr>=.50)
        score=.30*d.r60_pr+.28*d.flow_accel_pr+.22*(1-base_dist)+.20*d.amount20_pr
    elif name=='quiet_base_infiltration':
        mask=(d.vol20_pr<=.35)&(base_dist<=.055)&(d.flow20_pr>=.62)&(d.flow_accel_pr>=.62)
        score=.28*(1-d.vol20_pr)+.30*d.flow20_pr+.27*d.flow_accel_pr+.15*d.amount20_pr
    elif name=='residual_strength_confirmed':
        mask=(res20_pr>=.72)&(res60_pr>=.62)&(d.flow5_pr>=.55)&(d.aclose>d.ma20)
        score=.34*res20_pr+.24*res60_pr+.24*d.flow5_pr+.18*d.amount20_pr
    elif name=='dispersion_specialist':
        ctx=(d.dispersion20_z>0).astype(float)
        mask=(res20_pr>=.78)&(d.flow_accel_pr>=.62)&(d.r20_pr>=.60)
        score=.38*res20_pr+.25*d.flow_accel_pr+.17*d.amount20_pr+.20*ctx
    elif name=='calm_breakout':
        calm=(1/(1+np.exp(np.clip(d.median_vol20_z,-6,6))))
        mask=(d.r20_pr>=.74)&(d.vol20_pr<=.55)&(d.flow5_pr>=.55)
        score=.34*d.r20_pr+.25*(1-d.vol20_pr)+.21*d.flow5_pr+.20*calm
    elif name=='flow_divergence_repair':
        price_weak=(d.r20_pr<=.45)&(d.aclose>=d.ma60*.90)
        mask=price_weak&(d.flow5_pr>=.72)&(d.flow_accel_pr>=.72)
        score=.34*d.flow5_pr+.34*d.flow_accel_pr+.18*d.amount20_pr+.14*(1-d.r20_pr)
    elif name=='breadth_expansion_leader':
        b=(1/(1+np.exp(-np.clip(d.breadth_change_z,-6,6))))
        mask=(d.r20_pr>=.72)&(d.flow5_pr>=.58)&(d.aclose>d.ma20)
        score=.34*d.r20_pr+.24*d.flow5_pr+.18*d.r60_pr+.24*b
    elif name=='breadth_contraction_relative_strength':
        defensive=(1/(1+np.exp(np.clip(d.breadth_change_z,-6,6))))
        mask=(res20_pr>=.76)&(res60_pr>=.62)&(d.vol20_pr<=.70)
        score=.36*res20_pr+.24*res60_pr+.18*(1-d.vol20_pr)+.22*defensive
    elif name=='anticrowded_reentry':
        crowd=(d.flow5_pr>=.92)&(d.r20_pr>=.92)
        mask=(d.r60_pr>=.68)&(d.dist_ma20.between(-.06,.015))&(d.flow20_pr>=.55)&(~crowd)
        score=.30*d.r60_pr+.24*d.flow20_pr+.22*(1-base_dist)+.24*(1-(.5*d.flow5_pr+.5*d.r20_pr))
    else:
        stage1=(.34*d.flow_accel_pr+.26*d.flow20_pr+.20*res20_pr+.20*d.amount20_pr)
        stage2=(.32*d.r20_pr+.24*d.r60_pr+.22*d.flow5_pr+.22*(1-d.vol20_pr))
        pre=(stage1>=pct(stage1,d.date).quantile(.65) if False else stage1)
        mask=((d.flow_accel_pr>=.60)|(d.flow20_pr>=.65))&((d.r20_pr>=.58)|(res20_pr>=.65))
        score=.55*stage1+.45*stage2

    raw=d[(mask&liquid).fillna(False)].copy()
    if raw.empty:
        return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    # Separate market/liquidity quality filter only; no claim of business fundamentals.
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
    out=ROOT/'v19_stock_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
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
      'version':'v19-stock-repricing-setups','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'stock-level repricing/capital-flow trigger first; continuous market context is auxiliary, never a fixed bull/bear gate',
      'quality_layer':'market/liquidity quality only; point-in-time business fundamentals not claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yr,
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
