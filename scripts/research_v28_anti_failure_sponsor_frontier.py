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
import research_v27_dispersion_context_frontier as v27

ROOT=Path(__file__).resolve().parent.parent
INITIAL=1_300_000.0
DEV_START=20230523; DEV_END=20241231
BLIND_START=20250101; BLIND_END=20251231

# V28 is seeded from the prior Pareto evidence, especially the high-OOS-return residual/sponsor
# family and V27 breakout_failure_avoidance's substantially lower blind drawdown. These are
# structural arms, not threshold nudges: sequencing, absorption/retest, concentration,
# compression-release, cross-horizon sponsorship, and realization path differ materially.
# 2025 remains blind and is never used to choose thresholds/weights.
HYPOTHESES={
 'anti_failure_sponsor_breakout':(20,3),
 'sponsor_breakout_retest':(25,3),
 'quiet_absorption_release':(25,3),
 'residual_acceleration_breakout':(20,3),
 'persistent_sponsor_quality':(30,2),
 'flow_leads_price_confirmation':(20,3),
 'price_leads_flow_confirmation':(20,3),
 'dispersion_selective_breakout':(20,3),
 'calm_to_expansion_breakout':(20,3),
 'cross_horizon_sponsor_leader':(30,3),
 'two_slot_anti_failure_concentration':(25,2),
 'fast_realization_anti_crowding':(12,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return v27.lag(d,col,n)
def priors(d): return v27.priors(d)

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    lag3_r5=pct(lag(d,'r5_pr',3),d.date)
    lag5_r20=pct(lag(d,'r20',5)-lag(d,'mkt_r20',5),d.date)
    lag5_r60=pct(lag(d,'r60',5)-lag(d,'mkt_r60',5),d.date)
    lag5_flow=pct(lag(d,'flow5_pr',5),d.date)
    lag5_acc=pct(lag(d,'flow_accel_pr',5),d.date)
    lag10_flow=pct(lag(d,'flow20_pr',10),d.date)
    lag5_dist=lag(d,'dist_ma20',5)
    lag10_dist=lag(d,'dist_ma20',10)
    lag5_vol=lag(d,'vol20_pr',5)
    lag5_amt=lag(d,'amount20_pr',5)
    disp_lag=lag(pd.DataFrame({'code':d.code,'date':d.date,'disp':p['dispersion']}),'disp',5)
    fb_lag=lag(pd.DataFrame({'code':d.code,'date':d.date,'fb':p['flow_breadth']}),'fb',5)

    liquid=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)
    sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.50)
    leader=(r20p>=.60)&(d.r5_pr>=.52)&(d.aclose>=d.ma60*.98)
    sane_extension=d.dist_ma20.between(-.02,.12)&(d.r5_pr<=.95)
    not_crowded=~((d.flow5_pr>.94)&(d.r20_pr>.94))
    quality=liquid&(d.vol20_pr<=.94)

    if name=='anti_failure_sponsor_breakout':
        mask=sponsor&leader&sane_extension&not_crowded&(p['dispersion']>=.50)&(d.vol20_pr<=.88)
        score=.24*d.flow_accel_pr+.14*d.flow20_pr+.26*r20p+.10*r5p+.08*p['dispersion']+.10*d.amount20_pr+.08*(1-d.vol20_pr)
    elif name=='sponsor_breakout_retest':
        # prior extension, then controlled retest that still preserves leadership and sponsorship
        prior_break=(lag5_dist>=.035)&(lag5_r20>=.60)
        retest=d.dist_ma20.between(-.01,.055)&(r20p>=.58)&(r5p>=.45)
        mask=sponsor&prior_break&retest&not_crowded&(d.vol20_pr<=.90)
        score=.22*d.flow_accel_pr+.13*d.flow20_pr+.18*lag5_r20+.18*r20p+.09*r5p+.10*d.amount20_pr+.10*(1-d.vol20_pr)
    elif name=='quiet_absorption_release':
        # flow stays constructive through a quiet base; current price begins to leave that base
        quiet_base=(lag5_vol<=.60)&(lag5_dist.between(-.025,.035))&(lag5_flow>=.50)
        release=(d.dist_ma20.between(.01,.09))&(r5p>=.56)&(r20p>=.58)
        mask=quiet_base&release&(d.flow_accel_pr>=.58)&(d.flow20_pr>=.52)&(d.amount20_pr>=.40)&not_crowded
        score=.20*d.flow_accel_pr+.15*d.flow20_pr+.14*lag5_flow+.20*r20p+.13*r5p+.10*d.amount20_pr+.08*p['calm']
    elif name=='residual_acceleration_breakout':
        # leadership must be accelerating, not merely already high
        accel=(r5p>=.58)&(r20p>=.62)&(r20p>=lag5_r20)&(r5p>=lag3_r5)
        mask=sponsor&accel&sane_extension&not_crowded&(p['dispersion']>=.48)
        score=.21*d.flow_accel_pr+.13*d.flow20_pr+.22*r20p+.17*r5p+.10*(r20p-lag5_r20).clip(lower=0)+.09*d.amount20_pr+.08*p['dispersion']
    elif name=='persistent_sponsor_quality':
        # slower compounder: sponsorship persists across horizons and liquidity/volatility remain clean
        persistent=(d.flow20_pr>=.58)&(lag10_flow>=.52)&(d.flow5_pr>=.52)&(d.flow_accel_pr>=.55)
        mask=persistent&(r20p>=.58)&(r60p>=.54)&(r60p<=.90)&(d.vol20_pr<=.82)&(d.amount20_pr>=.45)&(p['calm']>=.45)
        score=.16*d.flow_accel_pr+.17*d.flow20_pr+.10*lag10_flow+.22*r20p+.14*r60p+.11*d.amount20_pr+.10*(1-d.vol20_pr)
    elif name=='flow_leads_price_confirmation':
        # capital sponsorship arrives first; only enter after price catches up causally
        lead_flow=(lag5_acc>=.58)&(lag5_flow>=.52)
        confirm=(d.flow_accel_pr>=.58)&(r5p>=.57)&(r20p>=.60)&(r20p>=lag5_r20)
        mask=lead_flow&confirm&sane_extension&not_crowded
        score=.15*lag5_acc+.10*lag5_flow+.19*d.flow_accel_pr+.22*r20p+.15*r5p+.11*d.amount20_pr+.08*p['dispersion']
    elif name=='price_leads_flow_confirmation':
        # price leadership exists first; fresh sponsorship is the second-stage confirmation
        lead_price=(lag5_r20>=.60)&(lag3_r5>=.52)
        confirm=(r20p>=.60)&(d.flow_accel_pr>=.66)&(d.flow5_pr>=.56)
        mask=lead_price&confirm&sane_extension&not_crowded&(d.vol20_pr<=.90)
        score=.18*lag5_r20+.09*lag3_r5+.22*d.flow_accel_pr+.12*d.flow5_pr+.20*r20p+.11*d.amount20_pr+.08*(1-d.vol20_pr)
    elif name=='dispersion_selective_breakout':
        # high cross-sectional opportunity set, but no hard bull/bear gate
        selective=(p['dispersion']>=.55)&(p['flow_breadth']<=.68)&(p['dispersion']>=disp_lag)
        mask=sponsor&leader&selective&sane_extension&not_crowded
        score=.20*d.flow_accel_pr+.13*d.flow20_pr+.26*r20p+.11*r5p+.14*p['dispersion']+.08*(1-p['flow_breadth'])+.08*d.amount20_pr
    elif name=='calm_to_expansion_breakout':
        # stock-level compression transitions into controlled expansion with sponsorship
        compressed=(lag5_vol<=.58)&(lag10_dist.between(-.035,.03))
        expansion=(r5p>=.58)&(r20p>=.58)&(d.dist_ma20.between(.015,.095))
        mask=compressed&expansion&(d.flow_accel_pr>=.60)&(d.flow20_pr>=.50)&not_crowded&(d.vol20_pr<=.88)
        score=.21*d.flow_accel_pr+.13*d.flow20_pr+.21*r20p+.16*r5p+.10*(1-lag5_vol.fillna(1))+.11*d.amount20_pr+.08*p['calm']
    elif name=='cross_horizon_sponsor_leader':
        # medium-term leader with confirmed but non-exhausted long-horizon strength
        cross=(r5p>=.52)&(r20p>=.60)&(r60p.between(.55,.86))&(r20p>=lag5_r20)&(r60p>=lag5_r60*.95)
        mask=sponsor&cross&sane_extension&not_crowded&(d.vol20_pr<=.86)
        score=.18*d.flow_accel_pr+.13*d.flow20_pr+.14*r5p+.24*r20p+.13*r60p+.10*d.amount20_pr+.08*(1-d.vol20_pr)
    elif name=='two_slot_anti_failure_concentration':
        # concentrated portfolio only when the stock itself satisfies the anti-failure structure
        mask=sponsor&leader&sane_extension&not_crowded&(d.flow5_pr>=.58)&(d.amount20_pr>=.45)&(d.vol20_pr<=.86)&(p['dispersion']>=.50)
        score=.25*d.flow_accel_pr+.14*d.flow5_pr+.12*d.flow20_pr+.26*r20p+.09*r5p+.08*d.amount20_pr+.06*(1-d.vol20_pr)
    else: # fast_realization_anti_crowding
        # short realization path, requiring fresh residual acceleration and avoiding exhaustion
        fresh=(r5p>=.62)&(r20p>=.58)&(d.flow_accel_pr>=.62)&(d.flow5_pr>=.54)
        mask=fresh&sane_extension&not_crowded&(d.amount20_pr>=.38)&(d.vol20_pr<=.90)&(p['dispersion']>=.48)
        score=.22*d.flow_accel_pr+.12*d.flow5_pr+.21*r20p+.20*r5p+.09*d.amount20_pr+.08*p['dispersion']+.08*(1-d.vol20_pr)

    raw=d[(mask&quality).fillna(False)].copy()
    if raw.empty:return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sig=(raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False])
           .groupby('signal_date',as_index=False).head(slots))
    return sig,hold,slots

def yearly(nav): return v27.yearly(nav)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); args=ap.parse_args(); name=args.hypothesis
    out=ROOT/'v28_anti_failure_sponsor_out'/name; out.mkdir(parents=True,exist_ok=True)
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
    audit={'version':'v28-anti-failure-sponsor-frontier','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'causal context -> setup/playbook -> stock screen -> separate market/liquidity quality -> T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'seed_rationale':'combine prior residual/sponsor return frontier with V27 anti-failure drawdown evidence without using 2025 for selection',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'setup-specific fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
