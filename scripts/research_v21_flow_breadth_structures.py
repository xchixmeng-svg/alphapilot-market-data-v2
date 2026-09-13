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

# Structural variants seeded from the robust DEV-positive flow-breadth family.
# These are not threshold micro-tweaks: each changes sequencing/archetype/holding logic.
HYPOTHESES={
 'flow_breadth_persistence':(20,3),
 'flow_breadth_early_turn':(15,3),
 'flow_breadth_residual_leader':(20,3),
 'flow_breadth_pullback_reload':(20,3),
 'flow_breadth_quiet_base':(30,2),
 'sponsor_then_price_confirmation':(20,3),
 'price_then_sponsor_confirmation':(20,3),
 'fresh_flow_pulse':(15,3),
 'institutional_consensus_breadth':(25,3),
 'anti_crowded_sponsor_rotation':(20,3),
 'fast_flow_capture':(10,3),
 'slow_flow_compounder':(40,2),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))

def priors(d):
    return {
      'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'dispersion':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
      'trend':pd.Series(sigmoid(d.mkt_r20_z).to_numpy(),index=d.index),
    }

def lag_by_code(d,col,n):
    return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    res20_pr=pct(res20,d.date); res60_pr=pct(res60,d.date)
    lag5_flow=pct(lag_by_code(d,'flow5_pr',5),d.date)
    lag5_acc=pct(lag_by_code(d,'flow_accel_pr',5),d.date)
    lag5_r20=pct(lag_by_code(d,'r20_pr',5),d.date)
    lag10_flow=pct(lag_by_code(d,'flow20_pr',10),d.date)
    liquid=(d.amount20_pr>=.35)&(d.close>=10)&(d.vol20_pr<=.97)&d.amount20.notna()
    not_ext=(d.dist_ma20<=.18)

    if name=='flow_breadth_persistence':
        mask=(d.flow_accel_pr>=.66)&(d.flow20_pr>=.58)&(lag5_acc>=.50)&(lag5_flow>=.50)&(d.r20_pr>=.55)
        score=.25*d.flow_accel_pr+.18*d.flow20_pr+.13*lag5_acc+.10*lag5_flow+.14*d.r20_pr+.20*p['flow_breadth']
    elif name=='flow_breadth_early_turn':
        mask=(d.flow_accel_pr>=.66)&(d.r5_pr>=.56)&(d.r20_pr.between(.40,.78))&(d.dist_ma20.between(-.08,.10))
        score=.30*d.flow_accel_pr+.20*d.r5_pr+.12*d.flow20_pr+.14*(1-d.vol20_pr)+.24*p['flow_breadth']
    elif name=='flow_breadth_residual_leader':
        mask=(d.flow_accel_pr>=.62)&(res20_pr>=.68)&(res60_pr>=.55)&(d.flow5_pr>=.52)&not_ext
        score=.25*d.flow_accel_pr+.22*res20_pr+.15*res60_pr+.13*d.flow5_pr+.10*d.amount20_pr+.15*p['flow_breadth']
    elif name=='flow_breadth_pullback_reload':
        mask=(d.flow20_pr>=.62)&(d.flow_accel_pr>=.55)&(d.dist_ma20.between(-.06,.025))&(res60_pr>=.58)&(d.r5_pr>=.42)
        score=.22*d.flow20_pr+.22*d.flow_accel_pr+.18*res60_pr+.13*d.r5_pr+.10*(1-d.vol20_pr)+.15*p['flow_breadth']
    elif name=='flow_breadth_quiet_base':
        mask=(d.vol20_pr<=.45)&(d.dist_ma20.abs()<=.07)&(d.flow20_pr>=.60)&(d.flow_accel_pr>=.60)&(d.r5_pr>=.45)
        score=.20*(1-d.vol20_pr)+.22*d.flow20_pr+.24*d.flow_accel_pr+.12*d.r5_pr+.10*d.amount20_pr+.12*p['flow_breadth']
    elif name=='sponsor_then_price_confirmation':
        # sponsorship existed five sessions earlier; price confirmation is current
        mask=(lag5_acc>=.58)&(lag5_flow>=.55)&(d.r5_pr>=.62)&(d.r20_pr>=.58)&(d.aclose>d.ma20)&not_ext
        score=.18*lag5_acc+.14*lag5_flow+.24*d.r5_pr+.18*d.r20_pr+.10*d.amount20_pr+.16*p['flow_breadth']
    elif name=='price_then_sponsor_confirmation':
        # prior price leadership followed by current flow expansion
        mask=(lag5_r20>=.62)&(d.flow_accel_pr>=.68)&(d.flow5_pr>=.58)&(d.r20_pr>=.55)&not_ext
        score=.16*lag5_r20+.28*d.flow_accel_pr+.18*d.flow5_pr+.14*d.r20_pr+.10*d.amount20_pr+.14*p['flow_breadth']
    elif name=='fresh_flow_pulse':
        # new sponsorship pulse rather than already-crowded long-duration ownership
        mask=(d.flow_accel_pr>=.74)&(lag5_acc<=.62)&(d.flow20_pr.between(.40,.78))&(d.r20_pr.between(.48,.82))&not_ext
        score=.34*d.flow_accel_pr+.15*(1-lag5_acc)+.14*d.r20_pr+.11*d.r5_pr+.10*d.amount20_pr+.16*p['flow_breadth']
    elif name=='institutional_consensus_breadth':
        mask=(d.flow5_pr>=.62)&(d.flow20_pr>=.62)&(lag10_flow>=.50)&(d.r20_pr>=.55)&(d.aclose>d.ma20)
        score=.21*d.flow5_pr+.22*d.flow20_pr+.13*lag10_flow+.14*d.r20_pr+.12*d.amount20_pr+.18*p['flow_breadth']
    elif name=='anti_crowded_sponsor_rotation':
        crowded=(d.r20_pr>=.92)&(d.flow5_pr>=.92)
        mask=(d.flow_accel_pr>=.64)&(res20_pr>=.62)&(d.r20_pr<=.88)&(~crowded)&(d.vol20_pr<=.82)
        score=.28*d.flow_accel_pr+.22*res20_pr+.14*(1-d.r20_pr)+.10*(1-d.vol20_pr)+.10*d.amount20_pr+.16*p['flow_breadth']
    elif name=='fast_flow_capture':
        mask=(d.flow_accel_pr>=.70)&(d.flow5_pr>=.60)&(d.r5_pr>=.58)&(d.r20_pr>=.55)&not_ext
        score=.30*d.flow_accel_pr+.22*d.flow5_pr+.18*d.r5_pr+.12*d.r20_pr+.08*d.amount20_pr+.10*p['flow_breadth']
    else: # slow_flow_compounder
        mask=(d.flow20_pr>=.68)&(lag10_flow>=.58)&(d.r60_pr>=.60)&(res60_pr>=.58)&(d.vol20_pr<=.85)&(d.aclose>d.ma20)
        score=.26*d.flow20_pr+.16*lag10_flow+.20*d.r60_pr+.14*res60_pr+.10*(1-d.vol20_pr)+.14*p['flow_breadth']

    raw=d[(mask&liquid).fillna(False)].copy()
    if raw.empty: return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw=raw[(raw.amount20_pr>=.35)&(raw.vol20_pr<=.97)&(raw.close>=10)&raw.amount20.notna()].copy()
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
    out=ROOT/'v21_flow_breadth_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily()
    d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    # v9/base ranks r20/r60/flow/liquidity but not the 5-day return used by several V21 structural variants.
    # Compute it causally cross-sectionally at each T close; this is feature completion, not parameter tuning.
    if 'r5_pr' not in d.columns:
        d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index()
    c=v18.build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty: raise RuntimeError(f'no signals for {name}')
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
    if not corpfull.empty: corpfull.to_csv(out/'corporate_actions.csv',index=False)
    audit={'version':'v21-flow-breadth-structures','hypothesis':name,'hold_days':hold,'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'continuous market/flow context first; structural flow-breadth playbook; stock screen; separate market/liquidity quality; T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
