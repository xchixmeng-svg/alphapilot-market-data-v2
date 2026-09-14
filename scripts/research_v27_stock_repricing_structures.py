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
HYPOTHESES={
 'flow_absorption_breakout':(20,3),
 'sponsor_reacceleration_after_cooloff':(20,3),
 'residual_leader_not_extended':(25,3),
 'neglected_sponsor_repricing':(25,3),
 'price_lead_flow_follow':(20,3),
 'flow_lead_price_follow':(20,3),
 'volatility_contraction_residual_release':(25,3),
 'turnover_accumulation_breakout':(20,3),
 'institutional_divergence_reversal':(20,3),
 'multi_horizon_residual_consensus':(25,3),
 'drawdown_resilient_sponsor_momentum':(20,3),
 'early_repricing_with_liquidity_confirmation':(20,3),
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
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]
    p=priors(d)
    res5=d.r5-d.groupby('date').r5.transform('median')
    res20=d.r20-d.mkt_r20
    res60=d.r60-d.mkt_r60
    r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
    lag3_dist=lag(d,'dist_ma20',3); lag5_dist=lag(d,'dist_ma20',5); lag10_dist=lag(d,'dist_ma20',10)
    lag3_f5=pct(lag(d,'flow5_pr',3),d.date); lag5_f5=pct(lag(d,'flow5_pr',5),d.date)
    lag3_fa=pct(lag(d,'flow_accel_pr',3),d.date); lag5_fa=pct(lag(d,'flow_accel_pr',5),d.date); lag10_fa=pct(lag(d,'flow_accel_pr',10),d.date)
    lag5_f20=pct(lag(d,'flow20_pr',5),d.date)
    lag5_r5=pct(lag(d,'r5_pr',5),d.date); lag5_r20=pct(lag(d,'r20_pr',5),d.date)
    lag10_vol=pct(lag(d,'vol20_pr',10),d.date); lag5_amt=pct(lag(d,'amount20_pr',5),d.date)
    base=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.aclose>=d.ma60*.95)
    sponsor=(d.flow_accel_pr>=.58)&(d.flow20_pr>=.50)
    not_ext=d.dist_ma20.between(-.08,.16)

    if name=='flow_absorption_breakout':
        prior_pressure=(lag5_fa>=.58)&(lag5_f5>=.52)
        absorbed=(lag5_dist.between(-.05,.035))&(d.dist_ma20.between(.0,.09))&(d.vol20_pr<=.72)
        mask=base&prior_pressure&absorbed&(d.flow20_pr>=.54)&(r20p>=.60)&(r5p>=.54)
        score=.18*lag5_fa+.10*lag5_f5+.16*d.flow20_pr+.25*r20p+.12*r5p+.10*(1-d.vol20_pr)+.09*d.amount20_pr
    elif name=='sponsor_reacceleration_after_cooloff':
        cooled=(lag10_fa>=.60)&(lag5_fa<=.58)&(d.flow_accel_pr>=.66)
        mask=base&cooled&(d.flow5_pr>=.56)&(r20p>=.58)&not_ext
        score=.24*d.flow_accel_pr+.13*d.flow5_pr+.13*lag10_fa+.23*r20p+.10*r5p+.09*d.amount20_pr+.08*(1-d.vol20_pr)
    elif name=='residual_leader_not_extended':
        mask=base&(r20p>=.72)&(r60p>=.60)&(r5p>=.52)&(d.dist_ma20.between(-.02,.09))&(d.vol20_pr<=.78)&(d.flow20_pr>=.48)
        score=.34*r20p+.18*r60p+.12*r5p+.12*d.flow20_pr+.08*d.flow_accel_pr+.09*d.amount20_pr+.07*(1-d.vol20_pr)
    elif name=='neglected_sponsor_repricing':
        neglected=(r60p.between(.25,.62))&(lag5_r20<=.58)
        mask=base&sponsor&neglected&(r20p>=.60)&(r5p>=.56)&(d.dist_ma20.between(-.03,.10))
        score=.22*d.flow_accel_pr+.14*d.flow20_pr+.26*r20p+.12*r5p+.10*(1-r60p)+.09*d.amount20_pr+.07*p['flow_breadth']
    elif name=='price_lead_flow_follow':
        price_first=(lag5_r20>=.62)&(lag5_fa<=.55)&(d.flow_accel_pr>=.64)
        mask=base&price_first&(r20p>=.62)&(r5p>=.54)&(d.flow5_pr>=.54)&not_ext
        score=.18*lag5_r20+.22*d.flow_accel_pr+.12*d.flow5_pr+.25*r20p+.10*r5p+.07*d.amount20_pr+.06*(1-d.vol20_pr)
    elif name=='flow_lead_price_follow':
        flow_first=(lag5_fa>=.64)&(lag5_r20<=.58)&(r20p>=.62)
        mask=base&flow_first&(d.flow20_pr>=.52)&(r5p>=.56)&(d.dist_ma20.between(-.015,.11))
        score=.20*lag5_fa+.14*d.flow20_pr+.28*r20p+.14*r5p+.08*d.amount20_pr+.08*(1-d.vol20_pr)+.08*p['breadth_up']
    elif name=='volatility_contraction_residual_release':
        contraction=(lag10_vol<=.50)&(lag5_dist.abs()<=.045)
        mask=base&contraction&(r20p>=.62)&(r5p>=.58)&(d.flow20_pr>=.52)&(d.dist_ma20.between(0,.10))
        score=.25*r20p+.14*r5p+.14*d.flow20_pr+.10*d.flow_accel_pr+.15*(1-d.vol20_pr)+.12*d.amount20_pr+.10*p['calm']
    elif name=='turnover_accumulation_breakout':
        accum=(lag5_amt<=.62)&(d.amount20_pr>=.72)&(d.flow20_pr>=.52)
        mask=base&accum&(r20p>=.62)&(r5p>=.55)&(d.dist_ma20.between(-.01,.10))&(d.vol20_pr<=.82)
        score=.21*d.amount20_pr+.15*(1-lag5_amt)+.14*d.flow20_pr+.24*r20p+.11*r5p+.08*d.flow_accel_pr+.07*(1-d.vol20_pr)
    elif name=='institutional_divergence_reversal':
        divergence=(lag5_f20>=.58)&(lag5_r20<=.42)
        mask=base&divergence&(r20p>=.54)&(r5p>=.58)&(d.flow20_pr>=.58)&(d.dist_ma20.between(-.04,.08))
        score=.18*lag5_f20+.20*d.flow20_pr+.12*d.flow_accel_pr+.22*r20p+.14*r5p+.08*d.amount20_pr+.06*p['flow_breadth']
    elif name=='multi_horizon_residual_consensus':
        mask=base&(r5p>=.58)&(r20p>=.66)&(r60p>=.58)&(d.flow20_pr>=.50)&not_ext&(d.vol20_pr<=.82)
        score=.18*r5p+.32*r20p+.18*r60p+.11*d.flow20_pr+.08*d.flow_accel_pr+.07*d.amount20_pr+.06*(1-d.vol20_pr)
    elif name=='drawdown_resilient_sponsor_momentum':
        quality=(d.vol20_pr<=.58)&(d.amount20_pr>=.58)&(d.dist_ma20.between(-.025,.085))
        mask=base&quality&sponsor&(r20p>=.62)&(r5p>=.54)
        score=.20*d.flow_accel_pr+.13*d.flow20_pr+.27*r20p+.10*r5p+.14*(1-d.vol20_pr)+.10*d.amount20_pr+.06*p['calm']
    else:
        early=(lag5_r20<=.56)&(r20p>=.60)&(r5p>=.60)
        liquidity=(d.amount20_pr>=.65)&(d.amount20_pr>=lag5_amt)
        mask=base&early&liquidity&(d.flow_accel_pr>=.60)&(d.flow5_pr>=.54)&(d.dist_ma20.between(-.02,.09))
        score=.22*r20p+.16*r5p+.18*d.flow_accel_pr+.10*d.flow5_pr+.14*d.amount20_pr+.08*(1-lag5_amt)+.07*(1-d.vol20_pr)+.05*p['breadth_up']

    raw=d[mask.fillna(False)].copy()
    if raw.empty: return pd.DataFrame(),hold,slots
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
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); args=ap.parse_args(); name=args.hypothesis
    out=ROOT/'v27_stock_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index(); c=v18.build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty:raise RuntimeError(f'no signals for {name}')
    bycode=v16.build_price_index(px_all); schedule=v16.simulate_fast(name,sig,hold,bycode); schedule.to_csv(out/'signal_schedule.csv',index=False)
    navdev,trdev,_=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL)
    nav25,tr25,_=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL)
    navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
    yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
    md=v13.metrics(navdev,trdev,yd); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,yf)
    navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False); nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False); navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
    if not corpfull.empty:corpfull.to_csv(out/'corporate_actions.csv',index=False)
    audit={'version':'v27-stock-repricing-structures','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal market/flow context as soft prior -> independent stock-level repricing playbook -> separate liquidity/quality filter -> T+1 shared-capital execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
