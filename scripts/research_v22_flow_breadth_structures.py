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

# Structural descendants of the V20 flow_breadth_acceleration Pareto seed.
# Each changes sequencing, archetype, portfolio concentration, or holding logic.
HYPOTHESES={
 'persistent_sponsor_breakout':(20,3),
 'fresh_sponsor_turn':(15,3),
 'pullback_reaccel':(20,3),
 'calm_accumulation_release':(30,2),
 'residual_repricing_leader':(20,3),
 'broad_sponsorship_consensus':(25,3),
 'uncrowded_acceleration':(20,3),
 'lagged_sponsor_followthrough':(20,3),
 'price_leader_new_sponsor':(20,3),
 'dual_horizon_acceleration':(30,2),
 'two_stage_flow_price':(25,3),
 'adaptive_hold_high_conviction':(0,2),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
    x=np.clip(pd.Series(x,dtype=float),-6,6)
    return 1/(1+np.exp(-x))
def lag_by_code(d,col,n):
    return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)

def priors(d):
    return {
      'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),
      'dispersion':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),
      'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index),
      'breadth_up':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),
    }

def make_signals(d,name):
    hold,slots=HYPOTHESES[name]; p=priors(d)
    r5_pr=pct(d.r5,d.date)
    res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
    res20_pr=pct(res20,d.date); res60_pr=pct(res60,d.date)
    lag5_acc=pct(lag_by_code(d,'flow_accel_pr',5),d.date)
    lag5_flow5=pct(lag_by_code(d,'flow5_pr',5),d.date)
    lag5_flow20=pct(lag_by_code(d,'flow20_pr',5),d.date)
    lag5_r20=pct(lag_by_code(d,'r20_pr',5),d.date)
    liquid=(d.amount20_pr>=.35)&(d.close>=10)&(d.vol20_pr<=.97)&d.amount20.notna()
    not_ext=d.dist_ma20.between(-.10,.18)

    if name=='persistent_sponsor_breakout':
        mask=(d.flow_accel_pr>=.66)&(d.flow20_pr>=.58)&(lag5_acc>=.55)&(lag5_flow20>=.52)&(d.r20_pr>=.56)&(d.aclose>d.ma20)
        score=.24*d.flow_accel_pr+.16*d.flow20_pr+.14*lag5_acc+.10*lag5_flow20+.16*d.r20_pr+.08*d.amount20_pr+.12*p['flow_breadth']
    elif name=='fresh_sponsor_turn':
        mask=(d.flow_accel_pr>=.70)&(lag5_acc<=.58)&(d.flow20_pr.between(.40,.76))&(r5_pr>=.54)&(d.r20_pr.between(.46,.82))&not_ext
        score=.30*d.flow_accel_pr+.16*(1-lag5_acc)+.14*r5_pr+.12*d.r20_pr+.10*d.amount20_pr+.18*p['flow_breadth']
    elif name=='pullback_reaccel':
        mask=(d.flow20_pr>=.60)&(d.flow_accel_pr>=.58)&(d.dist_ma20.between(-.07,.035))&(res60_pr>=.56)&(r5_pr>=.48)
        score=.22*d.flow20_pr+.22*d.flow_accel_pr+.18*res60_pr+.13*r5_pr+.10*(1-d.vol20_pr)+.15*p['flow_breadth']
    elif name=='calm_accumulation_release':
        mask=(d.vol20_pr<=.50)&(d.flow20_pr>=.62)&(lag5_flow20>=.54)&(d.flow_accel_pr>=.58)&(r5_pr>=.54)&(d.dist_ma20<=.10)
        score=.20*(1-d.vol20_pr)+.20*d.flow20_pr+.14*lag5_flow20+.18*d.flow_accel_pr+.12*r5_pr+.16*p['calm']
    elif name=='residual_repricing_leader':
        mask=(d.flow_accel_pr>=.62)&(res20_pr>=.68)&(res60_pr>=.56)&(d.flow5_pr>=.52)&not_ext
        score=.24*d.flow_accel_pr+.24*res20_pr+.15*res60_pr+.12*d.flow5_pr+.10*d.amount20_pr+.15*p['dispersion']
    elif name=='broad_sponsorship_consensus':
        mask=(d.flow5_pr>=.60)&(d.flow20_pr>=.60)&(d.flow_accel_pr>=.58)&(d.r20_pr>=.54)&(d.aclose>d.ma20)
        score=.18*d.flow5_pr+.20*d.flow20_pr+.18*d.flow_accel_pr+.14*d.r20_pr+.10*d.amount20_pr+.20*p['breadth_up']
    elif name=='uncrowded_acceleration':
        mask=(d.flow_accel_pr>=.66)&(d.flow20_pr>=.56)&(res20_pr>=.58)&(d.r20_pr.between(.48,.86))&(d.vol20_pr<=.82)&not_ext
        score=.28*d.flow_accel_pr+.18*d.flow20_pr+.18*res20_pr+.10*(1-d.r20_pr)+.10*(1-d.vol20_pr)+.16*p['flow_breadth']
    elif name=='lagged_sponsor_followthrough':
        mask=(lag5_acc>=.60)&(lag5_flow5>=.56)&(r5_pr>=.60)&(d.r20_pr>=.56)&(d.aclose>d.ma20)&not_ext
        score=.18*lag5_acc+.14*lag5_flow5+.22*r5_pr+.16*d.r20_pr+.10*d.amount20_pr+.20*p['flow_breadth']
    elif name=='price_leader_new_sponsor':
        mask=(lag5_r20>=.64)&(d.flow_accel_pr>=.68)&(d.flow5_pr>=.56)&(d.r20_pr>=.54)&not_ext
        score=.16*lag5_r20+.28*d.flow_accel_pr+.18*d.flow5_pr+.12*d.r20_pr+.10*d.amount20_pr+.16*p['flow_breadth']
    elif name=='dual_horizon_acceleration':
        mask=(d.flow5_pr>=.62)&(d.flow20_pr>=.64)&(d.flow_accel_pr>=.60)&(d.r60_pr>=.58)&(res60_pr>=.54)&(d.vol20_pr<=.86)
        score=.18*d.flow5_pr+.22*d.flow20_pr+.20*d.flow_accel_pr+.14*d.r60_pr+.10*res60_pr+.16*p['flow_breadth']
    elif name=='two_stage_flow_price':
        stage1=.34*d.flow_accel_pr+.24*d.flow20_pr+.12*d.flow5_pr+.12*d.amount20_pr+.18*p['flow_breadth']
        stage2=.30*d.r20_pr+.18*res20_pr+.16*r5_pr+.14*(1-d.vol20_pr)+.22*p['dispersion']
        s1=pct(stage1,d.date); s2=pct(stage2,d.date)
        mask=(s1>=.66)&(s2>=.58)&not_ext
        score=.56*stage1+.44*stage2
    else: # adaptive_hold_high_conviction
        mask=(d.flow_accel_pr>=.68)&(d.flow20_pr>=.58)&(d.r20_pr>=.58)&(d.aclose>d.ma60*.96)&not_ext
        conviction=.30*d.flow_accel_pr+.22*d.flow20_pr+.18*d.r20_pr+.12*d.amount20_pr+.18*p['flow_breadth']
        score=conviction

    raw=d[(mask&liquid).fillna(False)].copy()
    if raw.empty: return pd.DataFrame(),hold,slots
    raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    if name=='adaptive_hold_high_conviction':
        q=pct(raw.score,raw.signal_date)
        raw['planned_hold']=np.where(q>=.80,40,np.where(q>=.50,20,10)).astype(int)
    else:
        raw['planned_hold']=int(hold)
    sig=(raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False])
          .groupby('signal_date',as_index=False).head(slots))
    return sig,hold,slots

def build_schedule(name,sig,bycode):
    if sig.empty:return pd.DataFrame()
    out=[]; last_exit={}
    for s in sig.sort_values(['signal_date','score'],ascending=[True,False]).itertuples(index=False):
        d=bycode.get(s.code)
        if d is None: continue
        hold=int(getattr(s,'planned_hold'))
        arr=d.date.to_numpy(); k=int(np.searchsorted(arr,int(s.signal_date),side='right'))
        if k>=len(d) or k+hold>=len(d): continue
        e=d.iloc[k]; z=d.iloc[k+hold]
        if int(e.date)<=last_exit.get(s.code,0): continue
        if not np.isfinite(e.open) or e.open<=0 or not np.isfinite(z.open) or z.open<=0: continue
        ep=float(e.open)*(1+v12.BUY_SLIP); xp=float(z.open)*(1-v12.SELL_SLIP)
        ret=(xp*(1-v12.FEE-v12.TAX))/(ep*(1+v12.FEE))-1
        path=d.iloc[k:k+hold+1]
        out.append({'config':name,'code':s.code,'signal_date':int(s.signal_date),'entry_date':int(e.date),'exit_date':int(z.date),'entry_price':ep,'exit_price':xp,'return_net':ret,'mae':float(path.low.min()/ep-1),'mfe':float(path.high.max()/ep-1),'score':float(s.score)})
        last_exit[s.code]=int(z.date)
    return pd.DataFrame(out)

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
    out=ROOT/'v22_flow_breadth_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily()
    d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index()
    c=v18.build_context(d)
    d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
    sig,hold,slots=make_signals(d,name)
    if sig.empty: raise RuntimeError(f'no signals for {name}')
    bycode=v16.build_price_index(px_all); schedule=build_schedule(name,sig,bycode)
    if schedule.empty: raise RuntimeError(f'no executable T+1 schedule for {name}')
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
    audit={'version':'v22-flow-breadth-structures','hypothesis':name,'hold_days':('adaptive' if hold==0 else hold),'slots':slots,
      'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,
      'architecture':'continuous causal market/flow context -> structural stock-level playbook -> stock screen -> separate market/liquidity quality -> T+1 execution',
      'quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed',
      'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'planned daily-bar hold exit at open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
      'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),
      'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
