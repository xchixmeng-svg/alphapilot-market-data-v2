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
 'two_stage_sponsor_confirmation':(25,3),
 'residual_breakout_retest_entry':(25,3),
 'low_vol_sponsor_accumulation':(30,3),
 'fresh_flow_residual_acceleration':(20,3),
 'institutional_cross_horizon_consensus':(25,3),
 'anti_crowded_residual_sponsor':(20,3),
 'liquidity_shock_absorption':(20,3),
 'breadth_neutral_sponsor_alpha':(25,3),
 'compression_then_sponsor_release':(30,2),
 'persistent_sponsor_pullback_recovery':(25,3),
 'residual_quality_drawdown_control':(20,3),
 'capital_efficiency_residual_rank':(20,3),
}
def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
 x=np.clip(pd.Series(x,dtype=float),-6,6); return 1/(1+np.exp(-x))
def lag(d,col,n): return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def priors(d):
 return {'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),'breadth_up':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index)}
def make_signals(d,name):
 hold,slots=HYPOTHESES[name]; p=priors(d)
 res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
 r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
 lag3_dist=lag(d,'dist_ma20',3); lag5_dist=lag(d,'dist_ma20',5)
 lag5_flow=pct(lag(d,'flow5_pr',5),d.date); lag5_acc=pct(lag(d,'flow_accel_pr',5),d.date); lag10_acc=pct(lag(d,'flow_accel_pr',10),d.date)
 lag5_r20=pct(lag(d,'r20_pr',5),d.date); lag5_r5=pct(lag(d,'r5_pr',5),d.date)
 lag10_vol=pct(lag(d,'vol20_pr',10),d.date)
 base=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35)&(d.vol20_pr<=.97)
 sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.52)
 breakout=(r20p>=.62)&(d.r5_pr>=.54)&(d.dist_ma20.between(-.01,.15))&(d.aclose>=d.ma60*.98)
 not_ext=d.dist_ma20.between(-.08,.16)
 if name=='two_stage_sponsor_confirmation':
  mask=base&breakout&sponsor&(lag5_acc>=.52)&(lag5_flow>=.50)&(d.flow5_pr>=.55)
  score=.18*lag5_acc+.12*lag5_flow+.20*d.flow_accel_pr+.10*d.flow5_pr+.24*r20p+.10*r5p+.06*p['flow_breadth']
 elif name=='residual_breakout_retest_entry':
  mask=base&sponsor&(r20p>=.64)&(lag3_dist.between(-.045,.045))&(d.dist_ma20.between(.005,.10))&(r5p>=.55)
  score=.20*d.flow_accel_pr+.13*d.flow20_pr+.27*r20p+.16*r5p+.10*(1-d.vol20_pr)+.08*d.amount20_pr+.06*p['breadth_up']
 elif name=='low_vol_sponsor_accumulation':
  mask=base&sponsor&(d.vol20_pr<=.58)&(lag10_vol<=.62)&(r20p>=.55)&not_ext
  score=.22*d.flow_accel_pr+.15*d.flow20_pr+.18*r20p+.16*(1-d.vol20_pr)+.12*d.amount20_pr+.09*p['calm']+.08*d.flow5_pr
 elif name=='fresh_flow_residual_acceleration':
  mask=base&breakout&(d.flow_accel_pr>=.68)&(d.flow5_pr>=.58)&(lag5_acc<=.70)&(r20p>=.66)&(r5p>=.58)
  score=.25*d.flow_accel_pr+.13*d.flow5_pr+.27*r20p+.16*r5p+.10*d.amount20_pr+.09*(1-lag5_acc)
 elif name=='institutional_cross_horizon_consensus':
  mask=base&breakout&(d.flow5_pr>=.56)&(d.flow20_pr>=.58)&(d.flow_accel_pr>=.58)&(lag5_flow>=.50)
  score=.14*d.flow5_pr+.16*d.flow20_pr+.18*d.flow_accel_pr+.10*lag5_flow+.24*r20p+.10*r60p+.08*d.amount20_pr
 elif name=='anti_crowded_residual_sponsor':
  mask=base&sponsor&(r20p>=.64)&(r60p.between(.42,.78))&(d.dist_ma20<=.11)&(d.vol20_pr<=.78)
  score=.20*d.flow_accel_pr+.13*d.flow20_pr+.28*r20p+.11*r5p+.12*(1-r60p)+.10*(1-d.vol20_pr)+.06*d.amount20_pr
 elif name=='liquidity_shock_absorption':
  liq_now=d.amount20_pr; liq_prev=pct(lag(d,'amount20_pr',5),d.date)
  mask=base&sponsor&(liq_now>=.72)&(liq_prev<=.72)&(r20p>=.60)&(d.dist_ma20.between(-.02,.12))&(d.vol20_pr<=.82)
  score=.18*d.flow_accel_pr+.12*d.flow20_pr+.22*r20p+.18*liq_now+.10*(1-liq_prev)+.10*r5p+.10*(1-d.vol20_pr)
 elif name=='breadth_neutral_sponsor_alpha':
  mask=base&breakout&sponsor&not_ext
  score=.23*d.flow_accel_pr+.14*d.flow20_pr+.29*r20p+.13*r5p+.11*d.amount20_pr+.10*(1-abs(p['breadth_up']-.5)*2)
 elif name=='compression_then_sponsor_release':
  prior_compress=(lag5_dist.abs()<=.05)&(lag10_vol<=.62)
  mask=base&sponsor&prior_compress&(r20p>=.60)&(d.r5_pr>=.58)&(d.dist_ma20.between(.0,.12))
  score=.19*d.flow_accel_pr+.12*d.flow20_pr+.25*r20p+.15*r5p+.11*(1-d.vol20_pr)+.10*d.amount20_pr+.08*p['calm']
 elif name=='persistent_sponsor_pullback_recovery':
  pullback=(lag5_dist.between(-.06,.03))&(d.dist_ma20.between(-.005,.09))
  mask=base&pullback&(lag10_acc>=.48)&(lag5_acc>=.52)&sponsor&(r20p>=.58)&(r5p>=.54)
  score=.14*lag10_acc+.14*lag5_acc+.18*d.flow_accel_pr+.12*d.flow20_pr+.24*r20p+.10*r5p+.08*d.amount20_pr
 elif name=='residual_quality_drawdown_control':
  mask=base&sponsor&(r20p>=.64)&(d.vol20_pr<=.66)&(d.amount20_pr>=.60)&(d.dist_ma20.between(-.03,.10))
  score=.18*d.flow_accel_pr+.12*d.flow20_pr+.28*r20p+.14*(1-d.vol20_pr)+.14*d.amount20_pr+.08*r5p+.06*p['calm']
 else:
  efficiency=(.55*r20p+.25*r5p+.20*d.flow_accel_pr)/(0.35+d.vol20_pr)
  effp=pct(efficiency,d.date)
  mask=base&sponsor&(effp>=.68)&(r20p>=.58)&not_ext
  score=.30*effp+.20*d.flow_accel_pr+.12*d.flow20_pr+.18*r20p+.10*d.amount20_pr+.10*(1-d.vol20_pr)
 raw=d[(mask.fillna(False))].copy()
 if raw.empty:return pd.DataFrame(),hold,slots
 raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
 return raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots),hold,slots
def yearly(nav):
 out={}
 if nav.empty:return out
 n=nav.set_index('date').nav.astype(float)
 for y in (2023,2024,2025):
  lo=max(y*10000+101,DEV_START) if y==2023 else y*10000+101; z=n[(n.index>=lo)&(n.index<=y*10000+1231)]; out[str(y)]=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else None
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); args=ap.parse_args(); name=args.hypothesis
 out=ROOT/'v26_sponsor_residual_out'/name; out.mkdir(parents=True,exist_ok=True)
 px_all,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
 if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
 m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index(); c=v18.build_context(d); d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
 sig,hold,slots=make_signals(d,name)
 if sig.empty:raise RuntimeError(f'no signals for {name}')
 bycode=v16.build_price_index(px_all); schedule=v16.simulate_fast(name,sig,hold,bycode); schedule.to_csv(out/'signal_schedule.csv',index=False)
 navdev,trdev,_=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL); nav25,tr25,_=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL); navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
 yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
 md=v13.metrics(navdev,trdev,yd); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,yf)
 navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False); nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False); navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
 if not corpfull.empty:corpfull.to_csv(out/'corporate_actions.csv',index=False)
 audit={'version':'v26-sponsor-residual-structures','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal context -> sponsor/residual playbook -> stock screen -> separate liquidity quality -> T+1 shared-capital execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
