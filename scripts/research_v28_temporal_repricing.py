#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
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
# trigger-v28-matrix
HYPOTHESES={
 'flow_accum_then_breakout':(20,3),
 'breakout_then_absorption_resume':(25,3),
 'retest_reclaim_with_sponsor':(25,3),
 'fresh_sponsor_after_consolidation':(25,3),
 'persistent_sponsor_not_crowded':(25,3),
 'residual_leader_low_vol':(25,3),
 'residual_leader_liquid':(20,3),
 'residual_leader_flow_consensus':(25,3),
 'short_to_medium_rs_acceleration':(20,3),
 'medium_to_long_rs_confirmation':(30,2),
 'soft_breadth_rank_only':(20,3),
 'soft_dispersion_rank_only':(20,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def sigmoid(x):
 x=np.clip(pd.Series(x,dtype=float),-6,6); return 1/(1+np.exp(-x))
def lag(d,col,n): return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def priors(d):
 return {'flow_breadth':pd.Series(sigmoid(d.flow_accel_breadth_z).to_numpy(),index=d.index),'breadth_up':pd.Series(sigmoid(d.breadth_change_z).to_numpy(),index=d.index),'dispersion':pd.Series(sigmoid(d.dispersion20_z).to_numpy(),index=d.index),'calm':pd.Series(sigmoid(-d.median_vol20_z).to_numpy(),index=d.index)}

def make_signals(d,name):
 hold,slots=HYPOTHESES[name]; p=priors(d)
 res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
 r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
 lag3_dist=lag(d,'dist_ma20',3); lag5_dist=lag(d,'dist_ma20',5)
 lag5_flow=pct(lag(d,'flow5_pr',5),d.date); lag5_acc=pct(lag(d,'flow_accel_pr',5),d.date)
 lag5_r5=pct(lag(d,'r5_pr',5),d.date); lag5_r20=pct(lag(d,'r20_pr',5),d.date)
 lag10_acc=pct(lag(d,'flow_accel_pr',10),d.date)
 quality=(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.close>=10)&d.amount20.notna()
 sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.50)
 breakout=(r20p>=.62)&(r5p>=.54)&(d.dist_ma20.between(0,.14))&(d.aclose>=d.ma60*.98)
 not_ext=d.dist_ma20.between(-.08,.15)
 if name=='flow_accum_then_breakout':
  mask=(lag10_acc>=.52)&(lag5_acc>=.58)&(lag5_flow>=.54)&breakout&(d.flow_accel_pr>=.54); score=.18*lag10_acc+.18*lag5_acc+.12*lag5_flow+.18*d.flow_accel_pr+.22*r20p+.12*d.amount20_pr
 elif name=='breakout_then_absorption_resume':
  mask=sponsor&(lag5_dist.between(.02,.13))&(lag3_dist.between(-.015,.07))&(d.dist_ma20.between(.01,.11))&(r20p>=.60)&(d.vol20_pr<=.70); score=.20*d.flow_accel_pr+.14*d.flow20_pr+.24*r20p+.14*(1-d.vol20_pr)+.10*d.amount20_pr+.10*p['calm']+.08*p['flow_breadth']
 elif name=='retest_reclaim_with_sponsor':
  mask=sponsor&(lag5_dist.between(.00,.12))&(lag3_dist.between(-.06,.015))&(d.dist_ma20.between(.005,.08))&(r5p>=.52)&(r20p>=.58); score=.21*d.flow_accel_pr+.14*d.flow20_pr+.22*r20p+.15*r5p+.10*d.amount20_pr+.10*p['flow_breadth']+.08*(1-d.vol20_pr)
 elif name=='fresh_sponsor_after_consolidation':
  mask=(lag10_acc<=.62)&(d.flow_accel_pr>=.68)&(d.flow5_pr>=.56)&(lag5_dist.abs()<=.05)&(d.dist_ma20.between(.005,.10))&(r20p>=.54); score=.28*d.flow_accel_pr+.15*d.flow5_pr+.20*r20p+.11*r5p+.12*d.amount20_pr+.08*p['breadth_up']+.06*(1-d.vol20_pr)
 elif name=='persistent_sponsor_not_crowded':
  crowded=(d.flow5_pr>=.90)&(d.r20_pr>=.90); mask=sponsor&(lag5_acc>=.50)&(lag5_flow>=.50)&(~crowded)&(r20p>=.58)&not_ext; score=.20*d.flow_accel_pr+.14*d.flow20_pr+.12*lag5_acc+.10*lag5_flow+.22*r20p+.12*d.amount20_pr+.10*(1-d.vol20_pr)
 elif name=='residual_leader_low_vol':
  mask=sponsor&(r20p>=.64)&(r60p>=.50)&(d.vol20_pr<=.55)&not_ext; score=.19*d.flow_accel_pr+.13*d.flow20_pr+.28*r20p+.12*r60p+.14*(1-d.vol20_pr)+.08*d.amount20_pr+.06*p['calm']
 elif name=='residual_leader_liquid':
  mask=sponsor&(r20p>=.64)&(r5p>=.52)&(d.amount20_pr>=.72)&not_ext; score=.19*d.flow_accel_pr+.13*d.flow20_pr+.28*r20p+.13*r5p+.17*d.amount20_pr+.10*(1-d.vol20_pr)
 elif name=='residual_leader_flow_consensus':
  mask=sponsor&(d.flow5_pr>=.54)&(lag5_flow>=.48)&(r20p>=.62)&(r60p>=.50)&not_ext; score=.18*d.flow_accel_pr+.13*d.flow20_pr+.11*d.flow5_pr+.09*lag5_flow+.25*r20p+.12*r60p+.12*d.amount20_pr
 elif name=='short_to_medium_rs_acceleration':
  mask=sponsor&(lag5_r5<=.62)&(r5p>=.64)&(r20p>=.60)&not_ext; score=.19*d.flow_accel_pr+.13*d.flow20_pr+.22*r5p+.24*r20p+.10*(1-lag5_r5)+.12*d.amount20_pr
 elif name=='medium_to_long_rs_confirmation':
  mask=sponsor&(lag5_r20>=.50)&(r20p>=.62)&(r60p>=.60)&(d.dist_ma20.between(-.04,.12)); score=.18*d.flow_accel_pr+.13*d.flow20_pr+.14*lag5_r20+.25*r20p+.18*r60p+.12*d.amount20_pr
 elif name=='soft_breadth_rank_only':
  mask=sponsor&breakout; score=.20*d.flow_accel_pr+.14*d.flow20_pr+.28*r20p+.12*r5p+.10*d.amount20_pr+.16*p['flow_breadth']
 else:
  mask=sponsor&breakout; score=.20*d.flow_accel_pr+.14*d.flow20_pr+.28*r20p+.12*r5p+.10*d.amount20_pr+.16*p['dispersion']
 raw=d[(mask&quality).fillna(False)].copy()
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
 ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); a=ap.parse_args(); name=a.hypothesis
 out=ROOT/'v28_temporal_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
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
 audit={'version':'v28-temporal-repricing','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal context -> temporal repricing playbook -> stock screen -> separate market/liquidity quality -> T+1 execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
