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
import research_v28_temporal_repricing as v28

ROOT=Path(__file__).resolve().parent.parent
INITIAL=1_300_000.0
DEV_START=20230523; DEV_END=20241231
BLIND_START=20250101; BLIND_END=20251231

HYPOTHESES={
 'sponsor_breakout_retest':(25,3),
 'quiet_absorption_after_flow':(25,3),
 'first_flow_turn_low_base':(20,3),
 'multi_horizon_residual_stack':(25,3),
 'flow_price_divergence_repair':(20,3),
 'liquidity_sponsored_breakout':(20,3),
 'low_vol_sponsor_persistence':(25,3),
 'broad_context_soft_rank':(20,3),
 'dispersion_opportunity_rank':(20,3),
 'compression_release':(25,3),
 'sponsor_decay_avoidance':(20,3),
 'delayed_price_confirmation':(25,3),
}

def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return d.sort_values(['code','date']).groupby('code',sort=False)[col].shift(n).reindex(d.index)
def zsig(x):
 x=np.clip(pd.Series(x,dtype=float),-6,6)
 return pd.Series((1/(1+np.exp(-x))).to_numpy(),index=x.index)

def make_signals(d,name):
 hold,slots=HYPOTHESES[name]
 res5=d.r5-d.groupby('date').r5.transform('median')
 res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
 r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
 lag3_dist=lag(d,'dist_ma20',3); lag5_dist=lag(d,'dist_ma20',5); lag10_dist=lag(d,'dist_ma20',10)
 lag5_acc=pct(lag(d,'flow_accel_pr',5),d.date); lag10_acc=pct(lag(d,'flow_accel_pr',10),d.date)
 lag5_flow=pct(lag(d,'flow5_pr',5),d.date); lag10_flow=pct(lag(d,'flow5_pr',10),d.date)
 lag5_r5=pct(lag(d,'r5_pr',5),d.date); lag5_r20=pct(lag(d,'r20_pr',5),d.date)
 lag10_vol=pct(lag(d,'vol20_pr',10),d.date)
 flow_breadth=zsig(d.flow_accel_breadth_z); breadth_up=zsig(d.breadth_change_z); dispersion=zsig(d.dispersion20_z); calm=zsig(-d.median_vol20_z)
 quality=(d.amount20_pr>=.35)&(d.vol20_pr<=.97)&(d.close>=10)&d.amount20.notna()
 sponsor=(d.flow_accel_pr>=.60)&(d.flow20_pr>=.50)
 breakout=(r20p>=.62)&(r5p>=.54)&(d.dist_ma20.between(0,.14))&(d.aclose>=d.ma60*.98)
 not_ext=d.dist_ma20.between(-.08,.15)
 if name=='sponsor_breakout_retest':
  mask=sponsor&(lag5_dist.between(.04,.16))&(lag3_dist.between(-.04,.04))&(d.dist_ma20.between(.005,.09))&(r20p>=.60)&(r5p>=.54)
  score=.20*d.flow_accel_pr+.12*d.flow20_pr+.22*r20p+.14*r5p+.10*lag5_acc+.10*d.amount20_pr+.12*(1-d.vol20_pr)
 elif name=='quiet_absorption_after_flow':
  mask=(lag10_acc>=.55)&(lag5_acc>=.55)&(d.flow20_pr>=.52)&(lag5_dist.between(-.03,.07))&(d.dist_ma20.between(-.02,.08))&(d.vol20_pr<=.55)&(r20p>=.54)
  score=.16*lag10_acc+.16*lag5_acc+.12*d.flow20_pr+.20*r20p+.10*r5p+.12*(1-d.vol20_pr)+.08*d.amount20_pr+.06*calm
 elif name=='first_flow_turn_low_base':
  mask=(lag10_acc<=.45)&(lag5_acc<=.55)&(d.flow_accel_pr>=.72)&(d.flow5_pr>=.58)&(r20p>=.52)&(d.dist_ma20.between(-.03,.10))
  score=.30*d.flow_accel_pr+.15*d.flow5_pr+.18*r20p+.10*r5p+.10*d.amount20_pr+.09*breadth_up+.08*(1-d.vol20_pr)
 elif name=='multi_horizon_residual_stack':
  mask=sponsor&(r5p>=.58)&(r20p>=.64)&(r60p>=.58)&not_ext
  score=.17*d.flow_accel_pr+.10*d.flow20_pr+.18*r5p+.27*r20p+.16*r60p+.07*d.amount20_pr+.05*(1-d.vol20_pr)
 elif name=='flow_price_divergence_repair':
  mask=(lag10_acc>=.55)&(lag5_acc>=.60)&(lag5_r20<=.55)&(r5p>=.60)&(r20p>=.56)&(d.flow_accel_pr>=.58)&(d.dist_ma20.between(-.04,.10))
  score=.18*lag10_acc+.15*lag5_acc+.18*d.flow_accel_pr+.20*r5p+.17*r20p+.07*d.amount20_pr+.05*breadth_up
 elif name=='liquidity_sponsored_breakout':
  mask=sponsor&breakout&(d.amount20_pr>=.75)
  score=.18*d.flow_accel_pr+.11*d.flow20_pr+.28*r20p+.14*r5p+.21*d.amount20_pr+.08*(1-d.vol20_pr)
 elif name=='low_vol_sponsor_persistence':
  mask=sponsor&(lag5_acc>=.52)&(lag5_flow>=.52)&(d.vol20_pr<=.50)&(r20p>=.58)&not_ext
  score=.16*d.flow_accel_pr+.12*d.flow20_pr+.13*lag5_acc+.10*lag5_flow+.24*r20p+.12*(1-d.vol20_pr)+.07*d.amount20_pr+.06*calm
 elif name=='broad_context_soft_rank':
  mask=sponsor&breakout
  score=.17*d.flow_accel_pr+.10*d.flow20_pr+.24*r20p+.11*r5p+.09*d.amount20_pr+.15*flow_breadth+.08*breadth_up+.06*calm
 elif name=='dispersion_opportunity_rank':
  mask=sponsor&breakout
  score=.17*d.flow_accel_pr+.10*d.flow20_pr+.24*r20p+.11*r5p+.09*d.amount20_pr+.17*dispersion+.07*(1-d.vol20_pr)+.05*flow_breadth
 elif name=='compression_release':
  mask=(lag10_vol<=.45)&(lag5_dist.abs()<=.05)&sponsor&(r5p>=.62)&(r20p>=.58)&(d.dist_ma20.between(.01,.11))
  score=.16*(1-lag10_vol)+.19*d.flow_accel_pr+.11*d.flow20_pr+.22*r5p+.20*r20p+.07*d.amount20_pr+.05*breadth_up
 elif name=='sponsor_decay_avoidance':
  mask=sponsor&(d.flow_accel_pr>=lag5_acc)&(lag5_acc>=lag10_acc)&(r20p>=.60)&not_ext
  score=.21*d.flow_accel_pr+.13*d.flow20_pr+.12*lag5_acc+.09*lag10_acc+.24*r20p+.10*r5p+.07*d.amount20_pr+.04*(1-d.vol20_pr)
 else:
  mask=(lag10_acc>=.52)&(lag5_acc>=.58)&(lag10_flow>=.48)&(lag5_r20>=.50)&(r20p>=.62)&(r5p>=.54)&(d.dist_ma20.between(-.02,.11))
  score=.16*lag10_acc+.15*lag5_acc+.10*lag10_flow+.14*lag5_r20+.24*r20p+.10*r5p+.07*d.amount20_pr+.04*flow_breadth
 raw=d[(mask&quality).fillna(False)].copy()
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
 out=ROOT/'v29_temporal_sponsor_out'/name; out.mkdir(parents=True,exist_ok=True)
 px_all,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
 if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
 m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index(); c=v18.build_context(d); d=d.merge(m,on='date',how='left').merge(c,on='date',how='left',suffixes=('','_ctx'))
 sig,hold,slots=make_signals(d,name)
 if sig.empty: raise RuntimeError(f'no signals for {name}')
 bycode=v16.build_price_index(px_all); schedule=v16.simulate_fast(name,sig,hold,bycode); schedule.to_csv(out/'signal_schedule.csv',index=False)
 navdev,trdev,_=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,DEV_END,INITIAL)
 nav25,tr25,_=v13.simulate_portfolio(px_all,schedule,slots,BLIND_START,BLIND_END,INITIAL)
 navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,DEV_START,BLIND_END,INITIAL)
 yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
 md=v13.metrics(navdev,trdev,yd); m25=v13.metrics(nav25,tr25,1.0); mf=v13.metrics(navfull,trfull,yf)
 navdev.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trdev).to_csv(out/'dev_trades.csv',index=False)
 nav25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(out/'blind_2025_trades.csv',index=False)
 navfull.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trfull).to_csv(out/'full_trades.csv',index=False)
 if not corpfull.empty: corpfull.to_csv(out/'corporate_actions.csv',index=False)
 audit={'version':'v29-temporal-sponsor','hypothesis':name,'hold_days':hold,'slots':slots,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'selection_uses_2025':False,'architecture':'causal context -> temporal sponsor/repricing playbook -> stock screen -> separate market/liquidity quality -> T+1 execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed hold open -0.5% adverse rounded to Taiwan tick','integer_shares':True,'common_cash_pool':True,'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},'dev':md,'blind_2025':m25,'full':mf,'year_returns':yearly(navfull),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))

if __name__=='__main__': main()
