#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v16_friction_prescreen as v16
import validate_v12_champion_portfolio_v13 as v13
import research_v18_context_adaptive_stock_setups as v18
import research_v28_anti_failure_sponsor_frontier as v28
ROOT=Path(__file__).resolve().parent.parent; INITIAL=1_300_000.0
DEV_START=20230523; DEV_END=20241231; BLIND_START=20250101; BLIND_END=20251231
HYPOTHESES={
'low_volume_absorption_breakout':(25,3),'sponsor_lead_price_confirm':(20,3),'price_lead_sponsor_confirm':(20,3),
'quiet_base_first_concentration':(30,2),'leader_breakout_not_extended':(20,3),'breakout_retest_sponsor_hold':(25,3),
'selective_flow_weak_breadth':(20,3),'liquid_low_vol_breakout':(25,3),'multi_horizon_sponsor_consensus':(30,3),
'early_repricing_before_crowding':(20,3),'context_weighted_breakout':(25,3),'persistent_sponsor_compression_release':(30,3)}
def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return v28.lag(d,col,n)
def priors(d): return v28.priors(d)
def make_signals(d,name):
 hold,slots=HYPOTHESES[name]; p=priors(d)
 res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
 r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
 l3dist=lag(d,'dist_ma20',3); l5dist=lag(d,'dist_ma20',5); l5r5=pct(lag(d,'r5_pr',5),d.date); l5r20=pct(lag(d,'r20',5)-lag(d,'mkt_r20',5),d.date)
 l5f=pct(lag(d,'flow5_pr',5),d.date); l5a=pct(lag(d,'flow_accel_pr',5),d.date); l10f=pct(lag(d,'flow20_pr',10),d.date); l10v=lag(d,'vol20_pr',10)
 liquid=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35); quality=liquid&(d.vol20_pr<=.94)
 sponsor=(d.flow_accel_pr>=.58)&(d.flow20_pr>=.50); breakout=(r20p>=.60)&(d.r5_pr>=.52)&(d.dist_ma20.between(-.01,.14))&(d.aclose>=d.ma60*.98)
 notext=d.dist_ma20.between(-.07,.11); notcrowd=~((d.flow5_pr>.94)&(d.r20_pr>.94))
 if name=='low_volume_absorption_breakout':
  mask=sponsor&breakout&(d.vol20_pr<=.58)&(l10v<=.62)&(l3dist.between(-.045,.055))
  score=.22*d.flow_accel_pr+.14*d.flow20_pr+.25*r20p+.12*r5p+.14*(1-d.vol20_pr)+.07*d.amount20_pr+.06*p['calm']
 elif name=='sponsor_lead_price_confirm':
  mask=(l5a>=.62)&(l5f>=.55)&(d.flow_accel_pr>=.54)&breakout&notcrowd
  score=.20*l5a+.12*l5f+.17*d.flow_accel_pr+.25*r20p+.13*r5p+.08*d.amount20_pr+.05*p['flow_breadth']
 elif name=='price_lead_sponsor_confirm':
  mask=(l5r20>=.62)&(l5r5>=.54)&sponsor&(d.flow5_pr>=.55)&notext&notcrowd
  score=.20*l5r20+.11*l5r5+.20*d.flow_accel_pr+.12*d.flow20_pr+.21*r20p+.10*r5p+.06*d.amount20_pr
 elif name=='quiet_base_first_concentration':
  mask=(l5a<=.55)&(d.flow_accel_pr>=.68)&(d.vol20_pr<=.58)&(d.dist_ma20.between(-.055,.055))&(r20p.between(.48,.74))&notcrowd
  score=.30*d.flow_accel_pr+.13*d.flow5_pr+.18*r20p+.12*r5p+.12*(1-d.vol20_pr)+.09*d.amount20_pr+.06*p['dispersion']
 elif name=='leader_breakout_not_extended':
  mask=sponsor&breakout&(r60p.between(.52,.86))&(d.dist_ma20<=.085)&(d.r5_pr<=.90)&notcrowd
  score=.18*d.flow_accel_pr+.12*d.flow20_pr+.12*r5p+.30*r20p+.12*r60p+.09*d.amount20_pr+.07*(1-d.vol20_pr)
 elif name=='breakout_retest_sponsor_hold':
  mask=(l5r20>=.60)&(l5dist>=.015)&(d.dist_ma20.between(-.035,.045))&sponsor&(l5f>=.50)&(r5p>=.48)
  score=.18*d.flow_accel_pr+.13*d.flow20_pr+.16*l5f+.19*l5r20+.17*r20p+.09*r5p+.08*d.amount20_pr
 elif name=='selective_flow_weak_breadth':
  mask=(p['flow_breadth']<=.48)&(d.flow_accel_pr>=.68)&(d.flow20_pr>=.55)&(r20p>=.58)&notext&notcrowd
  score=.27*d.flow_accel_pr+.15*d.flow20_pr+.24*r20p+.10*r5p+.09*(1-p['flow_breadth'])+.09*d.amount20_pr+.06*p['dispersion']
 elif name=='liquid_low_vol_breakout':
  mask=breakout&sponsor&(d.amount20_pr>=.70)&(d.vol20_pr<=.62)&notcrowd
  score=.18*d.flow_accel_pr+.12*d.flow20_pr+.25*r20p+.18*d.amount20_pr+.15*(1-d.vol20_pr)+.07*r5p+.05*p['calm']
 elif name=='multi_horizon_sponsor_consensus':
  mask=(d.flow5_pr>=.55)&(d.flow20_pr>=.58)&(l10f>=.52)&(d.flow_accel_pr>=.58)&(r20p>=.57)&(r60p.between(.45,.86))&notext
  score=.14*d.flow5_pr+.18*d.flow20_pr+.13*l10f+.16*d.flow_accel_pr+.20*r20p+.10*r60p+.09*d.amount20_pr
 elif name=='early_repricing_before_crowding':
  mask=(l5a<=.56)&(d.flow_accel_pr>=.68)&(r20p.between(.50,.76))&(r5p>=.53)&(d.dist_ma20.between(-.03,.07))&(d.flow5_pr<=.88)
  score=.29*d.flow_accel_pr+.12*d.flow5_pr+.23*r20p+.13*r5p+.09*d.amount20_pr+.08*p['dispersion']+.06*(1-d.vol20_pr)
 elif name=='context_weighted_breakout':
  mask=sponsor&breakout&notcrowd; score=.18*d.flow_accel_pr+.12*d.flow20_pr+.23*r20p+.11*r5p+.09*d.amount20_pr+.09*p['flow_breadth']+.08*p['breadth_up']+.05*p['calm']+.05*p['dispersion']
 else:
  compress=(d.vol20_pr<=.60)&(l10v<=.64)&(d.dist_ma20.between(-.04,.07)); persistent=(d.flow20_pr>=.58)&(l10f>=.52)&(d.flow_accel_pr>=.58)
  mask=compress&persistent&(r20p>=.55)&(d.r5_pr>=.50)&notcrowd
  score=.17*d.flow_accel_pr+.18*d.flow20_pr+.12*l10f+.22*r20p+.10*r5p+.11*(1-d.vol20_pr)+.06*d.amount20_pr+.04*p['calm']
 raw=d[(mask&quality).fillna(False)].copy()
 if raw.empty:return pd.DataFrame(),hold,slots
 raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.)
 return raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots),hold,slots
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); a=ap.parse_args(); name=a.hypothesis
 out=ROOT/'v30_breakout_sequence_out'/name; out.mkdir(parents=True,exist_ok=True)
 px,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
 if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
 m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index(); d=d.merge(m,on='date',how='left').merge(v18.build_context(d),on='date',how='left',suffixes=('','_ctx'))
 sig,hold,slots=make_signals(d,name)
 if sig.empty:raise RuntimeError('no signals '+name)
 schedule=v16.simulate_fast(name,sig,hold,v16.build_price_index(px)); schedule.to_csv(out/'signal_schedule.csv',index=False)
 navd,trd,_=v13.simulate_portfolio(px,schedule,slots,DEV_START,DEV_END,INITIAL); navb,trb,_=v13.simulate_portfolio(px,schedule,slots,BLIND_START,BLIND_END,INITIAL); navf,trf,corp=v13.simulate_portfolio(px,schedule,slots,DEV_START,BLIND_END,INITIAL)
 yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(BLIND_END))-pd.Timestamp(str(DEV_START))).days/365.2425
 md=v13.metrics(navd,trd,yd); mb=v13.metrics(navb,trb,1.0); mf=v13.metrics(navf,trf,yf)
 navd.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trd).to_csv(out/'dev_trades.csv',index=False); navb.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(trb).to_csv(out/'blind_2025_trades.csv',index=False); navf.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trf).to_csv(out/'full_trades.csv',index=False)
 if not corp.empty:corp.to_csv(out/'corporate_actions.csv',index=False)
 audit={'version':'v30-breakout-sequence-structures','hypothesis':name,'hold_days':hold,'slots':slots,'selection_uses_2025':False,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'architecture':'causal context -> sequence/repricing playbook -> stock screen -> separate liquidity/volatility quality -> T+1 execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','dev':md,'blind_2025':mb,'full':mf,'year_returns':v28.yearly(navf),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
