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
'sponsor_absorption_repricing':(25,3),'fresh_sponsor_inflection':(20,3),'sponsor_pullback_reentry':(25,3),
'multi_horizon_flow_alignment':(30,3),'residual_strength_flow_alignment':(20,3),'liquidity_quality_sponsor_leader':(25,3),
'low_vol_sponsor_accumulation':(30,3),'flow_breadth_stock_divergence':(20,3),'anti_crowded_sponsor_accumulation':(25,3),
'capital_concentration_first_wave':(20,2),'relative_strength_not_extended':(25,3),'price_absorption_sponsor_persistence':(30,3)}
def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return v28.lag(d,col,n)
def priors(d): return v28.priors(d)
def make_signals(d,name):
 hold,slots=HYPOTHESES[name]; p=priors(d)
 res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
 r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
 l5r20=pct(lag(d,'r20',5)-lag(d,'mkt_r20',5),d.date); l5r5=pct(lag(d,'r5_pr',5),d.date)
 l5f=pct(lag(d,'flow5_pr',5),d.date); l5a=pct(lag(d,'flow_accel_pr',5),d.date); l10f=pct(lag(d,'flow20_pr',10),d.date)
 l5dist=lag(d,'dist_ma20',5); l5vol=lag(d,'vol20_pr',5)
 liquid=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35); quality=liquid&(d.vol20_pr<=.94)
 sponsor=(d.flow_accel_pr>=.58)&(d.flow20_pr>=.50); notext=d.dist_ma20.between(-.08,.10); notcrowd=~((d.flow5_pr>.94)&(d.r20_pr>.94))
 if name=='sponsor_absorption_repricing':
  mask=sponsor&notext&(r20p.between(.48,.78))&(d.r5_pr>=.48)&(d.vol20_pr<=.68)&(d.flow5_pr>=.52)
  score=.24*d.flow_accel_pr+.16*d.flow20_pr+.18*r20p+.12*r5p+.12*(1-d.vol20_pr)+.10*d.amount20_pr+.08*p['flow_breadth']
 elif name=='fresh_sponsor_inflection':
  fresh=(l5a<=.58)&(d.flow_accel_pr>=.68)&(d.flow5_pr>=.55); mask=fresh&notext&(r20p.between(.42,.76))&notcrowd
  score=.30*d.flow_accel_pr+.12*d.flow5_pr+.16*r20p+.12*r5p+.10*(1-l5a)+.10*d.amount20_pr+.10*p['dispersion']
 elif name=='sponsor_pullback_reentry':
  pull=(l5r20>=.62)&(l5dist>=.02)&(d.dist_ma20.between(-.045,.035))&(r5p>=.42); mask=sponsor&pull&(r20p>=.55)&notcrowd
  score=.21*d.flow_accel_pr+.14*d.flow20_pr+.18*l5r20+.18*r20p+.09*r5p+.10*d.amount20_pr+.10*(1-d.vol20_pr)
 elif name=='multi_horizon_flow_alignment':
  mask=(d.flow5_pr>=.56)&(d.flow20_pr>=.58)&(l10f>=.50)&(d.flow_accel_pr>=.56)&(r20p>=.54)&(r60p.between(.45,.86))&notext
  score=.15*d.flow5_pr+.18*d.flow20_pr+.12*l10f+.16*d.flow_accel_pr+.19*r20p+.11*r60p+.09*d.amount20_pr
 elif name=='residual_strength_flow_alignment':
  mask=sponsor&(r5p>=.55)&(r20p>=.62)&(r60p>=.48)&notext&notcrowd
  score=.19*d.flow_accel_pr+.13*d.flow20_pr+.17*r5p+.27*r20p+.10*r60p+.08*d.amount20_pr+.06*p['dispersion']
 elif name=='liquidity_quality_sponsor_leader':
  mask=sponsor&(d.amount20_pr>=.70)&(d.vol20_pr<=.72)&(r20p>=.55)&notext
  score=.20*d.flow_accel_pr+.13*d.flow20_pr+.22*r20p+.18*d.amount20_pr+.15*(1-d.vol20_pr)+.07*r5p+.05*p['calm']
 elif name=='low_vol_sponsor_accumulation':
  mask=sponsor&(d.vol20_pr<=.48)&(l5vol<=.55)&(r20p.between(.45,.76))&(d.dist_ma20.between(-.06,.06))
  score=.22*d.flow_accel_pr+.16*d.flow20_pr+.17*r20p+.18*(1-d.vol20_pr)+.10*l5f+.09*d.amount20_pr+.08*p['calm']
 elif name=='flow_breadth_stock_divergence':
  selective=(p['flow_breadth']<=.48)&(d.flow_accel_pr>=.68)&(d.flow20_pr>=.56); mask=selective&(r20p>=.55)&notext&notcrowd
  score=.26*d.flow_accel_pr+.15*d.flow20_pr+.24*r20p+.10*r5p+.10*(1-p['flow_breadth'])+.09*d.amount20_pr+.06*p['dispersion']
 elif name=='anti_crowded_sponsor_accumulation':
  mask=sponsor&(d.flow5_pr.between(.50,.84))&(r20p.between(.52,.82))&(d.r5_pr<=.86)&notext&(d.vol20_pr<=.82)
  score=.22*d.flow_accel_pr+.15*d.flow20_pr+.22*r20p+.12*r5p+.11*d.amount20_pr+.10*(1-d.vol20_pr)+.08*p['calm']
 elif name=='capital_concentration_first_wave':
  first=(l5a<=.55)&(d.flow_accel_pr>=.70)&(r20p.between(.48,.72))&(r5p>=.52); mask=first&notext&(d.amount20_pr>=.45)&notcrowd
  score=.32*d.flow_accel_pr+.13*d.flow5_pr+.22*r20p+.13*r5p+.10*d.amount20_pr+.10*p['dispersion']
 elif name=='relative_strength_not_extended':
  mask=(r20p>=.68)&(r60p.between(.52,.84))&(r5p>=.52)&sponsor&(d.dist_ma20.between(-.03,.075))&(d.r5_pr<=.90)&notcrowd
  score=.17*d.flow_accel_pr+.12*d.flow20_pr+.14*r5p+.30*r20p+.11*r60p+.09*d.amount20_pr+.07*(1-d.vol20_pr)
 else:
  absorb=(l5f>=.52)&(l5r5<=.68)&(d.dist_ma20.between(-.04,.055))&(d.vol20_pr<=.72); persistent=(d.flow20_pr>=.58)&(l10f>=.52)&(d.flow_accel_pr>=.56)
  mask=absorb&persistent&(r20p>=.52)&notcrowd
  score=.17*d.flow_accel_pr+.18*d.flow20_pr+.11*l10f+.11*l5f+.20*r20p+.10*r5p+.07*d.amount20_pr+.06*(1-d.vol20_pr)
 raw=d[(mask&quality).fillna(False)].copy()
 if raw.empty:return pd.DataFrame(),hold,slots
 raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.)
 return raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots),hold,slots
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); a=ap.parse_args(); name=a.hypothesis
 out=ROOT/'v29_sponsor_repricing_out'/name; out.mkdir(parents=True,exist_ok=True)
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
 audit={'version':'v29-sponsor-repricing-structures','hypothesis':name,'hold_days':hold,'slots':slots,'selection_uses_2025':False,'development_period':[DEV_START,DEV_END],'blind_period':[BLIND_START,BLIND_END],'architecture':'causal context -> sponsor/repricing playbook -> stock screen -> separate liquidity/volatility quality -> T+1 execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','dev':md,'blind_2025':mb,'full':mf,'year_returns':v28.yearly(navf),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
