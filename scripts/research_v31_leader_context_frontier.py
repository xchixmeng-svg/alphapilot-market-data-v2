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
DEV_START=20230523; DEV_END=20241231; VAL_START=20250101; VAL_END=20251231
# V31 is seeded from V30 DEV survivors (leader_breakout_not_extended and context_weighted_breakout).
# 2025 has already been observed at the meta-research level, so it is validation evidence, not an untouched holdout.
HYPOTHESES={
'leader_context_rank':(20,3),'leader_flow_persistence':(20,3),'leader_liquidity_quality':(20,3),
'leader_dispersion_context':(20,3),'leader_breadth_disagreement':(20,3),'sponsor_residual_leadership':(20,3),
'leader_retest_recovery':(25,3),'compression_first_breakout':(25,3),'selective_leader_weak_breadth':(20,3),
'context_rank_only_breakout':(20,3),'two_slot_high_conviction_leader':(20,2),'leader_patient_confirmation':(30,3)}
def pct(s,by): return s.groupby(by).rank(pct=True)
def lag(d,col,n): return v28.lag(d,col,n)
def priors(d): return v28.priors(d)
def make_signals(d,name):
 hold,slots=HYPOTHESES[name]; p=priors(d)
 res5=d.r5-d.groupby('date').r5.transform('median'); res20=d.r20-d.mkt_r20; res60=d.r60-d.mkt_r60
 r5p=pct(res5,d.date); r20p=pct(res20,d.date); r60p=pct(res60,d.date)
 l3dist=lag(d,'dist_ma20',3); l5dist=lag(d,'dist_ma20',5); l5f=pct(lag(d,'flow5_pr',5),d.date); l5a=pct(lag(d,'flow_accel_pr',5),d.date); l10v=lag(d,'vol20_pr',10)
 liquid=(d.close>=10)&d.amount20.notna()&(d.amount20_pr>=.35); quality=liquid&(d.vol20_pr<=.94)
 sponsor=(d.flow_accel_pr>=.58)&(d.flow20_pr>=.50); leader=(r20p>=.60)&(r60p>=.48); breakout=(d.r5_pr>=.52)&(d.dist_ma20.between(-.01,.12))&(d.aclose>=d.ma60*.98)
 notcrowd=~((d.flow5_pr>.94)&(d.r20_pr>.94)); notext=d.dist_ma20.between(-.06,.09)
 if name=='leader_context_rank':
  mask=sponsor&leader&breakout&notcrowd; score=.18*d.flow_accel_pr+.11*d.flow20_pr+.28*r20p+.12*r60p+.09*r5p+.07*d.amount20_pr+.06*p['flow_breadth']+.05*p['breadth_up']+.04*p['dispersion']
 elif name=='leader_flow_persistence':
  mask=leader&breakout&(d.flow5_pr>=.55)&(d.flow20_pr>=.56)&(l5f>=.52)&(l5a>=.50)&notcrowd; score=.14*d.flow_accel_pr+.13*d.flow20_pr+.11*d.flow5_pr+.11*l5f+.10*l5a+.25*r20p+.10*r60p+.06*d.amount20_pr
 elif name=='leader_liquidity_quality':
  mask=sponsor&leader&breakout&(d.amount20_pr>=.70)&(d.vol20_pr<=.62)&notcrowd; score=.16*d.flow_accel_pr+.11*d.flow20_pr+.27*r20p+.11*r60p+.17*d.amount20_pr+.13*(1-d.vol20_pr)+.05*p['calm']
 elif name=='leader_dispersion_context':
  mask=sponsor&leader&breakout&notext&notcrowd; score=.17*d.flow_accel_pr+.11*d.flow20_pr+.27*r20p+.10*r60p+.09*r5p+.11*p['dispersion']+.08*p['flow_breadth']+.07*d.amount20_pr
 elif name=='leader_breadth_disagreement':
  mask=(p['breadth_up']<=.52)&(d.flow_accel_pr>=.66)&leader&breakout&notcrowd; score=.25*d.flow_accel_pr+.12*d.flow20_pr+.28*r20p+.10*r60p+.09*(1-p['breadth_up'])+.09*d.amount20_pr+.07*p['dispersion']
 elif name=='sponsor_residual_leadership':
  mask=(d.flow_accel_pr>=.64)&(d.flow20_pr>=.56)&(r20p>=.64)&(r60p.between(.50,.86))&breakout&notcrowd; score=.22*d.flow_accel_pr+.15*d.flow20_pr+.30*r20p+.12*r60p+.09*r5p+.07*d.amount20_pr+.05*p['flow_breadth']
 elif name=='leader_retest_recovery':
  mask=sponsor&(r20p>=.60)&(l5dist>=.015)&(d.dist_ma20.between(-.035,.045))&(r5p>=.50)&(l5f>=.50); score=.18*d.flow_accel_pr+.12*d.flow20_pr+.15*l5f+.25*r20p+.10*r60p+.11*r5p+.09*d.amount20_pr
 elif name=='compression_first_breakout':
  compress=(l10v<=.62)&(l3dist.between(-.045,.045)); mask=compress&(d.vol20_pr<=.62)&(d.flow_accel_pr>=.64)&leader&breakout&notcrowd; score=.23*d.flow_accel_pr+.11*d.flow20_pr+.25*r20p+.10*r60p+.12*(1-d.vol20_pr)+.08*d.amount20_pr+.06*p['calm']+.05*p['dispersion']
 elif name=='selective_leader_weak_breadth':
  mask=(p['flow_breadth']<=.48)&(d.flow_accel_pr>=.68)&leader&notext&notcrowd; score=.26*d.flow_accel_pr+.12*d.flow20_pr+.29*r20p+.10*r60p+.08*(1-p['flow_breadth'])+.08*d.amount20_pr+.07*p['dispersion']
 elif name=='context_rank_only_breakout':
  mask=sponsor&leader&breakout&notcrowd; score=.14*d.flow_accel_pr+.10*d.flow20_pr+.22*r20p+.08*r60p+.08*r5p+.08*d.amount20_pr+.10*p['flow_breadth']+.10*p['breadth_up']+.10*p['dispersion']+.10*p['calm']
 elif name=='two_slot_high_conviction_leader':
  mask=(d.flow_accel_pr>=.64)&(d.flow20_pr>=.55)&(r20p>=.66)&(r60p>=.54)&breakout&notext&notcrowd; score=.22*d.flow_accel_pr+.14*d.flow20_pr+.31*r20p+.13*r60p+.08*r5p+.07*d.amount20_pr+.05*p['dispersion']
 else:
  mask=sponsor&(r20p>=.62)&(r60p>=.55)&breakout&(d.vol20_pr<=.82)&(d.amount20_pr>=.45)&notcrowd; score=.17*d.flow_accel_pr+.13*d.flow20_pr+.27*r20p+.16*r60p+.08*r5p+.08*d.amount20_pr+.06*(1-d.vol20_pr)+.05*p['flow_breadth']
 raw=d[(mask&quality).fillna(False)].copy()
 if raw.empty:return pd.DataFrame(),hold,slots
 raw['score']=pd.Series(score,index=d.index).loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.)
 return raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots),hold,slots
def window_metrics(nav,trades,start,end):
 z=nav[(nav.date>=start)&(nav.date<=end)].copy()
 tt=[x for x in trades if start<=int(x['exit_date'])<=end]
 if z.empty:return {'end_nav':None,'total_return':None,'cagr':None,'max_dd':None,'trades':0,'win_rate':0.0,'pf':0.0}
 n=z.nav.astype(float); ret=float(n.iloc[-1]/n.iloc[0]-1); dd=float((n/n.cummax()-1).min())
 rr=pd.Series([x['return'] for x in tt],dtype=float); return {'end_nav':float(n.iloc[-1]),'total_return':ret,'cagr':ret,'max_dd':dd,'trades':int(len(rr)),'win_rate':float((rr>0).mean()) if len(rr) else 0.0,'pf':float(v13.pf(rr)) if len(rr) else 0.0}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); a=ap.parse_args(); name=a.hypothesis
 out=ROOT/'v31_leader_context_out'/name; out.mkdir(parents=True,exist_ok=True)
 px,daily=v12.base.build_daily(); d=daily[(daily.date>=20230101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
 if 'r5_pr' not in d.columns:d['r5_pr']=d.groupby('date')['r5'].rank(pct=True)
 m=d.groupby('date').agg(mkt_r20=('r20','median'),mkt_r60=('r60','median')).reset_index(); d=d.merge(m,on='date',how='left').merge(v18.build_context(d),on='date',how='left',suffixes=('','_ctx'))
 sig,hold,slots=make_signals(d,name)
 if sig.empty:raise RuntimeError('no signals '+name)
 schedule=v16.simulate_fast(name,sig,hold,v16.build_price_index(px)); schedule.to_csv(out/'signal_schedule.csv',index=False)
 navd,trd,_=v13.simulate_portfolio(px,schedule,slots,DEV_START,DEV_END,INITIAL); navreset,trreset,_=v13.simulate_portfolio(px,schedule,slots,VAL_START,VAL_END,INITIAL); navf,trf,corp=v13.simulate_portfolio(px,schedule,slots,DEV_START,VAL_END,INITIAL)
 yd=(pd.Timestamp(str(DEV_END))-pd.Timestamp(str(DEV_START))).days/365.2425; yf=(pd.Timestamp(str(VAL_END))-pd.Timestamp(str(DEV_START))).days/365.2425
 md=v13.metrics(navd,trd,yd); mr=v13.metrics(navreset,trreset,1.0); mf=v13.metrics(navf,trf,yf); rolling=window_metrics(navf,trf,VAL_START,VAL_END)
 navd.to_csv(out/'dev_nav.csv',index=False); pd.DataFrame(trd).to_csv(out/'dev_trades.csv',index=False); navreset.to_csv(out/'fresh_reset_2025_nav.csv',index=False); pd.DataFrame(trreset).to_csv(out/'fresh_reset_2025_trades.csv',index=False); navf.to_csv(out/'full_nav.csv',index=False); pd.DataFrame(trf).to_csv(out/'full_trades.csv',index=False)
 if not corp.empty:corp.to_csv(out/'corporate_actions.csv',index=False)
 audit={'version':'v31-leader-context-frontier','hypothesis':name,'hold_days':hold,'slots':slots,'selection_uses_2025':False,'meta_research_has_seen_2025':True,'development_period':[DEV_START,DEV_END],'validation_period':[VAL_START,VAL_END],'validation_note':'2025 has been repeatedly observed across prior research batches; treat as reused validation, not untouched final holdout. Final untouched evidence must come from future forward data.','architecture':'causal context -> leader/residual sponsor playbook -> stock screen -> separate liquidity/volatility quality -> T+1 execution','quality_layer':'market/liquidity quality only; no point-in-time business fundamentals claimed','dev':md,'fresh_reset_2025':mr,'rolling_continuous_2025':rolling,'full':mf,'year_returns':v28.yearly(navf),'invariants':{'nonnegative_cash':True,'shared_capital':True,'legal_tick_rounding_v13':True},'minute_gate_open':False}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,default=float))
if __name__=='__main__':main()
