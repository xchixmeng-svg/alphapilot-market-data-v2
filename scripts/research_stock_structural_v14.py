#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import validate_v12_champion_portfolio_v13 as v13

ROOT=Path(__file__).resolve().parent.parent
SLOTS=(5,8,10,15,20)

HYPOTHESES={
 'low_vol_continuation': dict(hold=40, lo=1, hi=3, ctx=(.35,.25,.25,-.15), kind='lvm'),
 'low_vol_flow_confirm': dict(hold=40, lo=1, hi=3, ctx=(.20,.25,.45,-.10), kind='lvm_flow'),
 'low_vol_liquidity_expansion': dict(hold=30, lo=1, hi=3, ctx=(.30,.25,.20,-.10), kind='lvm_liq'),
 'momentum_acceleration': dict(hold=30, lo=1, hi=3, ctx=(.45,.25,.20,-.10), kind='accel'),
 'trend_pullback_recovery': dict(hold=20, lo=1, hi=3, ctx=(.20,.45,.25,-.10), kind='pullback'),
 'flow_leads_price': dict(hold=30, lo=1, hi=3, ctx=(.15,.20,.55,-.10), kind='flow_lead'),
 'quiet_coil_breakout': dict(hold=40, lo=1, hi=2, ctx=(.25,.35,.20,-.20), kind='coil'),
 'relative_strength_persistence': dict(hold=60, lo=1, hi=2, ctx=(.50,.25,.15,-.10), kind='rs_persist'),
 'broad_participation_leader': dict(hold=40, lo=1, hi=3, ctx=(.40,.40,.15,-.05), kind='broad_leader'),
 'concentrated_leader': dict(hold=40, lo=1, hi=2, ctx=(.45,.15,.20,.20), kind='concentrated'),
 'asymmetric_trend_pullback': dict(hold=30, lo=1, hi=2, ctx=(.20,.40,.25,-.15), kind='asym_pullback'),
 'flow_accel_low_extension': dict(hold=20, lo=1, hi=3, ctx=(.15,.20,.55,-.10), kind='flow_accel'),
}

def past_z(s: pd.Series)->pd.Series:
    mu=s.expanding(min_periods=40).mean().shift(1)
    sd=s.expanding(min_periods=40).std(ddof=0).shift(1).replace(0,np.nan)
    return ((s-mu)/sd).clip(-3,3).fillna(0.0)

def add_context(d: pd.DataFrame)->pd.DataFrame:
    g=d.groupby('date',sort=True)
    c=pd.DataFrame({
      'breadth20':g.r20.apply(lambda x: float((x>0).mean())),
      'breadth_ma60':g.apply(lambda x: float((x.aclose>x.ma60).mean()),include_groups=False),
      'flow_breadth':g.flow5.apply(lambda x: float((x>0).mean())),
      'dispersion':g.r20.std(ddof=0).fillna(0.0),
    }).reset_index()
    for x in ['breadth20','breadth_ma60','flow_breadth','dispersion']:
        c[x+'_z']=past_z(c[x])
    return d.merge(c,on='date',how='left')

def raw_signal(d,kind):
    if kind=='lvm':
        m=(d.r20_pr>=.65)&(d.r60_pr>=.60)&(d.vol20_pr<=.45)&(d.aclose>d.ma60)
        s=.35*d.r20_pr+.25*d.r60_pr+.25*(1-d.vol20_pr)+.15*d.amount20_pr
    elif kind=='lvm_flow':
        m=(d.r20_pr>=.60)&(d.r60_pr>=.60)&(d.vol20_pr<=.50)&(d.flow5_pr>=.55)&(d.aclose>d.ma60)
        s=.25*d.r20_pr+.20*d.r60_pr+.20*(1-d.vol20_pr)+.25*d.flow5_pr+.10*d.flow_accel_pr
    elif kind=='lvm_liq':
        m=(d.r20_pr>=.60)&(d.r60_pr>=.55)&(d.vol20_pr<=.50)&(d.amount20_pr>=.65)&(d.aclose>d.ma20)
        s=.25*d.r20_pr+.20*d.r60_pr+.20*(1-d.vol20_pr)+.25*d.amount20_pr+.10*d.flow5_pr
    elif kind=='accel':
        m=(d.r20_pr>=.70)&(d.r60_pr>=.55)&(d.r20>d.r60/3)&(d.aclose>d.ma20)&(d.vol20_pr<=.70)
        s=.40*d.r20_pr+.20*d.r60_pr+.15*d.flow5_pr+.15*d.amount20_pr+.10*(1-d.vol20_pr)
    elif kind=='pullback':
        m=(d.r60_pr>=.65)&d.r20.between(-.08,.04)&(d.aclose>d.ma60)&(d.flow5>=0)&(d.vol20_pr<=.70)
        s=.35*d.r60_pr+.20*(1-d.r20_pr)+.20*d.flow5_pr+.15*(1-d.vol20_pr)+.10*d.amount20_pr
    elif kind=='flow_lead':
        m=(d.flow5_pr>=.70)&(d.flow_accel_pr>=.65)&(d.r20_pr.between(.25,.75))&(d.vol20_pr<=.75)
        s=.30*d.flow5_pr+.30*d.flow_accel_pr+.15*(1-d.r20_pr)+.15*d.amount20_pr+.10*(1-d.vol20_pr)
    elif kind=='coil':
        m=(d.r60_pr>=.55)&(d.vol20_pr<=.30)&(d.dist_ma20.abs()<=.05)&(d.flow5_pr>=.45)&(d.aclose>d.ma60)
        s=.30*d.r60_pr+.30*(1-d.vol20_pr)+.15*d.flow5_pr+.15*d.amount20_pr+.10*(1-d.dist_ma20.abs().clip(0,.2)/.2)
    elif kind=='rs_persist':
        m=(d.r20_pr>=.75)&(d.r60_pr>=.75)&(d.vol20_pr<=.60)&(d.aclose>d.ma20)&(d.aclose>d.ma60)
        s=.40*d.r60_pr+.30*d.r20_pr+.15*(1-d.vol20_pr)+.10*d.flow5_pr+.05*d.amount20_pr
    elif kind=='broad_leader':
        m=(d.r20_pr>=.75)&(d.r60_pr>=.65)&(d.flow5_pr>=.40)&(d.aclose>d.ma60)
        s=.35*d.r20_pr+.25*d.r60_pr+.15*d.flow5_pr+.15*d.amount20_pr+.10*(1-d.vol20_pr)
    elif kind=='concentrated':
        m=(d.r20_pr>=.90)&(d.r60_pr>=.80)&(d.amount20_pr>=.70)&(d.aclose>d.ma20)
        s=.45*d.r20_pr+.25*d.r60_pr+.15*d.amount20_pr+.10*d.flow5_pr+.05*(1-d.vol20_pr)
    elif kind=='asym_pullback':
        m=(d.r60_pr>=.70)&d.r20.between(-.06,.03)&(d.vol20_pr<=.45)&(d.flow5_pr>=.40)&(d.aclose>d.ma60)
        s=.35*d.r60_pr+.25*(1-d.vol20_pr)+.15*d.flow5_pr+.15*(1-d.r20_pr)+.10*d.amount20_pr
    elif kind=='flow_accel':
        m=(d.flow_accel_pr>=.85)&(d.flow5_pr>=.60)&(d.r20_pr<=.70)&(d.vol20_pr<=.70)
        s=.40*d.flow_accel_pr+.25*d.flow5_pr+.15*(1-d.r20_pr)+.10*d.amount20_pr+.10*(1-d.vol20_pr)
    else: raise KeyError(kind)
    return m.fillna(False),s.replace([np.inf,-np.inf],np.nan).fillna(0.0)

def build_schedule(px_all,daily,name):
    h=HYPOTHESES[name]
    d=daily[(daily.date>=20210101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    d=add_context(d)
    # Separate market/liquidity quality layer only; this is NOT business-fundamental quality.
    quality=(d.amount20_pr>=.35)&(d.aclose>=5)&(d.vol20_pr<=.90)
    mask,score=raw_signal(d,h['kind'])
    z=d[mask].copy(); z['score']=score.loc[z.index]
    z=z[quality.loc[z.index]].copy()
    a,b,c,e=h['ctx']; z['ctx_affinity']=a*z.breadth20_z+b*z.breadth_ma60_z+c*z.flow_breadth_z+e*z.dispersion_z
    # Continuous context changes breadth of candidate intake, never a bull/bear trade ban.
    picked=[]
    for dt,g in z.groupby('signal_date',sort=True):
        affinity=float(g.ctx_affinity.iloc[0])
        n=h['hi'] if affinity>=0 else h['lo']
        gg=g.sort_values(['score','amount20'],ascending=[False,False]).head(n)
        picked.append(gg)
    if not picked: return pd.DataFrame()
    sig=pd.concat(picked,ignore_index=True)
    return v12.simulate(name,sig,h['hold'],px_all)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hypothesis',required=True,choices=sorted(HYPOTHESES)); args=ap.parse_args()
    name=args.hypothesis; out=ROOT/'v14_structural_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily(); sched=build_schedule(px_all,daily,name); sched.to_csv(out/'signal_schedule.csv',index=False)
    dev=[]
    for slots in SLOTS:
        nav,tr,corp=v13.simulate_portfolio(px_all,sched,slots,20210101,20241231)
        m=v13.metrics(nav,tr,4.0); dev.append({'slots':slots,**m})
    devdf=v13.rank_dev(dev); devdf.to_csv(out/'dev_slot_selection.csv',index=False); slots=int(devdf.iloc[0].slots)
    n25,t25,c25=v13.simulate_portfolio(px_all,sched,slots,20250101,20251231); m25=v13.metrics(n25,t25,1.0)
    nf,tf,cf=v13.simulate_portfolio(px_all,sched,slots,20210101,20251231); mf=v13.metrics(nf,tf,5.0)
    years={}
    for y in (2021,2022,2023,2024,2025):
        ny,ty,cy=v13.simulate_portfolio(px_all,sched,slots,y*10000+101,y*10000+1231)
        years[str(y)]=v13.metrics(ny,ty,1.0)
    n25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(t25).to_csv(out/'blind_2025_trades.csv',index=False)
    audit={'version':'v14-structural-stock-level','hypothesis':name,'selected_without_2025':True,'dev_period':'2021-2024','blind_holdout':'2025','selected_slots':slots,
      'quality_layer':'market/liquidity only, not business fundamentals','context':'causal T-close expanding-history z-scores; context alters candidate breadth rather than bull/bear hard gate',
      'execution':{'T_plus_1':True,'taiwan_ticks':True,'integer_shares':True,'common_cash_pool':True,'fees_tax':True,'nonnegative_cash':True,'official_corporate_actions':True},
      'blind_2025':m25,'full_2021_2025':mf,'yearly':years,'dev_slots':devdf.to_dict(orient='records')}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=float))
if __name__=='__main__': main()
