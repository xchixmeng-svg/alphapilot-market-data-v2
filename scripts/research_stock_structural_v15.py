#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import research_stock_structural_v14 as v14
import research_open_tournament_technical_v12 as v12
import validate_v12_champion_portfolio_v13 as v13

ROOT=Path(__file__).resolve().parent.parent
SLOTS=(5,8,10,15,20)
DEV_START=20230523
DEV_END=20241231
HOLDOUT_START=20250101
HOLDOUT_END=20251231
FULL_START=20230523
FULL_END=20251231

HYPOTHESES={
 'coil_flow_release': dict(hold=40, lo=1, hi=3, kind='coil_flow_release', ctx='broad_flow'),
 'coil_liquidity_release': dict(hold=30, lo=1, hi=3, kind='coil_liquidity_release', ctx='breadth'),
 'rs_stability': dict(hold=60, lo=1, hi=2, kind='rs_stability', ctx='breadth'),
 'rs_flow_stability': dict(hold=50, lo=1, hi=2, kind='rs_flow_stability', ctx='broad_flow'),
 'leader_pullback_resume': dict(hold=30, lo=1, hi=3, kind='leader_pullback_resume', ctx='breadth'),
 'flow_price_divergence': dict(hold=30, lo=1, hi=3, kind='flow_price_divergence', ctx='flow'),
 'flow_persistence_breakout': dict(hold=30, lo=1, hi=3, kind='flow_persistence_breakout', ctx='flow'),
 'multi_horizon_consensus': dict(hold=40, lo=1, hi=3, kind='multi_horizon_consensus', ctx='breadth'),
 'liquidity_expansion_followthrough': dict(hold=30, lo=1, hi=3, kind='liquidity_expansion_followthrough', ctx='breadth'),
 'broad_context_leader': dict(hold=40, lo=1, hi=4, kind='broad_context_leader', ctx='broad'),
 'concentrated_context_leader': dict(hold=40, lo=1, hi=2, kind='concentrated_context_leader', ctx='concentration'),
 'quality_compounder_technical': dict(hold=50, lo=1, hi=3, kind='quality_compounder_technical', ctx='broad_flow'),
}

def rolling_mean(g, col, w, mp):
    return g[col].transform(lambda s: s.rolling(w,min_periods=mp).mean())

def rolling_min(g, col, w, mp):
    return g[col].transform(lambda s: s.rolling(w,min_periods=mp).min())

def add_stock_history(d: pd.DataFrame)->pd.DataFrame:
    d=d.sort_values(['code','date']).copy(); g=d.groupby('code',sort=False)
    d['rs10_mean']=rolling_mean(g,'r20_pr',10,5)
    d['rs10_min']=rolling_min(g,'r20_pr',10,5)
    d['rs20_mean']=rolling_mean(g,'r60_pr',20,10)
    d['flow5_mean5']=rolling_mean(g,'flow5_pr',5,3)
    d['flow5_mean10']=rolling_mean(g,'flow5_pr',10,5)
    d['flow_pos10']=g['flow5'].transform(lambda s:(s>0).rolling(10,min_periods=5).mean())
    d['vol20_mean20']=rolling_mean(g,'vol20_pr',20,10)
    d['amount_mean5']=rolling_mean(g,'amount20_pr',5,3)
    d['amount_lag10']=g['amount20_pr'].shift(10)
    d['flow_lag5']=g['flow5_pr'].shift(5)
    d['r20_lag10']=g['r20_pr'].shift(10)
    d['amount_expansion']=(d.amount20_pr-d.amount_lag10).fillna(0)
    d['flow_reaccel']=(d.flow5_pr-d.flow_lag5).fillna(0)
    d['price_reaccel']=(d.r20_pr-d.r20_lag10).fillna(0)
    return d

def add_extra_context(d: pd.DataFrame)->pd.DataFrame:
    d=v14.add_context(d)
    rows=[]
    for dt,g in d.groupby('date',sort=True):
        r=g.r20.dropna(); f=g.flow5.dropna()
        if len(r):
            q90=float(r.quantile(.9)); med=float(r.median()); lead_spread=q90-med
        else: lead_spread=0.0
        if len(f):
            fq90=float(f.quantile(.9)); fmed=float(f.median()); flow_spread=fq90-fmed
        else: flow_spread=0.0
        rows.append((dt,lead_spread,flow_spread))
    c=pd.DataFrame(rows,columns=['date','lead_spread','flow_spread']).sort_values('date')
    c['lead_spread_z']=v14.past_z(c.lead_spread)
    c['flow_spread_z']=v14.past_z(c.flow_spread)
    return d.merge(c,on='date',how='left')

def context_affinity(d: pd.DataFrame, mode: str)->pd.Series:
    if mode=='breadth': return .55*d.breadth20_z+.45*d.breadth_ma60_z-.10*d.dispersion_z
    if mode=='flow': return .65*d.flow_breadth_z+.25*d.breadth20_z-.10*d.dispersion_z
    if mode=='broad_flow': return .35*d.breadth20_z+.25*d.breadth_ma60_z+.40*d.flow_breadth_z-.10*d.dispersion_z
    if mode=='broad': return .45*d.breadth20_z+.35*d.breadth_ma60_z+.20*d.flow_breadth_z-.15*d.lead_spread_z
    if mode=='concentration': return .45*d.lead_spread_z+.30*d.flow_spread_z+.15*d.dispersion_z-.10*d.breadth20_z
    raise KeyError(mode)

def raw_signal(d: pd.DataFrame, kind: str):
    if kind=='coil_flow_release':
        m=(d.vol20_mean20<=.38)&(d.r60_pr>=.58)&(d.flow_reaccel>=.10)&(d.flow5_pr>=.60)&(d.aclose>d.ma60)
        s=.25*d.r60_pr+.25*(1-d.vol20_mean20)+.20*d.flow5_pr+.15*d.flow_reaccel.clip(-1,1)+.15*d.amount20_pr
    elif kind=='coil_liquidity_release':
        m=(d.vol20_mean20<=.40)&(d.r60_pr>=.55)&(d.amount_expansion>=.12)&(d.amount20_pr>=.70)&(d.r20_pr.between(.45,.85))&(d.aclose>d.ma60)
        s=.25*d.r60_pr+.25*(1-d.vol20_mean20)+.25*d.amount20_pr+.15*d.amount_expansion.clip(-1,1)+.10*d.r20_pr
    elif kind=='rs_stability':
        m=(d.rs10_mean>=.72)&(d.rs10_min>=.55)&(d.rs20_mean>=.68)&(d.vol20_pr<=.62)&(d.aclose>d.ma20)&(d.aclose>d.ma60)
        s=.30*d.rs10_mean+.25*d.rs20_mean+.20*d.r20_pr+.15*(1-d.vol20_pr)+.10*d.amount20_pr
    elif kind=='rs_flow_stability':
        m=(d.rs10_mean>=.68)&(d.rs20_mean>=.65)&(d.flow5_mean10>=.60)&(d.flow_pos10>=.60)&(d.vol20_pr<=.65)&(d.aclose>d.ma60)
        s=.25*d.rs10_mean+.20*d.rs20_mean+.25*d.flow5_mean10+.15*d.flow_pos10+.10*(1-d.vol20_pr)+.05*d.amount20_pr
    elif kind=='leader_pullback_resume':
        m=(d.rs20_mean>=.72)&(d.r60_pr>=.72)&(d.r20_pr.between(.35,.68))&(d.dist_ma20.between(-.06,.025))&(d.flow_pos10>=.50)&(d.flow5_pr>=.50)&(d.aclose>d.ma60)
        s=.30*d.rs20_mean+.20*d.r60_pr+.20*d.flow5_pr+.15*d.flow_pos10+.10*(1-d.vol20_pr)+.05*d.amount20_pr
    elif kind=='flow_price_divergence':
        m=(d.flow5_mean10>=.70)&(d.flow_pos10>=.70)&(d.r20_pr.between(.25,.68))&(d.flow_reaccel>=0)&(d.vol20_pr<=.75)
        s=.30*d.flow5_mean10+.20*d.flow_pos10+.20*d.flow5_pr+.15*(1-d.r20_pr)+.10*d.amount20_pr+.05*(1-d.vol20_pr)
    elif kind=='flow_persistence_breakout':
        m=(d.flow5_mean10>=.62)&(d.flow_pos10>=.70)&(d.r20_pr>=.65)&(d.r60_pr>=.58)&(d.amount_expansion>=0)&(d.aclose>d.ma20)
        s=.25*d.r20_pr+.20*d.r60_pr+.20*d.flow5_mean10+.15*d.flow_pos10+.10*d.amount20_pr+.10*d.price_reaccel.clip(-1,1)
    elif kind=='multi_horizon_consensus':
        m=(d.r20_pr>=.68)&(d.r60_pr>=.68)&(d.rs10_mean>=.65)&(d.rs20_mean>=.65)&(d.vol20_pr<=.55)&(d.flow5_pr>=.45)&(d.aclose>d.ma60)
        s=.22*d.r20_pr+.22*d.r60_pr+.18*d.rs10_mean+.18*d.rs20_mean+.10*(1-d.vol20_pr)+.10*d.flow5_pr
    elif kind=='liquidity_expansion_followthrough':
        m=(d.amount_expansion>=.15)&(d.amount20_pr>=.75)&(d.rs10_mean>=.62)&(d.r20_pr>=.62)&(d.r60_pr>=.55)&(d.aclose>d.ma20)
        s=.25*d.amount20_pr+.20*d.amount_expansion.clip(-1,1)+.20*d.rs10_mean+.15*d.r20_pr+.10*d.r60_pr+.10*d.flow5_pr
    elif kind=='broad_context_leader':
        m=(d.rs10_mean>=.70)&(d.r20_pr>=.72)&(d.r60_pr>=.62)&(d.vol20_pr<=.65)&(d.aclose>d.ma60)
        s=.30*d.rs10_mean+.25*d.r20_pr+.20*d.r60_pr+.10*d.flow5_pr+.10*d.amount20_pr+.05*(1-d.vol20_pr)
    elif kind=='concentrated_context_leader':
        m=(d.r20_pr>=.88)&(d.r60_pr>=.78)&(d.rs10_mean>=.78)&(d.amount20_pr>=.65)&(d.aclose>d.ma20)
        s=.30*d.r20_pr+.25*d.r60_pr+.20*d.rs10_mean+.10*d.amount20_pr+.10*d.flow5_pr+.05*(1-d.vol20_pr)
    elif kind=='quality_compounder_technical':
        m=(d.rs10_mean>=.66)&(d.rs20_mean>=.66)&(d.vol20_mean20<=.48)&(d.amount_mean5>=.55)&(d.flow_pos10>=.50)&(d.aclose>d.ma60)
        s=.25*d.rs10_mean+.20*d.rs20_mean+.20*(1-d.vol20_mean20)+.15*d.amount_mean5+.10*d.flow_pos10+.10*d.flow5_pr
    else: raise KeyError(kind)
    return m.fillna(False),s.replace([np.inf,-np.inf],np.nan).fillna(0.0)

def build_schedule(px_all,daily,name):
    h=HYPOTHESES[name]
    d=daily[(daily.date>=FULL_START)&(daily.date<=FULL_END)].copy(); d['signal_date']=d.date.astype(int)
    d=add_stock_history(d); d=add_extra_context(d)
    # Separate market/liquidity quality layer only; no claim of business-fundamental quality.
    quality=(d.amount20_pr>=.35)&(d.aclose>=5)&(d.vol20_pr<=.92)
    mask,score=raw_signal(d,h['kind'])
    z=d[mask & quality].copy(); z['score']=score.loc[z.index]
    z['ctx_affinity']=context_affinity(z,h['ctx'])
    # Context only changes breadth/priority; it never disables trading via bull/bear labels.
    z['score']=z['score'] + .04*z['ctx_affinity'].clip(-2,2)
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
    name=args.hypothesis; out=ROOT/'v15_structural_out'/name; out.mkdir(parents=True,exist_ok=True)
    px_all,daily=v12.base.build_daily(); sched=build_schedule(px_all,daily,name); sched.to_csv(out/'signal_schedule.csv',index=False)
    dev=[]
    dev_years=(DEV_END-DEV_START)/10000.0
    for slots in SLOTS:
        nav,tr,corp=v13.simulate_portfolio(px_all,sched,slots,DEV_START,DEV_END)
        m=v13.metrics(nav,tr,dev_years); dev.append({'slots':slots,**m})
    devdf=v13.rank_dev(dev); devdf.to_csv(out/'dev_slot_selection.csv',index=False); slots=int(devdf.iloc[0].slots)
    n25,t25,c25=v13.simulate_portfolio(px_all,sched,slots,HOLDOUT_START,HOLDOUT_END); m25=v13.metrics(n25,t25,1.0)
    nf,tf,cf=v13.simulate_portfolio(px_all,sched,slots,FULL_START,FULL_END); mf=v13.metrics(nf,tf,(FULL_END-FULL_START)/10000.0)
    years={}
    n23,t23,c23=v13.simulate_portfolio(px_all,sched,slots,DEV_START,20231231); years['2023_partial']=v13.metrics(n23,t23,(20231231-DEV_START)/10000.0)
    for y in (2024,2025):
        ny,ty,cy=v13.simulate_portfolio(px_all,sched,slots,y*10000+101,y*10000+1231); years[str(y)]=v13.metrics(ny,ty,1.0)
    n25.to_csv(out/'blind_2025_nav.csv',index=False); pd.DataFrame(t25).to_csv(out/'blind_2025_trades.csv',index=False)
    audit={'version':'v15-structural-stock-level','hypothesis':name,'selected_without_2025':True,'dev_period':'2023-05-23..2024-12-31','blind_holdout':'2025','full_window':'2023-05-23..2025-12-31','selected_slots':slots,
      'quality_layer':'market/liquidity only, not business fundamentals','context':'causal continuous T-close breadth/flow/dispersion/concentration context; no fixed bull/bear trade gate',
      'execution':{'T_plus_1':True,'taiwan_ticks':True,'integer_shares':True,'common_cash_pool':True,'fees_tax':True,'nonnegative_cash':True,'official_corporate_actions':True},
      'blind_2025':m25,'full_2023m5_2025':mf,'yearly':years,'dev_slots':devdf.to_dict(orient='records')}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=float))
if __name__=='__main__': main()
