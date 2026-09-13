#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import research_fundamental_repricing_v9 as base

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'open_tournament_v12_out'; OUT.mkdir(parents=True,exist_ok=True)
HOLDS=(3,5,10,20,40,60)
TOP_NS=(1,3,5,10)
FEE=0.000855; TAX=0.003; BUY_SLIP=0.005; SELL_SLIP=0.005


def pf(r):
    r=pd.Series(r,dtype=float); g=float(r[r>0].sum()); l=float(-r[r<0].sum())
    return (99.0 if g>0 else 0.0) if l<=1e-12 else g/l


def perf(t,label):
    if t.empty:
        return {'slice':label,'trades':0,'win_rate':0.0,'mean_net':0.0,'median_net':0.0,'pf':0.0,'event_sharpe':0.0,'event_compound':0.0,'max_event_dd':0.0,'median_mae':np.nan,'median_mfe':np.nan}
    x=t.sort_values(['signal_date','code']); r=x.return_net.astype(float).replace([np.inf,-np.inf],np.nan).dropna()
    if r.empty: return perf(pd.DataFrame(),label)
    eq=(1+r).cumprod(); dd=eq/eq.cummax()-1; sd=float(r.std(ddof=1)) if len(r)>1 else 0.0
    return {'slice':label,'trades':int(len(r)),'win_rate':float((r>0).mean()),'mean_net':float(r.mean()),'median_net':float(r.median()),'pf':float(pf(r)),'event_sharpe':float(r.mean()/sd*math.sqrt(min(len(r),252))) if sd>1e-12 else 0.0,'event_compound':float(eq.iloc[-1]-1),'max_event_dd':float(dd.min()),'median_mae':float(x.mae.median()),'median_mfe':float(x.mfe.median())}


def simulate(name,sig,hold,px_all):
    if sig.empty: return pd.DataFrame()
    bycode={c:d.sort_values('date').reset_index(drop=True) for c,d in px_all.groupby('code')}
    out=[]; last_exit={}
    for s in sig.sort_values(['signal_date','score'],ascending=[True,False]).itertuples(index=False):
        d=bycode.get(s.code)
        if d is None: continue
        arr=d.date.to_numpy(); k=int(np.searchsorted(arr,int(s.signal_date),side='right'))
        if k>=len(d) or k+hold>=len(d): continue
        e=d.iloc[k]; z=d.iloc[k+hold]
        if int(e.date)<=last_exit.get(s.code,0): continue
        if not np.isfinite(e.open) or e.open<=0 or not np.isfinite(z.open) or z.open<=0: continue
        ep=float(e.open)*(1+BUY_SLIP); xp=float(z.open)*(1-SELL_SLIP)
        ret=(xp*(1-FEE-TAX))/(ep*(1+FEE))-1
        path=d.iloc[k:k+hold+1]
        mae=float(path.low.min()/ep-1) if path.low.notna().any() else np.nan
        mfe=float(path.high.max()/ep-1) if path.high.notna().any() else np.nan
        out.append({'config':name,'code':s.code,'signal_date':int(s.signal_date),'entry_date':int(e.date),'exit_date':int(z.date),'entry_price':ep,'exit_price':xp,'return_net':ret,'mae':mae,'mfe':mfe,'score':float(s.score)})
        last_exit[s.code]=int(z.date)
    return pd.DataFrame(out)


def families(d):
    return {
      'pure_momentum':((d.r20_pr>=.80)&(d.r60_pr>=.70)&(d.aclose>d.ma20)&(d.aclose>d.ma60), .50*d.r20_pr+.35*d.r60_pr+.15*d.amount20_pr),
      'momentum_flow':((d.r20_pr>=.70)&(d.r60_pr>=.60)&(d.flow5>0)&(d.aclose>d.ma20), .35*d.r20_pr+.25*d.r60_pr+.25*d.flow5_pr+.15*d.amount20_pr),
      'low_vol_momentum':((d.r20_pr>=.65)&(d.r60_pr>=.60)&(d.vol20_pr<=.45)&(d.aclose>d.ma60), .35*d.r20_pr+.25*d.r60_pr+.25*(1-d.vol20_pr)+.15*d.amount20_pr),
      'trend_pullback':((d.r60_pr>=.65)&d.r20.between(-.10,.05)&(d.aclose>d.ma60)&(d.flow5>=0), .35*d.r60_pr+.25*(1-d.r20_pr)+.25*d.flow5_pr+.15*(1-d.vol20_pr)),
      'breakout':((d.r20_pr>=.85)&(d.r60_pr>=.70)&(d.dist_ma20>=0)&(d.flow5>=0), .45*d.r20_pr+.30*d.r60_pr+.15*d.flow5_pr+.10*d.amount20_pr),
      'flow_acceleration':((d.flow_accel_pr>=.85)&(d.flow5_pr>=.65)&(d.r20_pr<=.80), .45*d.flow_accel_pr+.30*d.flow5_pr+.15*(1-d.r20_pr)+.10*d.amount20_pr),
      'flow_reversal':((d.r20_pr<=.35)&(d.flow_accel_pr>=.75)&(d.flow5>0), .40*d.flow_accel_pr+.25*d.flow5_pr+.20*(1-d.r20_pr)+.15*d.amount20_pr),
      'deep_reversal':((d.r20<=-.08)&(d.r60<=-.12)&(d.flow5>0)&(d.flow_accel>0), .35*(1-d.r20_pr)+.25*(1-d.r60_pr)+.25*d.flow_accel_pr+.15*d.flow5_pr),
      'mean_reversion_liquid':((d.r20_pr<=.25)&(d.vol20_pr<=.70)&(d.amount20_pr>=.60), .45*(1-d.r20_pr)+.25*d.amount20_pr+.20*(1-d.vol20_pr)+.10*d.flow_accel_pr),
      'quiet_strength':((d.r60_pr>=.60)&(d.vol20_pr<=.30)&(d.dist_ma20.abs()<=.06), .35*d.r60_pr+.30*(1-d.vol20_pr)+.20*d.amount20_pr+.15*d.flow5_pr),
      'price_lag_flow':((d.r60_pr<=.50)&(d.flow5_pr>=.70)&(d.flow_accel_pr>=.60), .35*(1-d.r60_pr)+.30*d.flow5_pr+.25*d.flow_accel_pr+.10*d.amount20_pr),
      'balanced_multifactor':(d.amount20.notna(), .22*d.r20_pr+.18*d.r60_pr+.20*d.flow5_pr+.15*d.flow_accel_pr+.10*(1-d.vol20_pr)+.15*d.amount20_pr),
    }


def yearly_consistency(tr):
    vals=[]
    for y in (2021,2022,2023,2024):
        z=tr[(tr.signal_date>=y*10000+101)&(tr.signal_date<=y*10000+1231)]
        vals.append(float(z.return_net.mean()) if len(z) else np.nan)
    a=np.array(vals,dtype=float); finite=a[np.isfinite(a)]
    return {'positive_years':int((finite>0).sum()),'worst_year_mean':float(finite.min()) if len(finite) else -9.0,'year_mean_std':float(finite.std()) if len(finite)>1 else 9.0}


def main():
    px_all,daily=base.build_daily()
    d=daily[(daily.date>=20210101)&(daily.date<=20251231)].copy()
    d['signal_date']=d.date.astype(int)
    fams=families(d)
    rows=[]; trades={}
    for fam,(mask,score) in fams.items():
        raw=d[mask.fillna(False)].copy(); raw['base_score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0)
        for hold in HOLDS:
            for topn in TOP_NS:
                cfg=f'{fam}__h{hold}__n{topn}'
                z=raw.copy(); z['score']=z.base_score
                sig=z.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(topn)
                tr=simulate(cfg,sig,hold,px_all); trades[cfg]=tr
                dev=tr[(tr.signal_date>=20210101)&(tr.signal_date<=20241231)] if len(tr) else pd.DataFrame()
                s=perf(dev,'2021_2024_dev'); yc=yearly_consistency(dev)
                rows.append({'config':cfg,'family':fam,'hold_days':hold,'top_n':topn,**{'dev_'+k:v for k,v in s.items() if k!='slice'},**yc})
    dev=pd.DataFrame(rows)
    elig=dev[dev.dev_trades>=120].copy()
    if elig.empty: elig=dev.copy()
    cols=['dev_mean_net','dev_pf','dev_event_sharpe','dev_max_event_dd','dev_win_rate','dev_median_mae','positive_years','worst_year_mean']
    ranks=[]
    for c in cols:
        x=elig[c].replace([np.inf,-np.inf],np.nan)
        ranks.append(x.rank(pct=True,ascending=True).fillna(.5))
    elig['robust_score']=pd.concat(ranks,axis=1).mean(axis=1)
    elig=elig.sort_values(['robust_score','positive_years','dev_pf','dev_mean_net'],ascending=False)
    shortlist=elig.groupby('family',as_index=False).head(1).head(12).copy(); shortlist['dev_rank']=np.arange(1,len(shortlist)+1)
    hold=[]
    for r in shortlist.itertuples(index=False):
        tr=trades[r.config]; ho=tr[(tr.signal_date>=20250101)&(tr.signal_date<=20251231)] if len(tr) else pd.DataFrame(); hs=perf(ho,'2025_blind')
        hold.append({'config':r.config,'family':r.family,'hold_days':r.hold_days,'top_n':r.top_n,'dev_rank':int(r.dev_rank),'robust_score':float(r.robust_score),**{'holdout_'+k:v for k,v in hs.items() if k!='slice'}})
    hodf=pd.DataFrame(hold).sort_values('dev_rank')
    dev.to_csv(OUT/'all_dev_configs.csv',index=False); shortlist.to_csv(OUT/'preregistered_shortlist.csv',index=False); hodf.to_csv(OUT/'blind_2025_results.csv',index=False)
    champ_cfg=str(shortlist.iloc[0].config) if len(shortlist) else None; champ=None
    if champ_cfg:
        h=hodf[hodf.config==champ_cfg].iloc[0].to_dict(); validated=bool(h['holdout_trades']>=40 and h['holdout_mean_net']>0 and h['holdout_pf']>1)
        champ={'config':champ_cfg,'selected_without_2025':True,'validated_on_2025':validated,'holdout':h}
        trades[champ_cfg].to_csv(OUT/'champion_trades.csv',index=False)
        if validated:(OUT/'survivors_present.flag').write_text(champ_cfg+'\n',encoding='utf-8')
    audit={'version':'open-tournament-technical-v12','objective':'unconstrained price-volume-institutional discovery','dev_period':'2021-2024','blind_holdout':'2025','selection_never_uses_2025':True,'families':list(fams),'holds':list(HOLDS),'top_n':list(TOP_NS),'config_count':int(len(dev)),'champion':champ,'note':'Event-level discovery only. Any survivor must next be promoted to a full common-cash portfolio backtest before live use.'}
    (OUT/'tournament_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(shortlist[['dev_rank','config','dev_trades','dev_win_rate','dev_mean_net','dev_pf','dev_event_sharpe','dev_max_event_dd','positive_years','worst_year_mean','robust_score']].to_string(index=False))
    print('\nBLIND 2025\n',hodf.to_string(index=False)); print(json.dumps({'champion':champ_cfg,'validated':None if champ is None else champ['validated_on_2025']},ensure_ascii=False))

if __name__=='__main__': main()
