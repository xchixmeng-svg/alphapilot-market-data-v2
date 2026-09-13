#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v14_optimistic_prescreen as v14

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'v16_friction_prescreen_out'; OUT.mkdir(parents=True,exist_ok=True)
R10_CAGR=0.131103
HOLDS=(5,10,20,40,60)
SLOTS=(1,2,3,5)


def pf(r):
    r=pd.Series(r,dtype=float)
    g=float(r[r>0].sum()); l=float(-r[r<0].sum())
    return (99.0 if g>0 else 0.0) if l<=1e-12 else g/l


def yearly_stats(tr):
    ys={}
    for y in (2021,2022,2023,2024):
        z=tr[(tr.signal_date>=y*10000+101)&(tr.signal_date<=y*10000+1231)]
        rr=z.return_net.astype(float) if len(z) else pd.Series(dtype=float)
        ys[y]={'trades':int(len(rr)),'mean':float(rr.mean()) if len(rr) else np.nan,'pf':float(pf(rr)) if len(rr) else 0.0}
    return ys


def build_price_index(px_all):
    # v12.simulate() rebuilds this full-market dictionary on every config.
    # v16 evaluates hundreds of configs, so build it once and reuse it.
    # This is a pure runtime optimization: fill dates/prices/costs remain byte-for-byte equivalent in formula.
    return {c:d.sort_values('date').reset_index(drop=True) for c,d in px_all.groupby('code')}


def simulate_fast(name,sig,hold,bycode):
    # Semantics intentionally mirror research_open_tournament_technical_v12.simulate().
    if sig.empty:
        return pd.DataFrame()
    out=[]; last_exit={}
    for s in sig.sort_values(['signal_date','score'],ascending=[True,False]).itertuples(index=False):
        d=bycode.get(s.code)
        if d is None:
            continue
        arr=d.date.to_numpy(); k=int(np.searchsorted(arr,int(s.signal_date),side='right'))
        if k>=len(d) or k+hold>=len(d):
            continue
        e=d.iloc[k]; z=d.iloc[k+hold]
        if int(e.date)<=last_exit.get(s.code,0):
            continue
        if not np.isfinite(e.open) or e.open<=0 or not np.isfinite(z.open) or z.open<=0:
            continue
        ep=float(e.open)*(1+v12.BUY_SLIP); xp=float(z.open)*(1-v12.SELL_SLIP)
        ret=(xp*(1-v12.FEE-v12.TAX))/(ep*(1+v12.FEE))-1
        path=d.iloc[k:k+hold+1]
        mae=float(path.low.min()/ep-1) if path.low.notna().any() else np.nan
        mfe=float(path.high.max()/ep-1) if path.high.notna().any() else np.nan
        out.append({'config':name,'code':s.code,'signal_date':int(s.signal_date),'entry_date':int(e.date),'exit_date':int(z.date),'entry_price':ep,'exit_price':xp,'return_net':ret,'mae':mae,'mfe':mfe,'score':float(s.score)})
        last_exit[s.code]=int(z.date)
    return pd.DataFrame(out)


def main():
    px_all,daily=v12.base.build_daily()
    bycode=build_price_index(px_all)
    d=daily[(daily.date>=20210101)&(daily.date<=20241231)].copy(); d['signal_date']=d.date.astype(int)
    rows=[]
    for fam,(mask,score) in v14.families(d).items():
        raw=d[mask.fillna(False)].copy()
        raw['score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        for hold in HOLDS:
            for slots in SLOTS:
                cfg=f'{fam}__h{hold}__s{slots}'
                sig=raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots)
                tr=simulate_fast(cfg,sig,hold,bycode)
                rr=tr.return_net.astype(float) if len(tr) else pd.Series(dtype=float)
                mean=float(rr.mean()) if len(rr) else 0.0
                med=float(rr.median()) if len(rr) else 0.0
                p=float(pf(rr)) if len(rr) else 0.0
                wr=float((rr>0).mean()) if len(rr) else 0.0
                ys=yearly_stats(tr) if len(tr) else {y:{'trades':0,'mean':np.nan,'pf':0.0} for y in (2021,2022,2023,2024)}
                positive_years=sum(1 for y in ys.values() if np.isfinite(y['mean']) and y['mean']>0)
                # Capacity-optimistic ceiling: assumes a slot can be continuously reinvested at the observed
                # after-friction mean trade return. It still ignores common-cash conflicts and integer shares,
                # so failing this ceiling is a strong reason not to spend a full portfolio audit.
                annualized_ceiling=(1+mean)**(252/max(hold,1))-1 if mean>-1 else -1.0
                passed=bool(len(rr)>=40 and mean>0 and p>1.0 and positive_years>=3 and annualized_ceiling>R10_CAGR)
                reasons=[]
                if len(rr)<40: reasons.append('too_few_after_friction_trades')
                if mean<=0: reasons.append('nonpositive_after_friction_edge')
                if p<=1.0: reasons.append('after_friction_pf_not_above_one')
                if positive_years<3: reasons.append('weak_after_friction_multiyear')
                if annualized_ceiling<=R10_CAGR: reasons.append('capacity_ceiling_not_above_r10')
                rows.append({'config':cfg,'family':fam,'hold_days':hold,'slots':slots,'pass_prescreen':passed,
                             'reject_reason':';'.join(reasons),'trades':int(len(rr)),'win_rate':wr,'mean_net_trade':mean,
                             'median_net_trade':med,'pf':p,'positive_years':positive_years,'annualized_capacity_ceiling':annualized_ceiling,
                             **{f'mean_{y}':ys[y]['mean'] for y in ys},**{f'pf_{y}':ys[y]['pf'] for y in ys}})
    out=pd.DataFrame(rows).sort_values(['pass_prescreen','annualized_capacity_ceiling','pf','mean_net_trade'],ascending=[False,False,False,False])
    out.to_csv(OUT/'all_configs.csv',index=False)
    surv=out[out.pass_prescreen].copy(); surv.to_csv(OUT/'survivors.csv',index=False)
    audit={'version':'v16-friction-aware-prescreen','period':'2021-2024','2025_used':False,'config_count':int(len(out)),
           'survivor_count':int(len(surv)),'r10_cagr_reference':R10_CAGR,
           'execution_in_event_screen':'T+1; +0.5% buy slippage; -0.5% sell slippage; buy/sell fees and sell tax already included by v12 simulator formulas',
           'runtime_optimization':'full-market by-code price index built once and reused; event semantics unchanged from v12.simulate',
           'purpose':'Reject high-turnover candidates whose edge disappears after realistic friction before expensive common-cash audit.',
           'warning':'Still an event/capacity upper screen, not a portfolio proof. Survivors require full common-cash audit.'}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(surv.head(50).to_string(index=False)); print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__': main()
