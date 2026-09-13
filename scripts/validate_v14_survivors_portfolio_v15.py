#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v14_optimistic_prescreen as v14
import validate_v12_champion_portfolio_v13 as v13

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'v15_finalists_portfolio_out'; OUT.mkdir(parents=True,exist_ok=True)
INITIAL=1_300_000.0
R10={'end_nav':2403427.0678222505,'total_return':0.848790052171,'cagr':0.131103,'max_dd':-0.195154,'trades':150,'win_rate':71/150,'pf':1.8627309}
FINALISTS=[
    ('stable_compounder_proxy',40,3),
    ('breadth_independent_leader',5,2),
    ('breadth_independent_leader',5,3),
    ('sponsor_accumulation',10,2),
]


def make_schedule(px_all,daily,family,hold,slots):
    d=daily[(daily.date>=20210101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    mask,score=v14.families(d)[family]
    raw=d[mask.fillna(False)].copy()
    raw['score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sig=raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots)
    cfg=f'{family}__h{hold}__s{slots}'
    # v12 simulator is used only to freeze causal T+1 entry/exit dates and suppress overlapping same-stock events.
    # v15 recomputes all actual fills/cash/fees/tax with the common-cash portfolio engine below.
    return cfg, v12.simulate(cfg,sig,hold,px_all)


def yearly(nav):
    out={}
    if nav.empty: return out
    n=nav.set_index('date').nav.astype(float)
    for y in (2021,2022,2023,2024,2025):
        z=n[(n.index>=y*10000+101)&(n.index<=y*10000+1231)]
        out[str(y)]=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else None
    return out


def main():
    px_all,daily=v12.base.build_daily()
    summaries=[]; audits={}
    for family,hold,slots in FINALISTS:
        cfg,schedule=make_schedule(px_all,daily,family,hold,slots)
        schedule.to_csv(OUT/f'{cfg}__signal_schedule.csv',index=False)

        nav_dev,tr_dev,corp_dev=v13.simulate_portfolio(px_all,schedule,slots,20210101,20241231,INITIAL)
        mdev=v13.metrics(nav_dev,tr_dev,4.0)

        nav25,tr25,corp25=v13.simulate_portfolio(px_all,schedule,slots,20250101,20251231,INITIAL)
        m25=v13.metrics(nav25,tr25,1.0)

        navfull,trfull,corpfull=v13.simulate_portfolio(px_all,schedule,slots,20210101,20251231,INITIAL)
        mfull=v13.metrics(navfull,trfull,5.0)
        yret=yearly(navfull)
        positive_years=sum(1 for v in yret.values() if v is not None and v>0)
        calmar=mfull['cagr']/(-mfull['max_dd']) if mfull['max_dd']<0 else np.inf
        beats_return=bool(mfull['cagr']>R10['cagr'])
        beats_dd=bool(mfull['max_dd']>R10['max_dd'])
        beats_both=bool(beats_return and beats_dd)

        nav25.to_csv(OUT/f'{cfg}__blind_2025_nav.csv',index=False)
        pd.DataFrame(tr25).to_csv(OUT/f'{cfg}__blind_2025_trades.csv',index=False)
        navfull.to_csv(OUT/f'{cfg}__full_nav.csv',index=False)
        pd.DataFrame(trfull).to_csv(OUT/f'{cfg}__full_trades.csv',index=False)
        if not corpfull.empty: corpfull.to_csv(OUT/f'{cfg}__corporate_actions.csv',index=False)

        row={'config':cfg,'family':family,'hold_days':hold,'slots':slots,
             'dev_end_nav':mdev['end_nav'],'dev_cagr':mdev['cagr'],'dev_max_dd':mdev['max_dd'],'dev_pf':mdev['pf'],'dev_trades':mdev['trades'],
             'blind_2025_end_nav':m25['end_nav'],'blind_2025_return':m25['total_return'],'blind_2025_max_dd':m25['max_dd'],'blind_2025_pf':m25['pf'],'blind_2025_trades':m25['trades'],'blind_2025_win_rate':m25['win_rate'],
             'full_end_nav':mfull['end_nav'],'full_total_return':mfull['total_return'],'full_cagr':mfull['cagr'],'full_max_dd':mfull['max_dd'],'full_pf':mfull['pf'],'full_trades':mfull['trades'],'full_win_rate':mfull['win_rate'],
             'full_calmar':float(calmar),'positive_years':positive_years,'beats_r10_cagr':beats_return,'beats_r10_dd':beats_dd,'beats_r10_both_cagr_and_dd':beats_both,
             **{f'ret_{y}':v for y,v in yret.items()}}
        summaries.append(row)
        audits[cfg]={'signal_selected_from_v14_2021_2024_only':True,'config_frozen_before_2025':True,'dev_2021_2024':mdev,'blind_2025':m25,'full_2021_2025':mfull,'year_returns':yret}

    s=pd.DataFrame(summaries)
    # No hidden parameter tuning: rank the four frozen finalists after all realistic results are known.
    s['rank_score']=pd.concat([
        s.full_cagr.rank(pct=True),
        s.full_calmar.replace([np.inf,-np.inf],np.nan).rank(pct=True),
        s.full_pf.rank(pct=True),
        s.full_max_dd.rank(pct=True),
        s.blind_2025_return.rank(pct=True),
    ],axis=1).mean(axis=1)
    s=s.sort_values(['beats_r10_both_cagr_and_dd','rank_score','full_cagr'],ascending=False)
    s.to_csv(OUT/'v15_summary.csv',index=False)
    winner=s.iloc[0].to_dict() if len(s) else None
    audit={'version':'v15-v14-finalists-common-cash','initial_capital':INITIAL,'r10_reference':R10,
           'finalists':[f'{a}__h{b}__s{c}' for a,b,c in FINALISTS],
           'selection_note':'All four configs were frozen from v14 2021-2024 optimistic prescreen before this realistic audit. No hold/slot/threshold tuning is performed in v15.',
           'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed frozen hold date open -0.5% adverse rounded to Taiwan tick','buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'integer_shares':True,'common_cash_pool':True,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
           'winner':winner,'per_config':audits}
    (OUT/'v15_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(s.to_string(index=False))
    print(json.dumps({'winner':None if winner is None else winner['config'],'beats_r10_both':None if winner is None else winner['beats_r10_both_cagr_and_dd']},ensure_ascii=False))

if __name__=='__main__': main()
