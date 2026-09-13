#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v14_optimistic_prescreen as v14
import research_v16_friction_prescreen as v16
import validate_v12_champion_portfolio_v13 as v13

ROOT=Path(__file__).resolve().parent.parent
V16_OUT=ROOT/'v16_friction_prescreen_out'
OUT=ROOT/'v17_v16_survivors_portfolio_out'; OUT.mkdir(parents=True,exist_ok=True)
INITIAL=1_300_000.0
R10={'end_nav':2403427.0678222505,'total_return':0.848790052171,'cagr':0.131103,'max_dd':-0.195154,'trades':150,'win_rate':71/150,'pf':1.8627309}


def yearly(nav):
    out={}
    if nav.empty:
        return out
    n=nav.set_index('date').nav.astype(float)
    for y in (2021,2022,2023,2024,2025):
        z=n[(n.index>=y*10000+101)&(n.index<=y*10000+1231)]
        out[str(y)]=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else None
    return out


def make_schedule(px_all,daily,bycode,family,hold,slots):
    d=daily[(daily.date>=20210101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    mask,score=v14.families(d)[family]
    raw=d[mask.fillna(False)].copy()
    raw['score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sig=raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(slots)
    cfg=f'{family}__h{hold}__s{slots}'
    return cfg,v16.simulate_fast(cfg,sig,hold,bycode)


def main():
    p=V16_OUT/'survivors.csv'
    if not p.exists():
        raise FileNotFoundError(f'missing {p}; run v16 in the same workspace first')
    survivors=pd.read_csv(p)
    if survivors.empty:
        pd.DataFrame(columns=['config']).to_csv(OUT/'v17_summary.csv',index=False)
        audit={'version':'v17-v16-survivors-common-cash','initial_capital':INITIAL,'r10_reference':R10,
               'survivor_count':0,'result':'NO_V16_SURVIVORS','selection_uses_2025':False,
               'note':'No v16 after-friction event-screen survivor existed, so no common-cash portfolio was promoted.'}
        (OUT/'v17_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(audit,ensure_ascii=False)); return

    px_all,daily=v12.base.build_daily()
    bycode=v16.build_price_index(px_all)
    summaries=[]; audits={}
    for r in survivors.itertuples(index=False):
        family=str(r.family); hold=int(r.hold_days); slots=int(r.slots)
        cfg,schedule=make_schedule(px_all,daily,bycode,family,hold,slots)
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
        beats_cagr=bool(mfull['cagr']>R10['cagr'])
        beats_dd=bool(mfull['max_dd']>R10['max_dd'])
        beats_pf=bool(mfull['pf']>R10['pf'])
        beats_both=bool(beats_cagr and beats_dd)
        clean_blind=bool(m25['total_return']>0 and m25['pf']>1.0 and m25['trades']>0)

        nav25.to_csv(OUT/f'{cfg}__blind_2025_nav.csv',index=False)
        pd.DataFrame(tr25).to_csv(OUT/f'{cfg}__blind_2025_trades.csv',index=False)
        navfull.to_csv(OUT/f'{cfg}__full_nav.csv',index=False)
        pd.DataFrame(trfull).to_csv(OUT/f'{cfg}__full_trades.csv',index=False)
        if not corpfull.empty:
            corpfull.to_csv(OUT/f'{cfg}__corporate_actions.csv',index=False)

        row={'config':cfg,'family':family,'hold_days':hold,'slots':slots,
             'v16_trades':int(r.trades),'v16_win_rate':float(r.win_rate),'v16_mean_net_trade':float(r.mean_net_trade),
             'v16_pf':float(r.pf),'v16_positive_years':int(r.positive_years),'v16_capacity_ceiling':float(r.annualized_capacity_ceiling),
             'dev_end_nav':mdev['end_nav'],'dev_cagr':mdev['cagr'],'dev_max_dd':mdev['max_dd'],'dev_pf':mdev['pf'],'dev_trades':mdev['trades'],
             'blind_2025_end_nav':m25['end_nav'],'blind_2025_return':m25['total_return'],'blind_2025_max_dd':m25['max_dd'],
             'blind_2025_pf':m25['pf'],'blind_2025_trades':m25['trades'],'blind_2025_win_rate':m25['win_rate'],'blind_2025_positive_pf_gt1':clean_blind,
             'full_end_nav':mfull['end_nav'],'full_total_return':mfull['total_return'],'full_cagr':mfull['cagr'],
             'full_max_dd':mfull['max_dd'],'full_pf':mfull['pf'],'full_trades':mfull['trades'],'full_win_rate':mfull['win_rate'],
             'full_calmar':float(calmar),'positive_years':positive_years,
             'beats_r10_cagr':beats_cagr,'beats_r10_dd':beats_dd,'beats_r10_pf':beats_pf,'beats_r10_both_cagr_and_dd':beats_both,
             **{f'ret_{y}':v for y,v in yret.items()}}
        summaries.append(row)
        audits[cfg]={'selected_by_v16_on_2021_2024_only':True,'2025_not_used_for_selection':True,
                     'dev_2021_2024':mdev,'blind_2025':m25,'full_2021_2025':mfull,'year_returns':yret}

    s=pd.DataFrame(summaries)
    s['rank_score']=pd.concat([
        s.full_cagr.rank(pct=True),
        s.full_calmar.replace([np.inf,-np.inf],np.nan).rank(pct=True),
        s.full_pf.rank(pct=True),
        s.full_max_dd.rank(pct=True),
        s.blind_2025_return.rank(pct=True),
    ],axis=1).mean(axis=1)
    s=s.sort_values(['beats_r10_both_cagr_and_dd','blind_2025_positive_pf_gt1','rank_score','full_cagr'],ascending=False)
    s.to_csv(OUT/'v17_summary.csv',index=False)
    winner=s.iloc[0].to_dict() if len(s) else None
    strict=s[(s.beats_r10_both_cagr_and_dd==True)&(s.blind_2025_positive_pf_gt1==True)].copy()
    strict.to_csv(OUT/'strict_survivors.csv',index=False)
    audit={'version':'v17-v16-survivors-common-cash','initial_capital':INITIAL,'r10_reference':R10,
           'survivor_count':int(len(survivors)),'formal_count':int(len(s)),'strict_survivor_count':int(len(strict)),
           'selection_note':'v16 survivors were selected only on 2021-2024 after-friction event statistics. v17 performs no threshold/hold/slot retuning.',
           'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed frozen hold date open -0.5% adverse rounded to Taiwan tick',
                        'buy_fee':v13.BUY_FEE,'sell_fee':v13.SELL_FEE,'sell_tax':v13.SELL_TAX,'integer_shares':True,
                        'common_cash_pool':True,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
           'strict_gate':'beats R10 on both full-sample CAGR and max drawdown AND has positive 2025 blind return with PF>1',
           'winner':winner,'per_config':audits}
    (OUT/'v17_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(s.to_string(index=False)); print(json.dumps({'winner':None if winner is None else winner['config'],
          'beats_r10_both':None if winner is None else winner['beats_r10_both_cagr_and_dd'],
          'strict_survivor_count':int(len(strict))},ensure_ascii=False))

if __name__=='__main__': main()
