#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12
import research_v14_optimistic_prescreen as v14
import research_v16_friction_prescreen as v16
import validate_v12_champion_portfolio_v13 as v13

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'v18_breadth_challenger_audit_out'; OUT.mkdir(parents=True,exist_ok=True)
INITIAL=1_300_000.0
FAMILY='breadth_independent_leader'; HOLD=40; SLOTS=3
CONFIG=f'{FAMILY}__h{HOLD}__s{SLOTS}'


def make_schedule(px_all,daily,bycode):
    d=daily[(daily.date>=20210101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    mask,score=v14.families(d)[FAMILY]
    raw=d[mask.fillna(False)].copy()
    raw['score']=score.loc[raw.index].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    sig=raw.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(SLOTS)
    return v16.simulate_fast(CONFIG,sig,HOLD,bycode)


def nav_metrics(nav,trades):
    n=nav.sort_values('date').nav.astype(float).reset_index(drop=True)
    r=n.pct_change().dropna()
    dd=n/n.cummax()-1
    years=(len(n)-1)/252 if len(n)>1 else 0
    cagr=float((n.iloc[-1]/n.iloc[0])**(1/max(years,1e-9))-1) if len(n)>1 else 0.0
    vol=float(r.std(ddof=1)*math.sqrt(252)) if len(r)>1 else 0.0
    sharpe=float(r.mean()/r.std(ddof=1)*math.sqrt(252)) if len(r)>1 and r.std(ddof=1)>1e-12 else 0.0
    rr=pd.Series([x['return'] if isinstance(x,dict) else np.nan for x in trades],dtype=float).dropna()
    pf=v13.pf(rr) if len(rr) else 0.0
    return {'end_nav':float(n.iloc[-1]),'total_return':float(n.iloc[-1]/n.iloc[0]-1),'cagr':cagr,
            'max_dd':float(dd.min()),'calmar':float(cagr/(-dd.min())) if dd.min()<0 else None,
            'daily_ann_vol':vol,'daily_sharpe0':sharpe,'trades':int(len(rr)),
            'win_rate':float((rr>0).mean()) if len(rr) else 0.0,'pf':float(pf)}


def df_trade_metrics(df):
    if df.empty: return {'trades':0,'win_rate':0.0,'pf':0.0}
    z=df.copy()
    if 'return' not in z:
        z['return']=z['pnl']/z['cost_total']
    rr=z['return'].astype(float)
    out={'trades':int(len(z)),'win_rate':float((rr>0).mean()),'pf':float(v13.pf(rr))}
    pos=z[z.pnl>0].sort_values('pnl',ascending=False)
    gross_profit=float(pos.pnl.sum())
    for k in (1,3,5,10):
        out[f'top{k}_profit_share']=float(pos.head(k).pnl.sum()/gross_profit) if gross_profit>0 else 0.0
    bycode=z.groupby('code',as_index=False).pnl.sum()
    abs_sum=float(bycode.pnl.abs().sum())
    out['code_pnl_abs_hhi']=float(((bycode.pnl.abs()/abs_sum)**2).sum()) if abs_sum>0 else 0.0
    return out


def exact_year_returns(nav):
    z=nav.sort_values('date').copy(); z['date']=z.date.astype(int)
    out={}; prior=float(z.iloc[0].nav)
    for y in range(2021,2026):
        yy=z[(z.date>=y*10000+101)&(z.date<=y*10000+1231)]
        if len(yy):
            out[str(y)]=float(yy.iloc[-1].nav/prior-1); prior=float(yy.iloc[-1].nav)
    return out


def segment(nav,start_date,end_date):
    z=nav.sort_values('date').copy(); z['date']=z.date.astype(int)
    pre=z[z.date<=start_date]
    end=z[z.date<=end_date]
    if pre.empty or end.empty: return None
    start=pre.iloc[-1]; fin=end.iloc[-1]
    w=z[(z.date>=int(start.date))&(z.date<=int(fin.date))].copy()
    n=w.nav.astype(float); dd=n/n.cummax()-1
    return {'start_date':int(start.date),'end_date':int(fin.date),'start_nav':float(start.nav),'end_nav':float(fin.nav),
            'return':float(fin.nav/start.nav-1),'max_dd':float(dd.min())}


def main():
    # Locked R10 artifacts are generated immediately before this script by the workflow.
    r10_summary=json.load(open(ROOT/'formal_run/r10max_formal_summary.json'))
    r10_nav=pd.read_csv(ROOT/'formal_run/r10max_formal_nav.csv')
    r10_tr=pd.read_csv(ROOT/'formal_run/r10max_formal_trades.csv')
    if 'return' not in r10_tr.columns:
        if {'pnl','entry_cost'}.issubset(r10_tr.columns): r10_tr['return']=r10_tr.pnl/r10_tr.entry_cost
        elif {'pnl','cost_total'}.issubset(r10_tr.columns): r10_tr['return']=r10_tr.pnl/r10_tr.cost_total
        else: r10_tr['return']=np.nan

    px_all,daily=v12.base.build_daily(); bycode=v16.build_price_index(px_all); schedule=make_schedule(px_all,daily,bycode)
    schedule.to_csv(OUT/'challenger_signal_schedule.csv',index=False)
    nav,tr,corp=v13.simulate_portfolio(px_all,schedule,SLOTS,20210101,20251231,INITIAL)
    ch=v13.metrics(nav,tr,5.0); ch['calmar']=ch['cagr']/(-ch['max_dd']) if ch['max_dd']<0 else None
    # Daily risk metrics, using exact NAV stream.
    ch_daily=nav_metrics(nav,tr)
    r10n=r10_nav.sort_values('date').nav.astype(float); r10r=r10n.pct_change().dropna(); r10dd=r10n/r10n.cummax()-1
    r10_daily={'calmar':float(r10_summary['strategy']['cagr']/(-r10_summary['strategy']['max_drawdown'])),
               'daily_ann_vol':float(r10r.std(ddof=1)*math.sqrt(252)),
               'daily_sharpe0':float(r10r.mean()/r10r.std(ddof=1)*math.sqrt(252)) if r10r.std(ddof=1)>1e-12 else 0.0}

    # Fresh-start 2025 and true continuation 2025 are intentionally both measured: a large gap means slot/path sensitivity.
    nav25,tr25,_=v13.simulate_portfolio(px_all,schedule,SLOTS,20250101,20251231,INITIAL)
    fresh25=v13.metrics(nav25,tr25,1.0)
    cont25=segment(nav,20241231,20251231)
    r10cont25=segment(r10_nav,20241231,20251231)
    fresh_set={(str(x['code']).zfill(4),int(x['entry_date'])) for x in tr25}
    full25_set={(str(x['code']).zfill(4),int(x['entry_date'])) for x in tr if int(x['entry_date'])>=20250101}
    inter=len(fresh_set&full25_set); union=len(fresh_set|full25_set)
    path={'fresh_2025':fresh25,'continuation_2025':cont25,'r10_continuation_2025':r10cont25,
          'continuation_minus_fresh_return_pp':float((cont25['return']-fresh25['total_return'])*100),
          'fresh_vs_continuation_trade_jaccard':float(inter/union) if union else 1.0,
          'fresh_trade_count':len(fresh_set),'continuation_new_entry_trade_count':len(full25_set)}

    # No retuning: hold/slots/filter remain frozen. Only execution adversity is increased.
    stress=[]; base_buy,base_sell=v13.BUY_SLIP,v13.SELL_SLIP
    try:
        for slip in (0.005,0.0075,0.0100):
            v13.BUY_SLIP=slip; v13.SELL_SLIP=slip
            ns,ts,_=v13.simulate_portfolio(px_all,schedule,SLOTS,20210101,20251231,INITIAL)
            ms=v13.metrics(ns,ts,5.0); ms['slip_each_side']=slip; ms['calmar']=ms['cagr']/(-ms['max_dd']) if ms['max_dd']<0 else None
            stress.append(ms)
    finally:
        v13.BUY_SLIP=base_buy; v13.SELL_SLIP=base_sell

    ch_tr=pd.DataFrame(tr)
    if not ch_tr.empty and 'return' not in ch_tr: ch_tr['return']=ch_tr.pnl/ch_tr.cost_total
    contribution={'challenger':df_trade_metrics(ch_tr),'r10':df_trade_metrics(r10_tr)}
    years={'challenger':exact_year_returns(nav),'r10':{str(x['year']):x['strategy_return'] for x in r10_summary['annual']}}

    r10s=r10_summary['strategy']
    compare={'cagr_delta_pp':float((ch['cagr']-r10s['cagr'])*100),
             'max_dd_penalty_pp':float((abs(ch['max_dd'])-abs(r10s['max_drawdown']))*100),
             'pf_delta':float(ch['pf']-r10s['profit_factor']),
             'calmar_delta':float(ch['calmar']-r10_daily['calmar']),
             '2025_continuation_return_delta_pp':float((cont25['return']-r10cont25['return'])*100),
             'challenger_return_pareto_advantage':bool(ch['cagr']>r10s['cagr'] and ch['pf']>r10s['profit_factor']),
             'challenger_risk_adjusted_calmar_advantage':bool(ch['calmar']>r10_daily['calmar'])}
    audit={'version':'v18-frozen-breadth-challenger-robustness','config':CONFIG,'selection':'frozen from v16 2021-2024 friction-aware survivors; no parameter retuning in v18',
           'r10':{**r10s,**r10_daily},'challenger':{**ch,**{k:v for k,v in ch_daily.items() if k in ('daily_ann_vol','daily_sharpe0')}},
           'comparison':compare,'year_returns':years,'path_sensitivity':path,'slippage_stress':stress,'trade_concentration':contribution,
           'interpretation_flags':{
             'large_2025_path_sensitivity_gap_gt20pp':bool(abs(path['continuation_minus_fresh_return_pp'])>20),
             'stress_1pct_each_side_still_positive_cagr':bool(stress[-1]['cagr']>0),
             'stress_1pct_each_side_still_beats_r10_cagr':bool(stress[-1]['cagr']>r10s['cagr']),
             'top5_winners_over_50pct_gross_profit':bool(contribution['challenger']['top5_profit_share']>0.5),
             'historical_pareto_dominates_r10_on_cagr_pf_dd':bool(ch['cagr']>=r10s['cagr'] and ch['pf']>=r10s['profit_factor'] and ch['max_dd']>=r10s['max_drawdown'])
           }}
    (OUT/'v18_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    nav.to_csv(OUT/'challenger_full_nav.csv',index=False); ch_tr.to_csv(OUT/'challenger_full_trades.csv',index=False)
    nav25.to_csv(OUT/'challenger_fresh_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(OUT/'challenger_fresh_2025_trades.csv',index=False)
    if not corp.empty: corp.to_csv(OUT/'challenger_corporate_actions.csv',index=False)
    pd.DataFrame(stress).to_csv(OUT/'slippage_stress.csv',index=False)
    pd.DataFrame([{'metric':k,'value':v} for k,v in compare.items()]).to_csv(OUT/'comparison.csv',index=False)
    print(json.dumps(audit,ensure_ascii=False,indent=2,default=float))

if __name__=='__main__': main()
