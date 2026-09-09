#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'calm_phase3_output'; OUT.mkdir(exist_ok=True)

def load_run(path,label):
    nav=pd.read_csv(path/'r10max_formal_nav.csv'); trades=pd.read_csv(path/'r10max_formal_trades.csv'); orders=pd.read_csv(path/'r10max_formal_orders.csv'); slots=pd.read_csv(path/'r10max_formal_slot_diag.csv')
    nav['year']=nav.date.astype(str).str[:4].astype(int)
    if len(trades): trades['year']=trades.exit_date.astype(str).str[:4].astype(int)
    if len(orders): orders['year']=orders.signal_date.astype(str).str[:4].astype(int)
    out=[]
    for y,g in nav.groupby('year'):
        t=trades[trades.year==y] if len(trades) else trades
        o=orders[orders.year==y] if len(orders) else orders
        buys=o[o.side=='BUY'] if len(o) else o
        filled_buys=buys[buys.status=='FILLED'] if len(buys) else buys
        sells=o[o.side=='SELL'] if len(o) else o
        reasons=t.reason.value_counts(normalize=True).to_dict() if len(t) else {}
        bystr={}
        if len(t):
            for s,z in t.groupby('strategy'):
                bystr[s]={'trades':int(len(z)),'win_rate':float((z.pnl>0).mean()),'avg_return':float(z['return'].mean()),'median_return':float(z['return'].median()),'avg_hold':float(z.hold_days.mean()),'pf':float(z.loc[z.pnl>0,'pnl'].sum()/-z.loc[z.pnl<0,'pnl'].sum()) if (z.pnl<0).any() else None}
        out.append({'period':label,'year':int(y),'avg_exposure':float(g.exposure.mean()),'median_exposure':float(g.exposure.median()),'max_drawdown':float(g.drawdown.min()),'avg_positions':float(g.positions.mean()),'no_buy_pct':float(g.no_buy_active.mean()),'completed_trades':int(len(t)),'win_rate':float((t.pnl>0).mean()) if len(t) else None,'avg_trade_return':float(t['return'].mean()) if len(t) else None,'median_trade_return':float(t['return'].median()) if len(t) else None,'avg_hold':float(t.hold_days.mean()) if len(t) else None,'buy_orders':int(len(buys)),'buy_fill_rate':float(len(filled_buys)/len(buys)) if len(buys) else None,'sell_orders':int(len(sells)),'forced_dd_share':float((t.reason=='FORCE_DD').mean()) if len(t) else 0.0,'reasons':json.dumps(reasons,ensure_ascii=False),'strategy_detail':json.dumps(bystr,ensure_ascii=False)})
    return pd.DataFrame(out)

old=load_run(ROOT/'phase3_old','2016-2020')
new=load_run(ROOT/'phase3_new','2021-2025')
annual=pd.concat([old,new],ignore_index=True).sort_values('year'); annual.to_csv(OUT/'annual_r10_trade_forensics.csv',index=False)
# Compare key contrast years and broad eras; no threshold selection.
def row(y): return annual[annual.year==y].iloc[0].to_dict()
summary={'design':'diagnostic_only_no_detector_threshold_selection','key_years':{str(y):row(y) for y in [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025] if (annual.year==y).any()},'contrast_2018_vs_2022':{},'era_means':{}}
for col in ['avg_exposure','max_drawdown','avg_positions','no_buy_pct','completed_trades','win_rate','avg_trade_return','avg_hold','buy_fill_rate','forced_dd_share']:
    a=float(row(2018)[col]) if row(2018)[col] is not None else None; b=float(row(2022)[col]) if row(2022)[col] is not None else None
    summary['contrast_2018_vs_2022'][col]={'2018':a,'2022':b,'difference_2018_minus_2022':(a-b if a is not None and b is not None else None)}
for name,years in {'2016_2018':[2016,2017,2018],'2019_2020':[2019,2020],'2021_2025':[2021,2022,2023,2024,2025]}.items():
    z=annual[annual.year.isin(years)]
    summary['era_means'][name]={c:float(z[c].dropna().mean()) for c in ['avg_exposure','max_drawdown','avg_positions','no_buy_pct','completed_trades','win_rate','avg_trade_return','avg_hold','buy_fill_rate','forced_dd_share']}
(OUT/'phase3_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('PHASE3_COMPLETE')
print(annual.to_string(index=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
