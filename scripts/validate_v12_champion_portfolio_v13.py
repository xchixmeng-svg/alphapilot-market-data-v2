#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import research_open_tournament_technical_v12 as v12

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'v13_champion_portfolio_out'; OUT.mkdir(parents=True,exist_ok=True)
INITIAL=1_300_000.0
BUY_FEE=0.000855; SELL_FEE=0.000855; SELL_TAX=0.003
BUY_SLIP=0.005; SELL_SLIP=0.005
SLOT_CANDIDATES=(1,2,3,5,8,10,15,20)

def tick(p):
    return 0.01 if p<10 else 0.05 if p<50 else 0.1 if p<100 else 0.5 if p<500 else 1.0 if p<1000 else 5.0

def ceil_tick(p):
    t=tick(p); return round(np.ceil((p-1e-10)/t)*t,4)

def floor_tick(p):
    t=tick(p); return round(np.floor((p+1e-10)/t)*t,4)

def pf(r):
    r=pd.Series(r,dtype=float); g=float(r[r>0].sum()); l=float(-r[r<0].sum())
    return (99.0 if g>0 else 0.0) if l<=1e-12 else g/l

def metrics(nav,trades,years):
    if nav.empty: return {'end_nav':INITIAL,'total_return':0.0,'cagr':0.0,'max_dd':0.0,'trades':0,'win_rate':0.0,'pf':0.0}
    n=nav.nav.astype(float); dd=n/n.cummax()-1
    total=float(n.iloc[-1]/n.iloc[0]-1)
    cagr=float((n.iloc[-1]/n.iloc[0])**(1/max(years,1e-9))-1)
    rr=pd.Series([x['return'] for x in trades],dtype=float)
    return {'end_nav':float(n.iloc[-1]),'total_return':total,'cagr':cagr,'max_dd':float(dd.min()),'trades':int(len(rr)),
            'win_rate':float((rr>0).mean()) if len(rr) else 0.0,'pf':float(pf(rr)) if len(rr) else 0.0}

def build_schedule(px_all,daily):
    d=daily[(daily.date>=20210101)&(daily.date<=20251231)].copy(); d['signal_date']=d.date.astype(int)
    mask=(d.r20_pr>=.65)&(d.r60_pr>=.60)&(d.vol20_pr<=.45)&(d.aclose>d.ma60)
    z=d[mask.fillna(False)].copy()
    z['score']=.35*z.r20_pr+.25*z.r60_pr+.25*(1-z.vol20_pr)+.15*z.amount20_pr
    sig=z.sort_values(['signal_date','score','amount20'],ascending=[True,False,False]).groupby('signal_date',as_index=False).head(1)
    return v12.simulate('low_vol_momentum__h40__n1',sig,40,px_all)

def simulate_portfolio(px_all,schedule,slots,start,end,initial=INITIAL):
    px=px_all[(px_all.date>=start)&(px_all.date<=end)].copy().sort_values(['date','code'])
    dates=sorted(px.date.unique().tolist()); by_date={d:s.set_index('code') for d,s in px.groupby('date')}
    ent=schedule[(schedule.signal_date>=start)&(schedule.signal_date<=end)].copy()
    entries={int(d):g.sort_values('score',ascending=False) for d,g in ent.groupby('entry_date')}
    exits={}
    for r in ent.itertuples(index=False): exits.setdefault(int(r.exit_date),[]).append(r)
    cash=float(initial); pos={}; nav_rows=[]; trades=[]; corp=[]; prev_nav=float(initial)
    for di in dates:
        sub=by_date[di]
        # Corporate actions apply to positions held before the event-day open.
        for code,p in list(pos.items()):
            if code not in sub.index: continue
            r=sub.loc[code]
            if not bool(r.get('is_official_event',False)): continue
            old=int(p['shares']); factor=float(r.get('share_factor',1.0)) if np.isfinite(r.get('share_factor',np.nan)) else 1.0
            exact=old*factor; new=max(0,int(np.floor(exact+1e-10)))
            div=old*float(r.get('cash_dividend_per_share',0.0) or 0.0)
            ref=float(r.get('reference_price',r['close'])) if np.isfinite(r.get('reference_price',np.nan)) else float(r['close'])
            cil=max(0.0,exact-new)*ref; credit=div+cil
            cash+=credit; p['shares']=new; p['corp_income']+=credit
            corp.append({'date':di,'code':code,'old_shares':old,'new_shares':new,'factor':factor,'cash_credit':credit})
        # Scheduled exits at T+1/open-style fixed hold date.
        for e in exits.get(di,[]):
            code=str(e.code).zfill(4); p=pos.get(code)
            if p is None or code not in sub.index: continue
            r=sub.loc[code]; fill=floor_tick(float(r['open'])*(1-SELL_SLIP))
            gross=fill*p['shares']; fee=gross*SELL_FEE; tax=gross*SELL_TAX; proceeds=gross-fee-tax
            cash+=proceeds; pnl=proceeds+p['corp_income']-p['cost_total']; ret=pnl/p['cost_total'] if p['cost_total'] else 0.0
            trades.append({'code':code,'entry_date':p['entry_date'],'exit_date':di,'entry_price':p['entry_price'],'exit_price':fill,
                           'entry_shares':p['entry_shares'],'exit_shares':p['shares'],'cost_total':p['cost_total'],
                           'corp_income':p['corp_income'],'proceeds':proceeds,'pnl':pnl,'return':ret})
            del pos[code]
        # Entries use previous-close NAV target; no leverage and integer shares only.
        free=max(0,slots-len(pos)); target=prev_nav/slots if slots>0 else 0.0
        if free>0 and di in entries:
            for e in entries[di].itertuples(index=False):
                if free<=0: break
                code=str(e.code).zfill(4)
                if code in pos or code not in sub.index: continue
                r=sub.loc[code]; fill=ceil_tick(float(r['open'])*(1+BUY_SLIP))
                budget=min(target,cash); sh=int(np.floor(budget/(fill*(1+BUY_FEE))))
                if sh<=0: continue
                gross=fill*sh; fee=gross*BUY_FEE; cost=gross+fee
                if cost>cash+1e-7: continue
                cash-=cost; pos[code]={'shares':sh,'entry_shares':sh,'entry_date':di,'entry_price':fill,'cost_total':cost,'corp_income':0.0}
                free-=1
        mv=0.0
        for code,p in pos.items():
            if code in sub.index: p['last_close']=float(sub.loc[code]['close'])
            mv+=p['shares']*float(p.get('last_close',p['entry_price']))
        nav=cash+mv; nav_rows.append({'date':di,'cash':cash,'market_value':mv,'positions':len(pos),'nav':nav}); prev_nav=nav
    # Liquidate remaining positions at final available close with sell costs for conservative terminal NAV.
    if dates:
        last=dates[-1]; sub=by_date[last]
        for code,p in list(pos.items()):
            if code not in sub.index: continue
            fill=floor_tick(float(sub.loc[code]['close'])*(1-SELL_SLIP)); gross=fill*p['shares']; fee=gross*SELL_FEE; tax=gross*SELL_TAX
            proceeds=gross-fee-tax; cash+=proceeds; pnl=proceeds+p['corp_income']-p['cost_total']; ret=pnl/p['cost_total'] if p['cost_total'] else 0.0
            trades.append({'code':code,'entry_date':p['entry_date'],'exit_date':last,'entry_price':p['entry_price'],'exit_price':fill,
                           'entry_shares':p['entry_shares'],'exit_shares':p['shares'],'cost_total':p['cost_total'],
                           'corp_income':p['corp_income'],'proceeds':proceeds,'pnl':pnl,'return':ret,'forced_eoy':True})
            del pos[code]
        if nav_rows: nav_rows[-1].update({'cash':cash,'market_value':0.0,'positions':0,'nav':cash})
    return pd.DataFrame(nav_rows),trades,pd.DataFrame(corp)

def rank_dev(rows):
    z=pd.DataFrame(rows).copy()
    z['calmar']=z.cagr/(-z.max_dd).replace(0,np.nan)
    parts=[z.cagr.rank(pct=True),z.calmar.rank(pct=True),z.pf.rank(pct=True),z.max_dd.rank(pct=True),z.win_rate.rank(pct=True)]
    z['robust_score']=pd.concat(parts,axis=1).mean(axis=1)
    return z.sort_values(['robust_score','cagr','pf'],ascending=False)

def main():
    px_all,daily=v12.base.build_daily(); schedule=build_schedule(px_all,daily)
    schedule.to_csv(OUT/'champion_signal_schedule.csv',index=False)
    dev_rows=[]
    for slots in SLOT_CANDIDATES:
        nav,tr,corp=simulate_portfolio(px_all,schedule,slots,20210101,20241231)
        m=metrics(nav,tr,4.0); dev_rows.append({'slots':slots,**m})
    dev=rank_dev(dev_rows); dev.to_csv(OUT/'dev_slot_selection.csv',index=False)
    best_slots=int(dev.iloc[0].slots)
    nav25,tr25,corp25=simulate_portfolio(px_all,schedule,best_slots,20250101,20251231)
    m25=metrics(nav25,tr25,1.0); nav25.to_csv(OUT/'blind_2025_nav.csv',index=False); pd.DataFrame(tr25).to_csv(OUT/'blind_2025_trades.csv',index=False)
    navfull,trfull,corpfull=simulate_portfolio(px_all,schedule,best_slots,20210101,20251231)
    mfull=metrics(navfull,trfull,5.0); navfull.to_csv(OUT/'full_2021_2025_nav.csv',index=False); pd.DataFrame(trfull).to_csv(OUT/'full_2021_2025_trades.csv',index=False)
    audit={'version':'v13-common-cash-validation','signal_config':'low_vol_momentum__h40__n1','signal_selected_without_2025':True,
           'slot_candidates':list(SLOT_CANDIDATES),'slot_selection_uses':'2021-2024 only','selected_slots':best_slots,
           'initial_capital':INITIAL,'execution':{'decision':'T close','entry':'T+1 open +0.5% adverse rounded to Taiwan tick','exit':'fixed 40 trading-day schedule open -0.5% adverse rounded to tick','buy_fee':BUY_FEE,'sell_fee':SELL_FEE,'sell_tax':SELL_TAX,'integer_shares':True,'common_cash_pool':True,'corporate_actions':'official effective-date share factor + cash dividend + cash-in-lieu'},
           'blind_2025':m25,'full_2021_2025':mfull,'dev_slot_table':dev.to_dict(orient='records')}
    (OUT/'portfolio_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    print(dev.to_string(index=False)); print('\nBLIND 2025',json.dumps(m25,ensure_ascii=False)); print('\nFULL',json.dumps(mfull,ensure_ascii=False)); print('selected_slots',best_slots)

if __name__=='__main__': main()
