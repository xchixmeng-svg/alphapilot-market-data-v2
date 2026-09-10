from pathlib import Path
import importlib.util, json, math
import numpy as np, pandas as pd

# Reuse Stage14 data/features/entry logic exactly. Importing also regenerates its baseline audit.
spec=importlib.util.spec_from_file_location('stage14','scripts/adaptive_router_stage14_leadership_portfolio.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

OUT=Path('adaptive_router_stage17'); OUT.mkdir(exist_ok=True)
INIT=b.INIT; FEE=b.FEE; TAX=b.TAX; SLIP=b.SLIP
MAX_POS=b.MAX_POS; MAX_W=b.MAX_W; BUY_LIMIT_GAP=b.BUY_LIMIT_GAP; MAX_HOLD=b.MAX_HOLD
raw=b.raw; bench=b.bench; market=b.market

VARIANTS={
 'BASELINE':{'stop':None,'be_trigger':None,'trail_trigger':None,'trail_pct':None,'tp':None},
 'STOP8':{'stop':-.08,'be_trigger':None,'trail_trigger':None,'trail_pct':None,'tp':None},
 'STOP8_BE10':{'stop':-.08,'be_trigger':.10,'trail_trigger':None,'trail_pct':None,'tp':None},
 'STOP8_TRAIL15_10':{'stop':-.08,'be_trigger':None,'trail_trigger':.15,'trail_pct':.10,'tp':None},
 'STOP8_TP25':{'stop':-.08,'be_trigger':None,'trail_trigger':None,'trail_pct':None,'tp':.25},
}

# Protective orders for day T+1 are computed only from information known by T close.
def protective_levels(p,cfg):
    stop=None; tp=None
    if cfg['stop'] is not None: stop=p['buy']*(1+cfg['stop'])
    if cfg['be_trigger'] is not None and p['peak_known']>=p['buy']*(1+cfg['be_trigger']):
        stop=max(stop if stop is not None else -np.inf,p['buy'])
    if cfg['trail_trigger'] is not None and p['peak_known']>=p['buy']*(1+cfg['trail_trigger']):
        trail=p['peak_known']*(1-cfg['trail_pct'])
        stop=max(stop if stop is not None else -np.inf,trail)
    if cfg['tp'] is not None: tp=p['buy']*(1+cfg['tp'])
    return stop,tp

def exit_trade(pos,c,dt,px,reason,age,cash,trades):
    p=pos[c]; proceeds=p['shares']*px*(1-FEE-TAX); cash+=proceeds
    pnl=proceeds-p['cost']; ret=proceeds/p['cost']-1
    trades.append({'code':c,'entry_date':p['entry_date'],'exit_date':pd.Timestamp(dt),'shares':p['shares'],'buy':p['buy'],'sell':px,'days':age,'pnl':pnl,'ret':ret,'exit_reason':reason,'mae':p['mae'],'mfe':p['mfe']})
    del pos[c]
    return cash

def run_period(start,end,label,variant,cfg):
    d=raw[(raw.date.dt.year>=start)&(raw.date.dt.year<=end)].copy()
    dates=sorted(d.date.unique()); by={dt:z.set_index('code') for dt,z in d.groupby('date')}
    cash=INIT; pos={}; trades=[]; navrows=[]; last_close={}; corp_bridges=0
    for i,dt in enumerate(dates):
        day=by[dt]; prev_dt=dates[i-1] if i>0 else None; prev=by.get(prev_dt)
        if prev is not None:
            for c,p in list(pos.items()):
                if c in prev.index and c in day.index:
                    rr=b.bridge_ratio(float(prev.loc[c,'close']),float(day.loc[c,'open']))
                    if rr!=1.0:
                        p['shares']=max(1,int(round(p['shares']*rr))); p['buy']/=rr; p['cost']=p['shares']*p['buy']*(1+FEE); p['peak_known']/=rr; corp_bridges+=1
        # All exits for current day are based on orders/signals known by prior close.
        if prev is not None:
            m_bad=bool(market.loc[prev_dt,'market_bad3']) if prev_dt in market.index and pd.notna(market.loc[prev_dt,'market_bad3']) else True
            for c,p in list(pos.items()):
                if c not in prev.index or c not in day.index: continue
                dr=day.loc[c]; pr=prev.loc[c]; age=i-p['entry_i']
                stop,tp=protective_levels(p,cfg)
                reason=None; px=None
                # Conservative OHLC ambiguity rule: if stop and TP both touch, stop is assumed first.
                if stop is not None and float(dr.low)<=stop:
                    px=(float(dr.open) if float(dr.open)<=stop else stop)*(1-SLIP); reason='STOP'
                elif tp is not None and float(dr.high)>=tp:
                    px=tp*(1-SLIP); reason='TAKE_PROFIT'
                else:
                    exit_sig=m_bad or (pd.notna(pr.ma60) and float(pr.close)<float(pr.ma60)) or (pd.notna(pr.r60_pct) and float(pr.r60_pct)<.55) or age>=MAX_HOLD
                    if exit_sig: px=float(dr.open)*(1-SLIP); reason='BASE_SIGNAL'
                if reason is not None: cash=exit_trade(pos,c,dt,px,reason,age,cash,trades)
        # Entries are IDENTICAL to Stage14: T close decision -> T+1 precommitted limit.
        if prev is not None:
            m_ok=bool(market.loc[prev_dt,'market_ok']) if prev_dt in market.index and pd.notna(market.loc[prev_dt,'market_ok']) else False
            if m_ok and len(pos)<MAX_POS:
                cand=prev[prev.candidate].sort_values('score',ascending=False)
                for c,r in cand.iterrows():
                    if c in pos or len(pos)>=MAX_POS or c not in day.index: continue
                    limit=float(r.close)*(1+BUY_LIMIT_GAP); dr=day.loc[c]
                    if float(dr.low)>limit: continue
                    px=min(float(dr.open)*(1+SLIP),limit)
                    equity=cash+sum(p['shares']*last_close.get(k,p['buy']) for k,p in pos.items())
                    budget=min(equity*MAX_W,cash/(1+FEE)); sh=int(budget//px)
                    if sh<=0: continue
                    cost=sh*px*(1+FEE)
                    if cost>cash: continue
                    cash-=cost
                    # Do not use same-day high for a newly filled order: it may have occurred before the fill.
                    pos[c]={'shares':sh,'buy':px,'cost':cost,'entry_i':i,'entry_date':pd.Timestamp(dt),'peak_known':max(px,float(dr.close)),'mae':min(0.0,float(dr.close)/px-1),'mfe':max(0.0,float(dr.close)/px-1)}
        # End-of-day update becomes available only for NEXT day's protective order.
        for c,p in list(pos.items()):
            if c in day.index:
                dr=day.loc[c]; p['peak_known']=max(p['peak_known'],float(dr.high)); p['mae']=min(p['mae'],float(dr.low)/p['buy']-1); p['mfe']=max(p['mfe'],float(dr.high)/p['buy']-1)
        for c in day.index: last_close[c]=float(day.loc[c,'close'])
        mv=sum(p['shares']*last_close.get(c,p['buy']) for c,p in pos.items()); nav=cash+mv
        navrows.append({'variant':variant,'period':label,'date':pd.Timestamp(dt),'nav':nav,'cash':cash,'positions':len(pos),'exposure':mv/nav if nav>0 else 0})
    last_dt=dates[-1]; last=by[last_dt]
    for c,p in list(pos.items()):
        if c not in last.index: continue
        px=float(last.loc[c,'close'])*(1-SLIP); age=len(dates)-1-p['entry_i']; cash=exit_trade(pos,c,last_dt,px,'PERIOD_END',age,cash,trades)
    nav=pd.DataFrame(navrows); nav.loc[nav.index[-1],['nav','cash','positions','exposure']]=[cash,cash,0,0]
    peak=nav.nav.cummax(); dd=float((nav.nav/peak-1).min()); years=(nav.date.iloc[-1]-nav.date.iloc[0]).days/365.25
    end_nav=float(cash); cagr=(end_nav/INIT)**(1/years)-1; tr=pd.DataFrame(trades)
    if len(tr):
        gp=float(tr.loc[tr.pnl>0,'pnl'].sum()); gl=float(-tr.loc[tr.pnl<0,'pnl'].sum()); pf=gp/gl if gl>0 else (999. if gp>0 else 0.); win=float((tr.pnl>0).mean()); avg_mae=float(tr.mae.mean()); avg_mfe=float(tr.mfe.mean())
    else: pf=0.; win=0.; avg_mae=np.nan; avg_mfe=np.nan
    annual=[]
    for y,z in nav.groupby(nav.date.dt.year):
        annual.append({'variant':variant,'period':label,'year':int(y),'strategy_return':float(z.nav.iloc[-1]/z.nav.iloc[0]-1)})
    met={'variant':variant,'period':label,'end_nav':end_nav,'return':end_nav/INIT-1,'cagr':cagr,'max_dd':dd,'trades':int(len(tr)),'win_rate':win,'pf':float(pf),'avg_exposure':float(nav.exposure.mean()),'avg_mae':avg_mae,'avg_mfe':avg_mfe,'corp_action_bridges':corp_bridges}
    if len(tr):
        rc=tr.exit_reason.value_counts(); met.update({f'exit_{k.lower()}':int(v) for k,v in rc.items()})
    return met,nav,tr,pd.DataFrame(annual)

metrics=[]; navs=[]; trs=[]; anns=[]
for v,cfg in VARIANTS.items():
    for (start,end,label) in [(2008,2015,'PRIMARY_2008_2015'),(2016,2020,'STRESS_2016_2020')]:
        m,n,t,a=run_period(start,end,label,v,cfg); metrics.append(m); navs.append(n); 
        if len(t): t=t.assign(variant=v,period=label); trs.append(t)
        anns.append(a)
mdf=pd.DataFrame(metrics); navdf=pd.concat(navs,ignore_index=True); trdf=pd.concat(trs,ignore_index=True) if trs else pd.DataFrame(); andf=pd.concat(anns,ignore_index=True)
# Annual statistics and fixed profit-first assessment; 0050 is reference only, not a hard gate.
posyrs=andf.assign(pos=lambda x:x.strategy_return>0).groupby(['variant','period']).agg(positive_years=('pos','sum'),years=('year','count')).reset_index()
mdf=mdf.merge(posyrs,on=['variant','period'],how='left')
base=mdf[mdf.variant.eq('BASELINE')].set_index('period')
assess=[]
for _,r in mdf.iterrows():
    b0=base.loc[r.period]
    assess.append({'variant':r.variant,'period':r.period,'return_delta_vs_baseline':r['return']-b0['return'],'dd_improvement_vs_baseline':r['max_dd']-b0['max_dd'],'pf_delta_vs_baseline':r['pf']-b0['pf'],'profit_positive':bool(r['return']>0),'dd_under_30':bool(r['max_dd']>=-.30),'pf_over_1_1':bool(r['pf']>1.10)})
assess=pd.DataFrame(assess)
mdf.to_csv(OUT/'stage17_metrics.csv',index=False); navdf.to_csv(OUT/'stage17_daily_nav.csv',index=False); trdf.to_csv(OUT/'stage17_trades.csv',index=False); andf.to_csv(OUT/'stage17_annual.csv',index=False); assess.to_csv(OUT/'stage17_assessment.csv',index=False)
manifest={'stage':'17_exit_engineering','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'entry_rules_identical_to_stage14':True,'selection_rules_identical_to_stage14':True,'variant_count':len(VARIANTS),'variants':VARIANTS,'execution':'All signal/protective decisions use information known by T close; earliest execution T+1. Stop/TP prices are precommitted for T+1. Same-day stop+TP ambiguity assumes stop first.','retuned_after_result':False,'benchmark_role':'reference_only_not_hard_gate','objective_order':['positive_return','lower_max_drawdown','stability','profit_factor','win_rate','benchmark_reference']}
(OUT/'stage17_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('STAGE17_COMPLETE'); print(json.dumps(manifest,ensure_ascii=False,indent=2)); print('\nMETRICS'); print(mdf.to_string(index=False)); print('\nASSESSMENT'); print(assess.to_string(index=False)); print('\nANNUAL'); print(andf.to_string(index=False))
