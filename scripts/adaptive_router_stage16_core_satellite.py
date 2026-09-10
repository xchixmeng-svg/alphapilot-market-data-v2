from pathlib import Path
import json, runpy
import numpy as np, pandas as pd

# Reuse Stage14 frozen stock feature/candidate/entry/exit logic exactly.
g=runpy.run_path('scripts/adaptive_router_stage14_leadership_portfolio.py')
raw=g['raw']; bench=g['bench']; market=g['market']; bridge_ratio=g['bridge_ratio']; benchmark_metrics=g['benchmark_metrics']
INIT=g['INIT']; FEE=g['FEE']; TAX=g['TAX']; SLIP=g['SLIP']; MAX_POS=g['MAX_POS']; MAX_W=g['MAX_W']; BUY_LIMIT_GAP=g['BUY_LIMIT_GAP']; MAX_HOLD=g['MAX_HOLD']
PRIMARY=g['PRIMARY']; STRESS=g['STRESS']; ETF_TAX=.001
OUT=Path('adaptive_router_stage16'); OUT.mkdir(exist_ok=True)
bench=bench.copy().sort_values('date')
# Core defense is deliberately structural, not optimized: cash only when ALL three dimensions are bad for 3 consecutive days.
bench['hard_bad']=(bench.close<bench.ma120)&(bench.r60<0)&(bench.breadth<bench.breadth_thr)
bench['hard_risk_off3']=bench.hard_bad.rolling(3,min_periods=3).sum()>=3
bench['core_on']=~bench.hard_risk_off3
bench_by=bench.set_index('date')


def run_period(start,end,label):
    d=raw[(raw.date.dt.year>=start)&(raw.date.dt.year<=end)].copy(); dates=sorted(d.date.unique())
    by={dt:z.set_index('code') for dt,z in d.groupby('date')}
    cash=INIT; pos={}; etf_sh=0; trades=[]; navrows=[]; last_close={}; bridges=0; etf_turnover=0.; core_days=0

    def sell_etf(n,dt,use_close=False):
        nonlocal cash,etf_sh,etf_turnover
        if n<=0 or etf_sh<=0 or dt not in bench_by.index: return
        n=min(int(n),etf_sh); base=float(bench_by.loc[dt,'close' if use_close else 'open']); px=base*(1-SLIP)
        cash+=n*px*(1-FEE-ETF_TAX); etf_turnover+=n*px; etf_sh-=n

    for i,dt in enumerate(dates):
        day=by[dt]; prev_dt=dates[i-1] if i>0 else None; prev=by.get(prev_dt)
        if prev is not None:
            for c,p in list(pos.items()):
                if c in prev.index and c in day.index:
                    rr=bridge_ratio(float(prev.loc[c,'close']),float(day.loc[c,'open']))
                    if rr!=1.0: p['shares']=max(1,int(round(p['shares']*rr))); p['buy']/=rr; bridges+=1

        # Stock exits remain Stage14-identical.
        if prev is not None:
            m_bad=bool(market.loc[prev_dt,'market_bad3']) if prev_dt in market.index and pd.notna(market.loc[prev_dt,'market_bad3']) else True
            for c,p in list(pos.items()):
                if c not in prev.index or c not in day.index: continue
                pr=prev.loc[c]; age=i-p['entry_i']
                exit_sig=m_bad or (pd.notna(pr.ma60) and float(pr.close)<float(pr.ma60)) or (pd.notna(pr.r60_pct) and float(pr.r60_pct)<.55) or age>=MAX_HOLD
                if exit_sig:
                    px=float(day.loc[c,'open'])*(1-SLIP); proceeds=p['shares']*px*(1-FEE-TAX); cash+=proceeds
                    trades.append({'period':label,'code':c,'entry_date':p['entry_date'],'exit_date':pd.Timestamp(dt),'shares':p['shares'],'buy':p['buy'],'sell':px,'days':age,'pnl':proceeds-p['cost'],'ret':proceeds/p['cost']-1}); del pos[c]

        core_on=bool(bench_by.loc[prev_dt,'core_on']) if prev_dt is not None and prev_dt in bench_by.index and pd.notna(bench_by.loc[prev_dt,'core_on']) else False
        if core_on: core_days+=1
        if prev is not None and not core_on and etf_sh>0: sell_etf(etf_sh,dt)

        # Stock entries remain Stage14-identical; 0050 core is sold only to fund a qualifying alpha slot.
        if prev is not None:
            m_ok=bool(market.loc[prev_dt,'market_ok']) if prev_dt in market.index and pd.notna(market.loc[prev_dt,'market_ok']) else False
            if m_ok and len(pos)<MAX_POS:
                cand=prev[prev.candidate].sort_values('score',ascending=False)
                for c,r in cand.iterrows():
                    if c in pos or len(pos)>=MAX_POS or c not in day.index: continue
                    limit=float(r.close)*(1+BUY_LIMIT_GAP); dr=day.loc[c]
                    if float(dr.low)>limit: continue
                    stock_mv=sum(p['shares']*last_close.get(k,p['buy']) for k,p in pos.items())
                    etf_px=float(bench_by.loc[dt,'close']) if dt in bench_by.index else 0.
                    nav_pre=cash+stock_mv+etf_sh*etf_px
                    px=min(float(dr.open)*(1+SLIP),limit); target_budget=nav_pre*MAX_W
                    sh=int(target_budget//(px*(1+FEE)))
                    if sh<=0: continue
                    cost=sh*px*(1+FEE)
                    if cash<cost and etf_sh>0:
                        sellpx=float(bench_by.loc[dt,'open'])*(1-SLIP); net=sellpx*(1-FEE-ETF_TAX)
                        sell_etf(int(np.ceil((cost-cash)/net)),dt)
                    if cash<cost:
                        sh=int((cash/(1+FEE))//px); cost=sh*px*(1+FEE)
                    if sh<=0 or cost>cash: continue
                    cash-=cost; pos[c]={'shares':sh,'buy':px,'cost':cost,'entry_i':i,'entry_date':pd.Timestamp(dt)}

        # EXPLICIT core-satellite rebalance: whenever core is on, every non-stock dollar targets 0050, not merely accidental residual cash.
        if prev is not None and core_on and dt in bench_by.index:
            stock_mv=sum(p['shares']*last_close.get(k,p['buy']) for k,p in pos.items())
            etf_close=float(bench_by.loc[dt,'close']); nav_pre=cash+stock_mv+etf_sh*etf_close
            desired_core=max(0.,nav_pre-stock_mv)
            current_core=etf_sh*etf_close
            gap=desired_core-current_core
            if gap>0:
                px=float(bench_by.loc[dt,'open'])*(1+SLIP); n=int(min(gap,cash)/(px*(1+FEE)))
                if n>0:
                    cost=n*px*(1+FEE); cash-=cost; etf_sh+=n; etf_turnover+=n*px
            elif gap<0 and etf_sh>0:
                px=float(bench_by.loc[dt,'open'])*(1-SLIP); net=px*(1-FEE-ETF_TAX); n=int(min(etf_sh,np.ceil((-gap)/max(net,1e-9))))
                if n>0: sell_etf(n,dt)

        for c in day.index: last_close[c]=float(day.loc[c,'close'])
        stock_mv=sum(p['shares']*last_close.get(c,p['buy']) for c,p in pos.items()); etf_mv=etf_sh*(float(bench_by.loc[dt,'close']) if dt in bench_by.index else 0.)
        nav=cash+stock_mv+etf_mv
        navrows.append({'period':label,'date':pd.Timestamp(dt),'nav':nav,'cash':cash,'stock_positions':len(pos),'etf_shares':etf_sh,'stock_exposure':stock_mv/nav if nav else 0,'etf_exposure':etf_mv/nav if nav else 0,'total_exposure':(stock_mv+etf_mv)/nav if nav else 0,'core_on':core_on})

    last_dt=dates[-1]; last=by[last_dt]
    for c,p in list(pos.items()):
        if c in last.index:
            px=float(last.loc[c,'close'])*(1-SLIP); proceeds=p['shares']*px*(1-FEE-TAX); cash+=proceeds
            trades.append({'period':label,'code':c,'entry_date':p['entry_date'],'exit_date':pd.Timestamp(last_dt),'shares':p['shares'],'buy':p['buy'],'sell':px,'days':len(dates)-1-p['entry_i'],'pnl':proceeds-p['cost'],'ret':proceeds/p['cost']-1})
    if etf_sh>0: sell_etf(etf_sh,last_dt,use_close=True)
    nav=pd.DataFrame(navrows); nav.loc[nav.index[-1],['nav','cash','stock_positions','etf_shares','stock_exposure','etf_exposure','total_exposure']]=[cash,cash,0,0,0,0,0]
    peak=nav.nav.cummax(); dd=float((nav.nav/peak-1).min()); years=(nav.date.iloc[-1]-nav.date.iloc[0]).days/365.25; end_nav=float(cash); cagr=(end_nav/INIT)**(1/years)-1
    tr=pd.DataFrame(trades)
    if len(tr):
        gp=float(tr.loc[tr.pnl>0,'pnl'].sum()); gl=float(-tr.loc[tr.pnl<0,'pnl'].sum()); pf=gp/gl if gl>0 else (999. if gp>0 else 0.); win=float((tr.pnl>0).mean())
    else: pf=0.; win=0.
    bm=benchmark_metrics(start,end); annual=[]
    for y,z in nav.groupby(nav.date.dt.year):
        sr=float(z.nav.iloc[-1]/z.nav.iloc[0]-1); b=bench[bench.date.dt.year==y]; br=float(b.close.iloc[-1]/b.close.iloc[0]-1) if len(b)>1 else np.nan
        annual.append({'period':label,'year':int(y),'strategy_return':sr,'benchmark_return':br,'alpha':sr-br})
    core_rows=nav[nav.core_on.astype(bool)]
    metrics={'period':label,'end_nav':end_nav,'return':end_nav/INIT-1,'cagr':cagr,'max_dd':dd,'stock_trades':int(len(tr)),'stock_win_rate':win,'stock_pf':float(pf),'avg_total_exposure':float(nav.total_exposure.mean()),'avg_stock_exposure':float(nav.stock_exposure.mean()),'avg_etf_exposure':float(nav.etf_exposure.mean()),'avg_etf_exposure_when_core_on':float(core_rows.etf_exposure.mean()) if len(core_rows) else 0.,'core_on_day_rate':float(nav.core_on.mean()),'etf_turnover':etf_turnover,'corp_action_bridges':bridges,'benchmark_end_nav':bm['end_nav'],'benchmark_return':bm['return'],'benchmark_cagr':bm['cagr'],'benchmark_max_dd':bm['max_dd'],'alpha_return':end_nav/INIT-1-bm['return'],'alpha_cagr':cagr-bm['cagr']}
    return metrics,nav,tr,pd.DataFrame(annual)

pm,pnav,ptr,pann=run_period(*PRIMARY,'PRIMARY_2008_2015'); sm,snav,strd,sann=run_period(*STRESS,'STRESS_2016_2020')
nav=pd.concat([pnav,snav]); trades=pd.concat([ptr,strd]); annual=pd.concat([pann,sann]); nav.to_csv(OUT/'stage16_daily_nav.csv',index=False); trades.to_csv(OUT/'stage16_stock_trades.csv',index=False); annual.to_csv(OUT/'stage16_annual.csv',index=False)
pa=int((pann.alpha>0).sum()); sa=int((sann.alpha>0).sum())
# Material-treatment gate prevents another invalid 'core' test with near-zero 0050 exposure.
treatment_valid=bool(pm['avg_etf_exposure_when_core_on']>=.20 and sm['avg_etf_exposure_when_core_on']>=.20)
primary_pass=bool(treatment_valid and pm['return']>pm['benchmark_return'] and pm['cagr']>pm['benchmark_cagr'] and pm['max_dd']>=-.25 and pm['stock_pf']>1.05 and pm['stock_trades']>=30 and pa>=5)
stress_pass=bool(treatment_valid and sm['return']>sm['benchmark_return'] and sm['cagr']>sm['benchmark_cagr'] and sm['max_dd']>=-.30 and sm['stock_pf']>1.0 and sa>=3)
manifest={'stage':'16_explicit_core_satellite','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'variant_count':1,'stock_rules_identical_to_stage14':True,'only_capital_change':'0050 core remains on unless close<MA120 AND r60<0 AND breadth<prior rolling median for 3 consecutive days; alpha slots replace core dollar-for-dollar','benchmark':'0050','treatment_valid':treatment_valid,'minimum_core_exposure_gate':.20,'primary_metrics':pm,'stress_metrics':sm,'primary_alpha_years':pa,'stress_alpha_years':sa,'primary_pass':primary_pass,'stress_pass':stress_pass,'final_pass':bool(primary_pass and stress_pass),'retuned_after_result':False}
(OUT/'stage16_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)); print('STAGE16_COMPLETE'); print(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)); print('\nANNUAL'); print(annual.to_string(index=False))
