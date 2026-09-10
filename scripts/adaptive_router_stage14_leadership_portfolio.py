from pathlib import Path
import json, zipfile, math
import numpy as np, pandas as pd

HIST=Path('historical_snapshot/data/history/2007-2019/raw')
REF=Path('reference_bundle/formal_2020/ohlcv_2020.parquet')
OUT=Path('adaptive_router_stage14'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; SLIP=.005
MAX_POS=4; MAX_W=.25; BUY_LIMIT_GAP=.01; MAX_HOLD=120
PRIMARY=(2008,2015); STRESS=(2016,2020)


def parse_date(s):
    if pd.api.types.is_datetime64_any_dtype(s): return pd.to_datetime(s,errors='coerce')
    ss=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
    out=pd.Series(pd.NaT,index=s.index,dtype='datetime64[ns]')
    m=ss.str.fullmatch(r'\d{8}')
    out.loc[m]=pd.to_datetime(ss.loc[m],format='%Y%m%d',errors='coerce')
    out.loc[~m]=pd.to_datetime(ss.loc[~m],errors='coerce')
    return out


def raw_year(y):
    if y==2020: return pd.read_parquet(REF)
    z=HIST/f'yearly_{y}.zip'
    if not z.exists(): raise RuntimeError(f'missing cached file {z}')
    with zipfile.ZipFile(z) as zz:
        names=[n for n in zz.namelist() if n.endswith('.csv')]
        if not names: raise RuntimeError(f'no csv in {z}')
        with zz.open(names[0]) as f: return pd.read_csv(f,low_memory=False)


def normalize(x,keep_etf=False):
    x=x.copy(); x.columns=[str(c).lower() for c in x.columns]
    ren={}
    for c in x.columns:
        if c in ('date','trade_date'): ren[c]='date'
        elif c in ('code','stock_id','symbol'): ren[c]='code'
        elif c in ('open','opening_price'): ren[c]='open'
        elif c in ('high','highest_price'): ren[c]='high'
        elif c in ('low','lowest_price'): ren[c]='low'
        elif c in ('close','closing_price'): ren[c]='close'
        elif c in ('volume','trade_volume'): ren[c]='volume'
        elif c in ('amount','trade_value','turnover'): ren[c]='amount'
    x=x.rename(columns=ren)
    need=['date','code','open','high','low','close','volume']
    miss=[c for c in need if c not in x.columns]
    if miss: raise RuntimeError(f'missing {miss}; columns={list(x.columns)}')
    x=x[[c for c in need+['amount'] if c in x.columns]]
    x['date']=parse_date(x.date)
    x['code']=x.code.astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    for c in ['open','high','low','close','volume']+[c for c in ['amount'] if c in x.columns]:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'amount' not in x.columns: x['amount']=x.close*x.volume
    x=x.dropna(subset=['date','code','open','high','low','close'])
    x=x[(x.close>0)&(x.volume>=0)&x.code.str.fullmatch(r'\d{4}')]
    if not keep_etf: x=x[~x.code.str.startswith('0')]
    return x.sort_values(['code','date'])

# Load once from preserved data only; no network market refetch.
frames=[]; etfs=[]
for y in range(2007,2021):
    r=raw_year(y); frames.append(normalize(r,False)); etfs.append(normalize(r,True).query("code=='0050'"))
raw=pd.concat(frames,ignore_index=True).drop_duplicates(['date','code'],keep='last')
bench=pd.concat(etfs,ignore_index=True).drop_duplicates('date',keep='last').sort_values('date')
if bench.empty: raise RuntimeError('0050 benchmark unavailable')

# Scale-invariant causal features.
g=raw.groupby('code',group_keys=False)
raw['r5']=g.close.pct_change(5); raw['r20']=g.close.pct_change(20); raw['r60']=g.close.pct_change(60); raw['r120']=g.close.pct_change(120)
raw['ma20']=g.close.transform(lambda s:s.rolling(20).mean()); raw['ma60']=g.close.transform(lambda s:s.rolling(60).mean()); raw['ma120']=g.close.transform(lambda s:s.rolling(120).mean())
raw['hi120_prev']=g.high.transform(lambda s:s.shift(1).rolling(120).max())
raw['avg_amt20']=g.amount.transform(lambda s:s.rolling(20).mean()); raw['avg_amt5']=g.amount.transform(lambda s:s.rolling(5).mean())
raw['vol_ratio']=raw.avg_amt5/raw.avg_amt20
raw['liq_pct']=raw.groupby('date').avg_amt20.rank(pct=True)
raw['r20_pct']=raw.groupby('date').r20.rank(pct=True); raw['r60_pct']=raw.groupby('date').r60.rank(pct=True); raw['r120_pct']=raw.groupby('date').r120.rank(pct=True)
raw['break120']=raw.close/raw.hi120_prev-1; raw['break120_pct']=raw.groupby('date').break120.rank(pct=True)
raw['volratio_pct']=raw.groupby('date').vol_ratio.rank(pct=True)
raw['rs60_stable20']=raw.groupby('code').r60_pct.transform(lambda s:(s>=.80).rolling(20,min_periods=20).sum())

# Market breadth: past-only rolling median; no fixed index-point thresholds.
raw['above120']=raw.close>raw.ma120
breadth=raw.groupby('date').above120.mean().rename('breadth').reset_index().sort_values('date')
breadth['breadth_thr']=breadth.breadth.shift(1).rolling(252,min_periods=126).median()
raw=raw.merge(breadth,on='date',how='left')

# 0050 market gate.
bench=bench[['date','open','high','low','close']].copy().sort_values('date')
bench['ma120']=bench.close.rolling(120).mean(); bench['r60']=bench.close.pct_change(60)
bench=bench.merge(breadth,on='date',how='left')
bench['market_ok']=(bench.close>bench.ma120)&(bench.r60>0)&(bench.breadth>=bench.breadth_thr)
bench['market_bad3']=(~bench.market_ok).rolling(3,min_periods=3).sum()>=3
market=bench.set_index('date')[['market_ok','market_bad3','close']]

# One fixed leadership score; no variant tournament.
raw['score']=.35*raw.r60_pct+.25*raw.r120_pct+.20*raw.r20_pct+.10*raw.break120_pct+.10*raw.volratio_pct
raw['candidate']=(raw.close>raw.ma120)&(raw.ma60>raw.ma120)&(raw.r20_pct>=.80)&(raw.r60_pct>=.85)&(raw.r120_pct>=.75)&(raw.break120>=-.08)&(raw.volratio_pct>=.50)&(raw.liq_pct>=.50)&(raw.rs60_stable20>=12)

# Approximate corporate-action bridge for pre-2020 raw history: preserve position market value across isolated >18% overnight discontinuities.
def bridge_ratio(prev_close,cur_open):
    if not np.isfinite(prev_close) or not np.isfinite(cur_open) or prev_close<=0 or cur_open<=0: return 1.0
    r=prev_close/cur_open
    return r if (r>1.22 or r<.82) else 1.0


def benchmark_metrics(start,end):
    b=bench[(bench.date.dt.year>=start)&(bench.date.dt.year<=end)].copy()
    if len(b)<2: raise RuntimeError('benchmark period missing')
    initial=float(b.close.iloc[0]); nav=INIT*b.close/initial
    peak=nav.cummax(); dd=float((nav/peak-1).min())
    years=(b.date.iloc[-1]-b.date.iloc[0]).days/365.25
    end_nav=float(nav.iloc[-1]); cagr=(end_nav/INIT)**(1/years)-1
    return {'end_nav':end_nav,'return':end_nav/INIT-1,'cagr':cagr,'max_dd':dd}


def run_period(start,end,label):
    d=raw[(raw.date.dt.year>=start)&(raw.date.dt.year<=end)].copy()
    dates=sorted(d.date.unique())
    by={dt:z.set_index('code') for dt,z in d.groupby('date')}
    cash=INIT; pos={}; trades=[]; navrows=[]; last_close={}; corp_bridges=0
    for i,dt in enumerate(dates):
        day=by[dt]; prev_dt=dates[i-1] if i>0 else None; prev=by.get(prev_dt)
        # Bridge suspected splits/reverse-splits before execution/NAV.
        if prev is not None:
            for c,p in list(pos.items()):
                if c in prev.index and c in day.index:
                    rr=bridge_ratio(float(prev.loc[c,'close']),float(day.loc[c,'open']))
                    if rr!=1.0:
                        p['shares']=max(1,int(round(p['shares']*rr))); p['buy']/=rr; corp_bridges+=1
        # T decision -> T+1 sell at adverse open.
        if prev is not None:
            m_bad=bool(market.loc[prev_dt,'market_bad3']) if prev_dt in market.index and pd.notna(market.loc[prev_dt,'market_bad3']) else True
            for c,p in list(pos.items()):
                if c not in prev.index or c not in day.index: continue
                pr=prev.loc[c]
                age=i-p['entry_i']
                exit_sig=m_bad or (pd.notna(pr.ma60) and float(pr.close)<float(pr.ma60)) or (pd.notna(pr.r60_pct) and float(pr.r60_pct)<.55) or age>=MAX_HOLD
                if exit_sig:
                    px=float(day.loc[c,'open'])*(1-SLIP)
                    proceeds=p['shares']*px*(1-FEE-TAX); cash+=proceeds
                    pnl=proceeds-p['cost']; ret=proceeds/p['cost']-1
                    trades.append({'period':label,'code':c,'entry_date':p['entry_date'],'exit_date':pd.Timestamp(dt),'shares':p['shares'],'buy':p['buy'],'sell':px,'days':age,'pnl':pnl,'ret':ret})
                    del pos[c]
        # T decision -> T+1 precommitted limit buy, after sells free cash.
        if prev is not None:
            m_ok=bool(market.loc[prev_dt,'market_ok']) if prev_dt in market.index and pd.notna(market.loc[prev_dt,'market_ok']) else False
            if m_ok and len(pos)<MAX_POS:
                cand=prev[prev.candidate].sort_values('score',ascending=False)
                for c,r in cand.iterrows():
                    if c in pos or len(pos)>=MAX_POS or c not in day.index: continue
                    limit=float(r.close)*(1+BUY_LIMIT_GAP); dr=day.loc[c]
                    if float(dr.low)>limit: continue
                    px=min(float(dr.open)*(1+SLIP),limit)
                    budget=min((cash+sum(p['shares']*last_close.get(k,p['buy']) for k,p in pos.items()))*MAX_W,cash/(1+FEE))
                    sh=int(budget//px)
                    if sh<=0: continue
                    cost=sh*px*(1+FEE)
                    if cost>cash: continue
                    cash-=cost; pos[c]={'shares':sh,'buy':px,'cost':cost,'entry_i':i,'entry_date':pd.Timestamp(dt)}
        for c in day.index: last_close[c]=float(day.loc[c,'close'])
        mv=sum(p['shares']*last_close.get(c,p['buy']) for c,p in pos.items())
        nav=cash+mv
        navrows.append({'period':label,'date':pd.Timestamp(dt),'nav':nav,'cash':cash,'positions':len(pos),'exposure':mv/nav if nav>0 else 0})
    # Liquidate at final close with adverse slippage so ending NAV is realizable.
    last_dt=dates[-1]; last=by[last_dt]
    for c,p in list(pos.items()):
        if c not in last.index: continue
        px=float(last.loc[c,'close'])*(1-SLIP); proceeds=p['shares']*px*(1-FEE-TAX); cash+=proceeds
        pnl=proceeds-p['cost']; ret=proceeds/p['cost']-1
        trades.append({'period':label,'code':c,'entry_date':p['entry_date'],'exit_date':pd.Timestamp(last_dt),'shares':p['shares'],'buy':p['buy'],'sell':px,'days':len(dates)-1-p['entry_i'],'pnl':pnl,'ret':ret})
    nav=pd.DataFrame(navrows)
    if nav.empty: raise RuntimeError(f'no nav {label}')
    nav.loc[nav.index[-1],'nav']=cash; nav.loc[nav.index[-1],'cash']=cash; nav.loc[nav.index[-1],'positions']=0; nav.loc[nav.index[-1],'exposure']=0
    peak=nav.nav.cummax(); dd=float((nav.nav/peak-1).min())
    years=(nav.date.iloc[-1]-nav.date.iloc[0]).days/365.25
    end_nav=float(cash); cagr=(end_nav/INIT)**(1/years)-1
    tr=pd.DataFrame(trades)
    if len(tr):
        gp=float(tr.loc[tr.pnl>0,'pnl'].sum()); gl=float(-tr.loc[tr.pnl<0,'pnl'].sum()); pf=gp/gl if gl>0 else (999. if gp>0 else 0.)
        win=float((tr.pnl>0).mean()); top3_share=float(tr.nlargest(min(3,len(tr)),'pnl').pnl.sum()/gp) if gp>0 else np.nan
    else: pf=0.; win=0.; top3_share=np.nan
    bm=benchmark_metrics(start,end)
    annual=[]
    for y,z in nav.groupby(nav.date.dt.year):
        sr=float(z.nav.iloc[-1]/z.nav.iloc[0]-1)
        b=bench[bench.date.dt.year==y]
        br=float(b.close.iloc[-1]/b.close.iloc[0]-1) if len(b)>1 else np.nan
        annual.append({'period':label,'year':int(y),'strategy_return':sr,'benchmark_return':br,'alpha':sr-br})
    metrics={'period':label,'start':start,'end':end,'end_nav':end_nav,'return':end_nav/INIT-1,'cagr':cagr,'max_dd':dd,'trades':int(len(tr)),'win_rate':win,'pf':float(pf),'avg_exposure':float(nav.exposure.mean()),'top3_gross_profit_share':top3_share,'corp_action_bridges':corp_bridges,'benchmark_end_nav':bm['end_nav'],'benchmark_return':bm['return'],'benchmark_cagr':bm['cagr'],'benchmark_max_dd':bm['max_dd'],'alpha_return':end_nav/INIT-1-bm['return'],'alpha_cagr':cagr-bm['cagr']}
    return metrics,nav,tr,pd.DataFrame(annual)

pm,pnav,ptr,pann=run_period(*PRIMARY,'PRIMARY_2008_2015')
sm,snav,strd,sann=run_period(*STRESS,'STRESS_2016_2020')
nav=pd.concat([pnav,snav],ignore_index=True); trades=pd.concat([ptr,strd],ignore_index=True); annual=pd.concat([pann,sann],ignore_index=True)
nav.to_csv(OUT/'stage14_daily_nav.csv',index=False); trades.to_csv(OUT/'stage14_trades.csv',index=False); annual.to_csv(OUT/'stage14_annual.csv',index=False)

# Locked pass gates: portfolio-level, not factor IC. Stress is a veto only and is never used to retune rules.
primary_alpha_years=int((pann.alpha>0).sum())
primary_pass=bool(pm['return']>pm['benchmark_return'] and pm['cagr']>pm['benchmark_cagr'] and pm['max_dd']>=-.25 and pm['pf']>1.10 and pm['trades']>=30 and primary_alpha_years>=5)
stress_alpha_years=int((sann.alpha>0).sum())
stress_pass=bool(sm['return']>sm['benchmark_return'] and sm['cagr']>sm['benchmark_cagr'] and sm['max_dd']>=-.30 and sm['pf']>1.0 and stress_alpha_years>=3)
final_pass=bool(primary_pass and stress_pass)
manifest={'stage':'14_fixed_leadership_portfolio','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'variant_count':1,'benchmark':'0050','execution':'T close decision -> T+1; buy precommitted +1% limit with 0.5% adverse slippage; sell T+1 open with 0.5% adverse slippage; fee/tax included','integer_shares':True,'shared_cash':True,'max_positions':MAX_POS,'max_weight':MAX_W,'max_hold_days':MAX_HOLD,'corp_action_method':'inferred_overnight_price_ratio_bridge_approximate','primary_metrics':pm,'stress_metrics':sm,'primary_alpha_years':primary_alpha_years,'stress_alpha_years':stress_alpha_years,'primary_pass':primary_pass,'stress_pass':stress_pass,'final_pass':final_pass,'retuned_after_result':False}
(OUT/'stage14_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
print('STAGE14_COMPLETE'); print(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)); print('\nANNUAL'); print(annual.to_string(index=False))
