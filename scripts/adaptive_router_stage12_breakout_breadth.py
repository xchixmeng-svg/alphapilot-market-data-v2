from pathlib import Path
import json, zipfile, math
import numpy as np, pandas as pd

ROOT=Path('reference_bundle'); OUT=Path('adaptive_router_stage12'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; SLIP=.005; MAX_POS=4; MAX_W=.25

def load_year(y):
    if y == 2020:
        return pd.read_parquet(ROOT/'formal_2020'/'ohlcv_2020.parquet')
    z=ROOT/'ohlcv_raw'/f'yearly_{y}.zip'
    with zipfile.ZipFile(z) as zz:
        names=[n for n in zz.namelist() if n.endswith(('.csv','.parquet'))]
        parts=[]
        for n in names:
            with zz.open(n) as f:
                x=pd.read_parquet(f) if n.endswith('.parquet') else pd.read_csv(f)
                parts.append(x)
    return pd.concat(parts,ignore_index=True)

def norm(x):
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
    x=x[[c for c in list(dict.fromkeys(need+['amount'])) if c in x.columns]]
    x['date']=pd.to_datetime(x.date); x['code']=x.code.astype(str).str.extract(r'(\d+)')[0]
    for c in ['open','high','low','close','volume']+[c for c in ['amount'] if c in x]: x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'amount' not in x: x['amount']=x.close*x.volume
    return x.dropna(subset=['date','code','open','high','low','close']).sort_values(['code','date'])

raw=pd.concat([norm(load_year(y)) for y in range(2015,2021)],ignore_index=True).drop_duplicates(['date','code'],keep='last')
g=raw.groupby('code',group_keys=False)
raw['ma20']=g.close.transform(lambda s:s.rolling(20).mean()); raw['ma60']=g.close.transform(lambda s:s.rolling(60).mean())
raw['hi60_prev']=g.high.transform(lambda s:s.shift(1).rolling(60).max())
raw['r20']=g.close.pct_change(20); raw['r60']=g.close.pct_change(60)
raw['vol20']=g.close.pct_change().transform(lambda s:s.rolling(20).std())
raw['avg_amt20']=g.amount.transform(lambda s:s.rolling(20).mean())
raw['liq_pct']=raw.groupby('date').avg_amt20.rank(pct=True)
raw['breadth']=raw.assign(above=lambda x:x.close>x.ma60).groupby('date').above.transform('mean')
bd=raw[['date','breadth']].drop_duplicates().sort_values('date'); bd['thr']=bd.breadth.shift(1).rolling(252,min_periods=126).quantile(.60)
raw=raw.merge(bd[['date','thr']],on='date',how='left'); raw['breadth_high']=raw.breadth>=raw.thr
raw['break60']=raw.close/raw.hi60_prev-1
variants={
 'BASIC': lambda x:(x.break60>=0)&(x.breadth_high),
 'TREND': lambda x:(x.break60>=0)&(x.breadth_high)&(x.close>x.ma60)&(x.ma20>x.ma60),
 'MOM': lambda x:(x.break60>=0)&(x.breadth_high)&(x.r20>0)&(x.r60>0),
 'QUALITY': lambda x:(x.break60>=0)&(x.breadth_high)&(x.close>x.ma60)&(x.r20>0)&(x.liq_pct>=.5),
}

def benchmark(year):
    b=raw[(raw.code=='0050')&(raw.date.dt.year==year)].sort_values('date')
    if len(b)<2: return np.nan
    return float(b.close.iloc[-1]/b.close.iloc[0]-1)

def run(year,name,maskfn):
    d=raw[raw.date.dt.year==year].copy(); dates=sorted(d.date.unique()); cash=INIT; pos={}; trades=[]; navs=[]
    by={dt:z.set_index('code') for dt,z in d.groupby('date')}
    sig=d[maskfn(d)&(d.liq_pct>=.35)].copy(); sig['score']=sig.break60.rank(pct=True)+sig.r20.rank(pct=True)
    sigby={dt:z.sort_values('score',ascending=False) for dt,z in sig.groupby('date')}
    for i,dt in enumerate(dates):
        day=by[dt]
        for c,p in list(pos.items()):
            if i-p['entry_i']>=20 and c in day.index:
                px=float(day.loc[c,'open'])*(1-SLIP); proceeds=p['shares']*px*(1-FEE-TAX); cash+=proceeds
                trades.append({'code':c,'buy':p['buy'],'sell':px,'ret':px*(1-FEE-TAX)/(p['buy']*(1+FEE))-1}); del pos[c]
        if i>0:
            prev=dates[i-1]
            for _,r in sigby.get(prev,pd.DataFrame()).iterrows():
                c=r.code
                if c in pos or len(pos)>=MAX_POS or c not in day.index: continue
                limit=float(r.close)*(1+SLIP); dr=day.loc[c]
                if float(dr.low)>limit: continue
                px=min(float(dr.open)*(1+SLIP),limit)
                budget=min(INIT*MAX_W,cash/(1+FEE)); sh=int(budget//px)
                if sh<=0: continue
                cost=sh*px*(1+FEE)
                if cost>cash: continue
                cash-=cost; pos[c]={'shares':sh,'buy':px,'entry_i':i}
        mv=sum(p['shares']*float(day.loc[c,'close']) for c,p in pos.items() if c in day.index)
        navs.append(cash+mv)
    last=by[dates[-1]]
    for c,p in list(pos.items()):
        if c in last.index:
            px=float(last.loc[c,'close'])*(1-SLIP); cash+=p['shares']*px*(1-FEE-TAX); trades.append({'code':c,'buy':p['buy'],'sell':px,'ret':px*(1-FEE-TAX)/(p['buy']*(1+FEE))-1})
    nav=np.array(navs or [INIT],float); peak=np.maximum.accumulate(nav); dd=float(np.min(nav/peak-1))
    rets=np.array([t['ret'] for t in trades]); wins=rets[rets>0].sum(); losses=-rets[rets<0].sum(); pf=float(wins/losses) if losses>0 else (999. if wins>0 else 0.)
    ret=float(cash/INIT-1); bench=benchmark(year)
    return {'variant':name,'year':year,'return':ret,'benchmark_return':bench,'alpha':ret-bench if pd.notna(bench) else np.nan,'max_dd':dd,'trades':len(trades),'win_rate':float((rets>0).mean()) if len(rets) else 0.,'pf':pf}

rows=[]
for name,fn in variants.items():
    for y in range(2016,2021): rows.append(run(y,name,fn))
r=pd.DataFrame(rows); r.to_csv(OUT/'stage12_yearly.csv',index=False)
disc=r[r.year<=2018].groupby('variant').agg(alpha_mean=('alpha','mean'),alpha_years=('alpha',lambda s:int((s>0).sum())),ret_mean=('return','mean'),dd_min=('max_dd','min'),trades=('trades','sum'),pf_mean=('pf','mean')).reset_index()
disc['discovery_pass']=(disc.alpha_mean>0)&(disc.alpha_years>=2)&(disc.ret_mean>0)&(disc.dd_min>=-.25)&(disc.trades>=12)&(disc.pf_mean>1.05)
passers=disc[disc.discovery_pass].sort_values('alpha_mean',ascending=False)
selected=None if passers.empty else str(passers.iloc[0].variant)
if selected:
    o19=r[(r.variant==selected)&(r.year==2019)].iloc[0]; o20=r[(r.variant==selected)&(r.year==2020)].iloc[0]
    oos1=bool(o19.alpha>0 and o19['return']>0 and o19.pf>1 and o19.max_dd>=-.25 and o19.trades>=3)
    oos2=bool(oos1 and o20.alpha>0 and o20['return']>0 and o20.pf>1 and o20.max_dd>=-.25 and o20.trades>=3)
else: oos1=oos2=False
manifest={'stage':'12_breakout_breadth_engine','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'benchmark_gate_required':True,'benchmark':'0050','discovery_years':[2016,2017,2018],'oos1':2019,'oos2':2020,'selected':selected,'oos1_pass':oos1,'oos2_pass':oos2,'transaction_parameters_retuned_from_stage10':False,'variants':list(variants)}
(OUT/'stage12_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
disc.to_csv(OUT/'stage12_discovery.csv',index=False)
print('STAGE12_COMPLETE'); print(json.dumps(manifest,ensure_ascii=False,indent=2)); print('\nDISCOVERY'); print(disc.to_string(index=False)); print('\nYEARLY'); print(r.to_string(index=False))