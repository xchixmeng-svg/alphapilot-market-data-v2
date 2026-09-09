from pathlib import Path
import json, zipfile
import numpy as np, pandas as pd

ROOT=Path('reference_bundle'); OUT=Path('adaptive_router_stage13'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; SLIP=.005; MAX_POS=4; MAX_W=.25; HOLD=20
DISCOVERY=[2016,2017,2018]; OOS1=2019; OOS2=2020

def load_year(y):
    if y==2020: return pd.read_parquet(ROOT/'formal_2020'/'ohlcv_2020.parquet')
    z=ROOT/'ohlcv_raw'/f'yearly_{y}.zip'
    with zipfile.ZipFile(z) as zz:
        parts=[]
        for n in zz.namelist():
            if n.endswith('.csv'):
                with zz.open(n) as f: parts.append(pd.read_csv(f))
            elif n.endswith('.parquet'):
                with zz.open(n) as f: parts.append(pd.read_parquet(f))
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
    missing=[c for c in need if c not in x]
    if missing: raise RuntimeError(f'missing columns {missing}')
    x=x[[c for c in list(dict.fromkeys(need+['amount'])) if c in x]]
    s=x.date
    if pd.api.types.is_datetime64_any_dtype(s): x['date']=pd.to_datetime(s,errors='coerce')
    else:
        ss=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True); m=ss.str.fullmatch(r'\d{8}')
        dt=pd.Series(pd.NaT,index=x.index,dtype='datetime64[ns]')
        dt.loc[m]=pd.to_datetime(ss.loc[m],format='%Y%m%d',errors='coerce'); dt.loc[~m]=pd.to_datetime(ss.loc[~m],errors='coerce'); x['date']=dt
    x['code']=x.code.astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    for c in ['open','high','low','close','volume']+[c for c in ['amount'] if c in x]: x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'amount' not in x: x['amount']=x.close*x.volume
    return x.dropna(subset=['date','code','open','high','low','close']).sort_values(['code','date'])

raw=pd.concat([norm(load_year(y)) for y in range(2015,2021)],ignore_index=True).drop_duplicates(['date','code'],keep='last')
for y in range(2016,2021):
    if not (raw.date.dt.year==y).any(): raise RuntimeError(f'no rows {y}')

g=raw.groupby('code',group_keys=False)
ret1=g.close.pct_change()
for n in [10,20,40,60,120]: raw[f'r{n}']=g.close.pct_change(n)
for n in [20,60,120]: raw[f'ma{n}']=g.close.transform(lambda s,n=n:s.rolling(n).mean())
raw['hi60_prev']=g.high.transform(lambda s:s.shift(1).rolling(60).max())
raw['hi120_prev']=g.high.transform(lambda s:s.shift(1).rolling(120).max())
raw['break60']=raw.close/raw.hi60_prev-1; raw['break120']=raw.close/raw.hi120_prev-1
raw['ma60_gap']=raw.close/raw.ma60-1; raw['ma120_gap']=raw.close/raw.ma120-1
raw['vol20']=ret1.groupby(raw.code).transform(lambda s:s.rolling(20).std())
raw['range20']=(raw.high/raw.low-1).groupby(raw.code).transform(lambda s:s.rolling(20).mean())
raw['avg_amt20']=g.amount.transform(lambda s:s.rolling(20).mean()); raw['liq_pct']=raw.groupby('date').avg_amt20.rank(pct=True)
raw['vol_ratio20']=raw.volume/g.volume.transform(lambda s:s.rolling(20).mean())

# 0050 future return label, used ONLY in 2016-2018 discovery diagnostics.
b=raw[raw.code=='0050'][['date','close']].drop_duplicates('date').sort_values('date').copy(); b['bench_fwd20']=b.close.shift(-20)/b.close-1
raw['fwd20']=g.close.shift(-20)/raw.close-1
raw=raw.merge(b[['date','bench_fwd20']],on='date',how='left'); raw['fwd20_excess']=raw.fwd20-raw.bench_fwd20

# Common-stock-like research universe; exclude ETF-looking 00xx codes and 0050 itself.
elig=(raw.code.str.fullmatch(r'\d{4}'))&(~raw.code.str.startswith('00'))&(raw.close>0)&(raw.liq_pct>=.35)
features=['r10','r20','r40','r60','r120','break60','break120','ma60_gap','ma120_gap','vol20','range20','vol_ratio20']

# Causal feature values at T; future excess labels only evaluate discovery edge, never enter live score directly.
ic_rows=[]
for y in DISCOVERY:
    z=raw[elig&(raw.date.dt.year==y)].dropna(subset=['fwd20_excess']).copy()
    for f in features:
        vals=[]
        for dt,d in z[['date',f,'fwd20_excess']].dropna().groupby('date'):
            if len(d)>=30:
                c=d[f].corr(d.fwd20_excess,method='spearman')
                if pd.notna(c): vals.append(c)
        ic_rows.append({'year':y,'feature':f,'median_daily_ic':float(np.median(vals)) if vals else np.nan,'n_days':len(vals)})
ic=pd.DataFrame(ic_rows); ic.to_csv(OUT/'stage13_factor_ic.csv',index=False)
piv=ic.pivot(index='feature',columns='year',values='median_daily_ic')
rob=[]
for f,row in piv.iterrows():
    v=row.dropna()
    if len(v)==3 and ((v>0).all() or (v<0).all()):
        direction=1 if (v>0).all() else -1
        rob.append({'feature':f,'direction':direction,'min_abs_year_ic':float(v.abs().min()),'mean_abs_ic':float(v.abs().mean())})
robust=pd.DataFrame(rob).sort_values(['min_abs_year_ic','mean_abs_ic'],ascending=False) if rob else pd.DataFrame(columns=['feature','direction','min_abs_year_ic','mean_abs_ic'])
robust.to_csv(OUT/'stage13_robust_factors.csv',index=False)

# Freeze a small family from discovery-only robust factors. No 2019/2020 data selects features or weights.
top=robust.head(5).to_dict('records'); top3=top[:3]
if not top: raise RuntimeError('no robust discovery factors')

def score_frame(d, specs):
    s=pd.Series(0.0,index=d.index); used=0
    for q in specs:
        f=q['feature']; direction=q['direction']; p=d.groupby('date')[f].rank(pct=True)
        s += p if direction>0 else (1-p); used+=1
    return s/max(used,1)

variant_specs={'TOP1':top[:1],'TOP3_EQ':top3,'TOP5_EQ':top}
# economically predefined ensembles, included only if all named features survived discovery direction consistency
for name,names in {
    'MOMENTUM_CORE':['r20','r60','r120','break60'],
    'TREND_QUALITY':['r60','ma60_gap','break60','vol20'],
}.items():
    m={x['feature']:x for x in top+robust.to_dict('records')}
    specs=[m[f] for f in names if f in m]
    if len(specs)>=3: variant_specs[name]=specs

def benchmark(year):
    q=raw[(raw.code=='0050')&(raw.date.dt.year==year)].sort_values('date')
    if len(q)<2: raise RuntimeError(f'0050 missing {year}')
    return float(q.close.iloc[-1]/q.close.iloc[0]-1)

def run(year,name,specs):
    d=raw[raw.date.dt.year==year].copy(); dates=sorted(d.date.unique()); by={dt:z.set_index('code') for dt,z in d.groupby('date')}
    cand=d[elig.loc[d.index]].copy(); cand['score']=score_frame(cand,specs)
    cand=cand[cand.score>=.80]
    sigby={dt:z.sort_values(['score','liq_pct'],ascending=False) for dt,z in cand.groupby('date')}
    cash=INIT; pos={}; trades=[]; navs=[]
    for i,dt in enumerate(dates):
        day=by[dt]
        for c,p in list(pos.items()):
            if i-p['entry_i']>=HOLD and c in day.index:
                px=float(day.loc[c,'open'])*(1-SLIP); cash+=p['shares']*px*(1-FEE-TAX)
                trades.append(px*(1-FEE-TAX)/(p['buy']*(1+FEE))-1); del pos[c]
        if i>0:
            prev=dates[i-1]
            for _,r in sigby.get(prev,pd.DataFrame()).iterrows():
                c=r.code
                if c in pos or len(pos)>=MAX_POS or c not in day.index: continue
                limit=float(r.close)*(1+SLIP); dr=day.loc[c]
                if float(dr.low)>limit: continue
                px=min(float(dr.open)*(1+SLIP),limit); budget=min(INIT*MAX_W,cash/(1+FEE)); sh=int(budget//px)
                if sh<=0: continue
                cost=sh*px*(1+FEE)
                if cost<=cash: cash-=cost; pos[c]={'shares':sh,'buy':px,'entry_i':i}
        mv=sum(p['shares']*float(day.loc[c,'close']) for c,p in pos.items() if c in day.index); navs.append(cash+mv)
    last=by[dates[-1]]
    for c,p in list(pos.items()):
        if c in last.index:
            px=float(last.loc[c,'close'])*(1-SLIP); cash+=p['shares']*px*(1-FEE-TAX); trades.append(px*(1-FEE-TAX)/(p['buy']*(1+FEE))-1)
    nav=np.array(navs or [INIT]); peak=np.maximum.accumulate(nav); dd=float(np.min(nav/peak-1)); tr=np.array(trades)
    gp=tr[tr>0].sum(); gl=-tr[tr<0].sum(); pf=float(gp/gl) if gl>0 else (999. if gp>0 else 0.)
    ret=float(cash/INIT-1); bench=benchmark(year)
    return {'variant':name,'year':year,'return':ret,'benchmark_return':bench,'alpha':ret-bench,'max_dd':dd,'trades':len(tr),'win_rate':float((tr>0).mean()) if len(tr) else 0.,'pf':pf}

rows=[]
for name,specs in variant_specs.items():
    for y in DISCOVERY+[OOS1,OOS2]: rows.append(run(y,name,specs))
r=pd.DataFrame(rows); r.to_csv(OUT/'stage13_yearly.csv',index=False)
disc=r[r.year.isin(DISCOVERY)].groupby('variant').agg(alpha_mean=('alpha','mean'),alpha_years=('alpha',lambda s:int((s>0).sum())),ret_mean=('return','mean'),dd_min=('max_dd','min'),trades=('trades','sum'),pf_mean=('pf','mean')).reset_index()
disc['discovery_pass']=(disc.alpha_mean>0)&(disc.alpha_years>=2)&(disc.ret_mean>0)&(disc.dd_min>=-.25)&(disc.trades>=15)&(disc.pf_mean>1.05)
passers=disc[disc.discovery_pass].sort_values(['alpha_mean','pf_mean'],ascending=False)
selected=None if passers.empty else str(passers.iloc[0].variant)
if selected:
    o19=r[(r.variant==selected)&(r.year==OOS1)].iloc[0]; o20=r[(r.variant==selected)&(r.year==OOS2)].iloc[0]
    oos1=bool(o19.alpha>0 and o19['return']>0 and o19.pf>1 and o19.max_dd>=-.25 and o19.trades>=3)
    oos2=bool(oos1 and o20.alpha>0 and o20['return']>0 and o20.pf>1 and o20.max_dd>=-.25 and o20.trades>=3)
else: oos1=oos2=False
manifest={'stage':'13_benchmark_relative_alpha_ranking','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'target':'future_20d_excess_return_vs_0050','future_label_used_only_in_discovery_diagnostics':True,'discovery_years':DISCOVERY,'oos1':OOS1,'oos2':OOS2,'oos1_used_for_feature_discovery':False,'oos2_used_for_feature_or_variant_selection':False,'benchmark_gate_required':True,'benchmark':'0050','robust_factor_count':len(robust),'variant_count':len(variant_specs),'selected':selected,'oos1_pass':oos1,'oos2_pass':oos2,'t_plus_1':True,'integer_shares':True,'shared_cash':True,'scale_invariant_liquidity_rank':True}
(OUT/'stage13_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
disc.to_csv(OUT/'stage13_discovery.csv',index=False)
print('STAGE13_COMPLETE'); print(json.dumps(manifest,indent=2)); print('\nROBUST'); print(robust.to_string(index=False)); print('\nDISCOVERY'); print(disc.to_string(index=False)); print('\nYEARLY'); print(r.to_string(index=False))