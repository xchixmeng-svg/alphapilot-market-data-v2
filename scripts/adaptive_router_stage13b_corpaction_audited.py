from pathlib import Path
import json, zipfile, math
import numpy as np, pandas as pd

ROOT=Path('reference_bundle'); OUT=Path('adaptive_router_stage13b'); OUT.mkdir(exist_ok=True)
INIT=1_300_000.0; FEE=.000855; TAX=.003; SLIP=.005; MAX_POS=4; MAX_W=.25; HOLD=20
DISC=[2016,2017,2018]; OOS1=2019; OOS2=2020

def load_year(y):
    if y==2020: return pd.read_parquet(ROOT/'formal_2020'/'ohlcv_2020.parquet')
    with zipfile.ZipFile(ROOT/'ohlcv_raw'/f'yearly_{y}.zip') as z:
        n=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with z.open(n) as f: return pd.read_csv(f,dtype={'code':str},low_memory=False)

def norm(x):
    x=x.copy(); x.columns=[str(c).lower() for c in x.columns]
    ren={}
    for c in x.columns:
        if c in ('date','trade_date'):ren[c]='date'
        elif c in ('code','stock_id','symbol'):ren[c]='code'
        elif c in ('open','opening_price'):ren[c]='open'
        elif c in ('high','highest_price'):ren[c]='high'
        elif c in ('low','lowest_price'):ren[c]='low'
        elif c in ('close','closing_price'):ren[c]='close'
        elif c in ('volume','trade_volume'):ren[c]='volume'
        elif c in ('amount','trade_value','turnover'):ren[c]='amount'
    x=x.rename(columns=ren)
    s=x.date
    if pd.api.types.is_datetime64_any_dtype(s): x['date']=pd.to_datetime(s,errors='coerce')
    else:
        ss=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True); m=ss.str.fullmatch(r'\d{8}')
        d=pd.Series(pd.NaT,index=x.index,dtype='datetime64[ns]'); d.loc[m]=pd.to_datetime(ss.loc[m],format='%Y%m%d',errors='coerce'); d.loc[~m]=pd.to_datetime(ss.loc[~m],errors='coerce'); x['date']=d
    x['code']=x.code.astype(str).str.replace(r'\.0$','',regex=True).str.extract(r'(\d+)')[0].str.zfill(4)
    for c in ['open','high','low','close','volume']+[c for c in ['amount'] if c in x]: x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'amount' not in x:x['amount']=x.close*x.volume
    return x.dropna(subset=['date','code','open','high','low','close','volume'])

raw=pd.concat([norm(load_year(y)) for y in range(2015,2021)],ignore_index=True).sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)
g=raw.groupby('code',group_keys=False)
raw['prev_close']=g.close.shift(1); raw['bridge']=raw.prev_close/raw.open
raw['action_flag']=raw.prev_close.notna()&((raw.bridge<.82)|(raw.bridge>1.18)); raw.loc[~raw.action_flag,'bridge']=1.0
raw.loc[(raw.bridge<.10)|(raw.bridge>10.0),'bridge']=1.0
raw['adj_factor']=raw.groupby('code').bridge.cumprod()
for c in ['open','high','low','close']:raw['adj_'+c]=raw[c]*raw.adj_factor
raw['action_recent']=raw.groupby('code').action_flag.transform(lambda s:s.shift(1).rolling(5,min_periods=1).max()).fillna(False).astype(bool)
raw['amount']=raw.close*raw.volume; g=raw.groupby('code',group_keys=False)
for n in [10,20,40,60,120]:raw[f'r{n}']=g.adj_close.transform(lambda s,n=n:s.pct_change(n,fill_method=None))
for n in [20,60,120]:raw[f'ma{n}']=g.adj_close.transform(lambda s,n=n:s.rolling(n,min_periods=n).mean())
raw['hi60_prev']=g.adj_high.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max()); raw['hi120_prev']=g.adj_high.transform(lambda s:s.shift(1).rolling(120,min_periods=120).max())
raw['break60']=raw.adj_close/raw.hi60_prev-1; raw['break120']=raw.adj_close/raw.hi120_prev-1
raw['ma60_gap']=raw.adj_close/raw.ma60-1; raw['ma120_gap']=raw.adj_close/raw.ma120-1
r1=g.adj_close.transform(lambda s:s.pct_change(fill_method=None)); raw['vol20']=r1.groupby(raw.code).transform(lambda s:s.rolling(20,min_periods=20).std())
raw['range20']=(raw.adj_high/raw.adj_low-1).groupby(raw.code).transform(lambda s:s.rolling(20,min_periods=20).mean())
raw['amt20']=g.amount.transform(lambda s:s.rolling(20,min_periods=20).mean()); raw['liq_pct']=raw.groupby('date').amt20.rank(pct=True)
raw['vol_ratio20']=raw.volume/g.volume.transform(lambda s:s.rolling(20,min_periods=20).mean())

b=raw[raw.code=='0050'][['date','adj_close']].drop_duplicates('date').sort_values('date').copy(); b['bench_fwd20']=b.adj_close.shift(-20)/b.adj_close-1
raw['fwd20']=g.adj_close.transform(lambda s:s.shift(-20)/s-1); raw=raw.merge(b[['date','bench_fwd20']],on='date',how='left'); raw['fwd20_excess']=raw.fwd20-raw.bench_fwd20
elig=raw.code.str.fullmatch(r'[1-9]\d{3}')&(raw.liq_pct>=.35)&(~raw.action_recent)
features=['r10','r20','r40','r60','r120','break60','break120','ma60_gap','ma120_gap','vol20','range20','vol_ratio20']

ics=[]
for y in DISC:
    z=raw[elig&(raw.date.dt.year==y)].dropna(subset=['fwd20_excess'])
    for f in features:
        vals=[]
        for _,d in z[['date',f,'fwd20_excess']].dropna().groupby('date'):
            if len(d)>=30:
                c=d[f].corr(d.fwd20_excess,method='spearman')
                if pd.notna(c):vals.append(c)
        ics.append({'year':y,'feature':f,'median_ic':float(np.median(vals)) if vals else np.nan,'n':len(vals)})
ic=pd.DataFrame(ics); ic.to_csv(OUT/'factor_ic.csv',index=False); p=ic.pivot(index='feature',columns='year',values='median_ic')
rob=[]
for f,v in p.iterrows():
    v=v.dropna()
    if len(v)==3 and ((v>0).all() or (v<0).all()):rob.append({'feature':f,'direction':1 if (v>0).all() else -1,'min_abs_ic':float(v.abs().min()),'mean_abs_ic':float(v.abs().mean())})
rob=pd.DataFrame(rob).sort_values(['min_abs_ic','mean_abs_ic'],ascending=False); rob.to_csv(OUT/'robust_factors.csv',index=False)
if rob.empty:raise RuntimeError('no robust factors')
top=rob.head(5).to_dict('records'); variants={'TOP1':top[:1],'TOP3_EQ':top[:3],'TOP5_EQ':top}
allm={x['feature']:x for x in rob.to_dict('records')}
for name,names in {'MOM_CORE':['r20','r60','r120','break60'],'TREND_QUALITY':['r60','ma60_gap','break60','vol20']}.items():
    sp=[allm[x] for x in names if x in allm]
    if len(sp)>=3:variants[name]=sp

def score(d,sp):
    s=pd.Series(0.,index=d.index)
    for q in sp:
        p=d.groupby('date')[q['feature']].rank(pct=True); s+=p if q['direction']>0 else 1-p
    return s/len(sp)

def bench(y):
    q=raw[(raw.code=='0050')&(raw.date.dt.year==y)].sort_values('date'); return float(q.adj_close.iloc[-1]/q.adj_close.iloc[0]-1)

def sim(y,name,sp):
    d=raw[raw.date.dt.year==y].copy(); dates=sorted(d.date.unique()); by={dt:x.set_index('code',drop=False) for dt,x in d.groupby('date')}
    c=d[elig.loc[d.index]].copy(); c['score']=score(c,sp); c=c[c.score>=.80]; sig={dt:x.sort_values(['score','liq_pct'],ascending=False) for dt,x in c.groupby('date')}
    cash=INIT; pos={}; pend=[]; trades=[]; navs=[]
    for i,dt in enumerate(dates):
        day=by[dt]
        # existing holdings: bridge shares across inferred corporate action
        for code,p0 in list(pos.items()):
            if code in day.index and bool(day.loc[code].action_flag): p0['shares']=max(int(math.floor(p0['shares']*float(day.loc[code].bridge))),0)
        # fixed HOLD exits at T+1 open
        for code,p0 in list(pos.items()):
            if i-p0['entry_i']>=HOLD and code in day.index:
                px=float(day.loc[code].open)*(1-SLIP); proceeds=p0['shares']*px*(1-FEE-TAX); cash+=proceeds; trades.append(proceeds-p0['cost']); del pos[code]
        # execute yesterday signals; skip overnight corporate-action bridge days
        for o in pend:
            code=o['code']
            if code in pos or len(pos)>=MAX_POS or code not in day.index:continue
            r=day.loc[code]
            if bool(r.action_flag):continue
            if float(r.low)>o['limit']:continue
            px=min(float(r.open)*(1+SLIP),o['limit']); budget=min(INIT*MAX_W,cash/(1+FEE)); sh=int(budget//px)
            if sh<=0:continue
            cost=sh*px*(1+FEE)
            if cost<=cash:cash-=cost;pos[code]={'shares':sh,'cost':cost,'entry_i':i}
        pend=[]
        mv=sum(p0['shares']*float(day.loc[code].close) for code,p0 in pos.items() if code in day.index); navs.append(cash+mv)
        slots=MAX_POS-len(pos)
        if slots>0:
            for _,r in sig.get(dt,pd.DataFrame()).head(slots).iterrows():
                if r.code not in pos:pend.append({'code':r.code,'limit':float(r.close)*(1+SLIP)})
    last=by[dates[-1]]
    for code,p0 in list(pos.items()):
        if code in last.index:
            px=float(last.loc[code].close)*(1-SLIP); proceeds=p0['shares']*px*(1-FEE-TAX); cash+=proceeds; trades.append(proceeds-p0['cost'])
    n=np.array(navs or [INIT]); dd=float(np.min(n/np.maximum.accumulate(n)-1)); t=np.array(trades); gp=t[t>0].sum();gl=-t[t<0].sum();pf=float(gp/gl) if gl>0 else (999. if gp>0 else 0.)
    rr=float(cash/INIT-1); bb=bench(y);return {'variant':name,'year':y,'return':rr,'benchmark_return':bb,'alpha':rr-bb,'max_dd':dd,'trades':len(t),'win_rate':float((t>0).mean()) if len(t) else 0.,'pf':pf}

rows=[]
for name,sp in variants.items():
    for y in DISC+[OOS1,OOS2]:rows.append(sim(y,name,sp))
r=pd.DataFrame(rows);r.to_csv(OUT/'yearly.csv',index=False)
disc=r[r.year.isin(DISC)].groupby('variant').agg(alpha_mean=('alpha','mean'),alpha_years=('alpha',lambda s:int((s>0).sum())),ret_mean=('return','mean'),dd_min=('max_dd','min'),trades=('trades','sum'),pf_mean=('pf','mean')).reset_index(); disc['discovery_pass']=(disc.alpha_mean>0)&(disc.alpha_years>=2)&(disc.ret_mean>0)&(disc.dd_min>=-.25)&(disc.trades>=15)&(disc.pf_mean>1.05);disc.to_csv(OUT/'discovery.csv',index=False)
ps=disc[disc.discovery_pass].sort_values(['alpha_mean','pf_mean'],ascending=False); selected=None if ps.empty else str(ps.iloc[0].variant)
if selected:
    a=r[(r.variant==selected)&(r.year==OOS1)].iloc[0];b2=r[(r.variant==selected)&(r.year==OOS2)].iloc[0];o1=bool(a.alpha>0 and a['return']>0 and a.pf>1 and a.max_dd>=-.25 and a.trades>=3);o2=bool(o1 and b2.alpha>0 and b2['return']>0 and b2.pf>1 and b2.max_dd>=-.25 and b2.trades>=3)
else:o1=o2=False
m={'stage':'13b_benchmark_alpha_corpaction_audited','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'corporate_action_method':'inferred_bridge_0p82_1p18_same_as_stage7','future_label_discovery_only':True,'benchmark':'0050','benchmark_gate_required':True,'robust_factor_count':len(rob),'variant_count':len(variants),'selected':selected,'oos1_pass':o1,'oos2_pass':o2,'oos1_used_for_feature_discovery':False,'oos2_used_for_selection':False,'t_plus_1':True,'integer_shares':True,'shared_cash':True};(OUT/'manifest.json').write_text(json.dumps(m,indent=2))
print('STAGE13B_COMPLETE');print(json.dumps(m,indent=2));print('\nROBUST');print(rob.to_string(index=False));print('\nDISCOVERY');print(disc.to_string(index=False));print('\nYEARLY');print(r.to_string(index=False))