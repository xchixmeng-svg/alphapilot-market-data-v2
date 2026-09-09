from pathlib import Path
import json, zipfile
import numpy as np, pandas as pd

HIST=Path('historical_snapshot/data/history/2007-2019/raw')
REF=Path('reference_bundle/formal_2020/ohlcv_2020.parquet')
OUT=Path('adaptive_router_stage13'); OUT.mkdir(exist_ok=True)
DISC_YEARS=list(range(2008,2013)); OOS_YEARS=[2013,2014,2015]; STRESS_YEARS=list(range(2016,2021))


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
    miss=[c for c in need if c not in x.columns]
    if miss: raise RuntimeError(f'missing columns {miss}; got {list(x.columns)}')
    x=x[[c for c in need+['amount'] if c in x.columns]]
    s=x.date
    if pd.api.types.is_datetime64_any_dtype(s): x['date']=pd.to_datetime(s,errors='coerce')
    else:
        ss=s.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
        m=ss.str.fullmatch(r'\d{8}')
        dt=pd.Series(pd.NaT,index=x.index,dtype='datetime64[ns]')
        dt.loc[m]=pd.to_datetime(ss.loc[m],format='%Y%m%d',errors='coerce')
        dt.loc[~m]=pd.to_datetime(ss.loc[~m],errors='coerce')
        x['date']=dt
    x['code']=x.code.astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    for c in ['open','high','low','close','volume']+[c for c in ['amount'] if c in x.columns]: x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'amount' not in x.columns: x['amount']=x.close*x.volume
    x=x.dropna(subset=['date','code','open','high','low','close'])
    x=x[(x.code.str.fullmatch(r'\d{4}')) & (~x.code.str.startswith('0')) & (x.close>0) & (x.volume>=0)]
    return x.sort_values(['code','date'])


def load_year(y):
    if y==2020: return norm(pd.read_parquet(REF))
    z=HIST/f'yearly_{y}.zip'
    if not z.exists(): raise RuntimeError(f'missing cached historical file {z}')
    with zipfile.ZipFile(z) as zz:
        n=[n for n in zz.namelist() if n.endswith('.csv')][0]
        with zz.open(n) as f: return norm(pd.read_csv(f))

# 2007 warm-up + 2008-2020 analysis, all from already-preserved repository/artifact data.
raw=pd.concat([load_year(y) for y in range(2007,2021)],ignore_index=True).drop_duplicates(['date','code'],keep='last')
for y in range(2007,2021):
    if int((raw.date.dt.year==y).sum())==0: raise RuntimeError(f'no rows for {y}')

# 0050 benchmark comes from same preserved yearly files; load separately because ETF rows are excluded above.
def load_0050(y):
    if y==2020: x=pd.read_parquet(REF)
    else:
        with zipfile.ZipFile(HIST/f'yearly_{y}.zip') as zz:
            n=[n for n in zz.namelist() if n.endswith('.csv')][0]
            with zz.open(n) as f: x=pd.read_csv(f)
    x.columns=[str(c).lower() for c in x.columns]
    s=x['date']; ss=s.astype(str).str.replace(r'\.0$','',regex=True)
    x['date']=pd.to_datetime(ss,format='%Y%m%d',errors='coerce') if not pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s)
    x['code']=x['code'].astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    x['close']=pd.to_numeric(x['close'],errors='coerce')
    return x.loc[x.code.eq('0050'),['date','close']]
bench=pd.concat([load_0050(y) for y in range(2007,2021)]).drop_duplicates('date').sort_values('date')
for h in [5,20,60,120]: bench[f'b_r{h}']=bench.close.pct_change(h)
bench['b_fwd20']=bench.close.shift(-20)/bench.close-1

# Causal stock features.
g=raw.groupby('code',group_keys=False)
raw['r5']=g.close.pct_change(5); raw['r20']=g.close.pct_change(20); raw['r60']=g.close.pct_change(60); raw['r120']=g.close.pct_change(120)
raw['ma20']=g.close.transform(lambda s:s.rolling(20).mean()); raw['ma60']=g.close.transform(lambda s:s.rolling(60).mean()); raw['ma120']=g.close.transform(lambda s:s.rolling(120).mean())
raw['hi60_prev']=g.high.transform(lambda s:s.shift(1).rolling(60).max()); raw['hi120_prev']=g.high.transform(lambda s:s.shift(1).rolling(120).max())
raw['vol20']=g.close.pct_change().transform(lambda s:s.rolling(20).std())
raw['avg_amt20']=g.amount.transform(lambda s:s.rolling(20).mean()); raw['amt5']=g.amount.transform(lambda s:s.rolling(5).mean())
raw['fwd20']=g.close.shift(-20)/raw.close-1
raw=raw.merge(bench.drop(columns='close'),on='date',how='left')
raw['excess20']=raw.fwd20-raw.b_fwd20
raw['rs20']=raw.r20-raw.b_r20; raw['rs60']=raw.r60-raw.b_r60; raw['rs120']=raw.r120-raw.b_r120
raw['break60']=raw.close/raw.hi60_prev-1; raw['break120']=raw.close/raw.hi120_prev-1
raw['trend120']=raw.close/raw.ma120-1; raw['vol_ratio']=raw.amt5/raw.avg_amt20
raw['liq_pct']=raw.groupby('date').avg_amt20.rank(pct=True)
raw['vol20_pct']=raw.groupby('date').vol20.rank(pct=True)
raw['rs20_pct']=raw.groupby('date').rs20.rank(pct=True); raw['rs60_pct']=raw.groupby('date').rs60.rank(pct=True); raw['rs120_pct']=raw.groupby('date').rs120.rank(pct=True)
raw['break120_pct']=raw.groupby('date').break120.rank(pct=True); raw['volratio_pct']=raw.groupby('date').vol_ratio.rank(pct=True)
raw['pullback5_pct']=raw.groupby('date').r5.rank(pct=True,ascending=True)
raw['breadth']=raw.assign(above=lambda x:x.close>x.ma120).groupby('date').above.transform('mean')
bd=raw[['date','breadth']].drop_duplicates().sort_values('date'); bd['breadth_thr']=bd.breadth.shift(1).rolling(252,min_periods=126).median()
raw=raw.merge(bd[['date','breadth_thr']],on='date',how='left'); raw['breadth_high']=raw.breadth>=raw.breadth_thr

# Predeclared, scale-invariant factor families. Higher score = preferred long candidate.
raw['F_RS20']=raw.rs20_pct
raw['F_RS60']=raw.rs60_pct
raw['F_RS120']=raw.rs120_pct
raw['F_MULTI_RS']=(raw.rs20_pct+raw.rs60_pct+raw.rs120_pct)/3
raw['F_BREAKOUT_RS']=(raw.break120_pct+raw.rs60_pct+raw.volratio_pct)/3
raw['F_LOWVOL_RS']=(raw.rs60_pct+(1-raw.vol20_pct))/2
raw['F_PULLBACK_TREND']=(raw.rs60_pct+raw.pullback5_pct+(raw.trend120>0).astype(float))/3
raw['F_BREADTH_RS']=np.where(raw.breadth_high,(raw.rs20_pct+raw.rs60_pct)/2,np.nan)
FACTORS=['F_RS20','F_RS60','F_RS120','F_MULTI_RS','F_BREAKOUT_RS','F_LOWVOL_RS','F_PULLBACK_TREND','F_BREADTH_RS']

eligible=(raw.liq_pct>=.40)&raw.excess20.notna()&raw.ma120.notna()&raw.b_fwd20.notna()
base=raw.loc[eligible,['date','code','excess20','fwd20','b_fwd20']+FACTORS].copy()

rows=[]
for fac in FACTORS:
    z=base.dropna(subset=[fac]).copy()
    z['pct']=z.groupby('date')[fac].rank(pct=True)
    z=z[z.pct>=.80]
    for y in DISC_YEARS+OOS_YEARS+STRESS_YEARS:
        q=z[z.date.dt.year==y]
        daily=q.groupby('date').agg(excess=('excess20','mean'),stock_ret=('fwd20','mean'),bench=('b_fwd20','first'),n=('code','size')).reset_index()
        rows.append({'factor':fac,'year':y,'days':len(daily),'samples':len(q),'mean_excess20':float(daily.excess.mean()) if len(daily) else np.nan,'median_excess20':float(daily.excess.median()) if len(daily) else np.nan,'positive_alpha_days':float((daily.excess>0).mean()) if len(daily) else np.nan,'mean_stock20':float(daily.stock_ret.mean()) if len(daily) else np.nan,'mean_bench20':float(daily.bench.mean()) if len(daily) else np.nan})
res=pd.DataFrame(rows); res.to_csv(OUT/'stage13_factor_yearly.csv',index=False)

summary=[]
for fac in FACTORS:
    d=res[(res.factor==fac)&res.year.isin(DISC_YEARS)]
    o=res[(res.factor==fac)&res.year.isin(OOS_YEARS)]
    s=res[(res.factor==fac)&res.year.isin(STRESS_YEARS)]
    discovery_pass=bool(len(d)==5 and (d.mean_excess20>0).sum()>=4 and d.mean_excess20.mean()>0 and d.positive_alpha_days.mean()>.50 and d.samples.sum()>=500)
    oos_pass=bool(discovery_pass and len(o)==3 and (o.mean_excess20>0).all() and (o.positive_alpha_days>.50).sum()>=2 and o.samples.sum()>=300)
    stress_positive_years=int((s.mean_excess20>0).sum()) if len(s) else 0
    summary.append({'factor':fac,'discovery_mean_alpha20':float(d.mean_excess20.mean()),'discovery_positive_years':int((d.mean_excess20>0).sum()),'discovery_alpha_day_rate':float(d.positive_alpha_days.mean()),'discovery_samples':int(d.samples.sum()),'discovery_pass':discovery_pass,'oos_2013_2015_mean_alpha20':float(o.mean_excess20.mean()),'oos_positive_years':int((o.mean_excess20>0).sum()),'oos_pass':oos_pass,'stress_2016_2020_mean_alpha20':float(s.mean_excess20.mean()),'stress_positive_years':stress_positive_years})
sumdf=pd.DataFrame(summary).sort_values(['oos_pass','discovery_mean_alpha20'],ascending=[False,False]); sumdf.to_csv(OUT/'stage13_summary.csv',index=False)
survivors=sumdf.loc[sumdf.oos_pass,'factor'].tolist()
manifest={'stage':'13_long_history_alpha_discovery','historical_source_commit':'dc17eff999c4f4bd0a4ea7303b5d178b9100ec35','market_refetch':False,'formal_r10_modified':False,'uses_2021_2025':False,'warmup_year':2007,'discovery_years':DISC_YEARS,'oos_years':OOS_YEARS,'stress_years':STRESS_YEARS,'stress_used_for_selection':False,'benchmark':'0050','objective':'forward_20d_excess_return_vs_0050','scale_invariant_features':True,'factor_count':len(FACTORS),'survivors':survivors,'survivor_count':len(survivors)}
(OUT/'stage13_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('STAGE13_COMPLETE'); print(json.dumps(manifest,ensure_ascii=False,indent=2)); print('\nSUMMARY'); print(sumdf.to_string(index=False)); print('\nYEARLY'); print(res.to_string(index=False))
