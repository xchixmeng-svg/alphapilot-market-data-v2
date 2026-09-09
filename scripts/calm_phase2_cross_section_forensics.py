#!/usr/bin/env python3
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
RAW=ROOT/'input'/'raw'
HIST=ROOT/'data'/'history'/'2020-2025'
OUT=ROOT/'calm_phase2_output'; OUT.mkdir(exist_ok=True)

def load_year(y):
    if y>=2020:
        p=HIST/f'ohlcv_{y}.parquet'; d=pd.read_parquet(p)
    else:
        zp=RAW/f'yearly_{y}.zip'
        if not zp.exists(): raise FileNotFoundError(zp)
        with zipfile.ZipFile(zp) as z:
            member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
            with z.open(member) as f: d=pd.read_csv(f,dtype={'code':str},low_memory=False)
    d.columns=[str(c).lower().strip() for c in d.columns]
    d['code']=d['code'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)
    if pd.api.types.is_datetime64_any_dtype(d['date']): d['date']=d['date'].dt.strftime('%Y%m%d').astype(int)
    else: d['date']=pd.to_numeric(d['date'].astype(str).str.replace('-','',regex=False),errors='coerce').astype('Int64')
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['date','code','close','volume']).assign(date=lambda x:x.date.astype(int))

df=pd.concat([load_year(y) for y in range(2015,2026)],ignore_index=True)
df=df.sort_values(['code','date']).drop_duplicates(['code','date'],keep='last').reset_index(drop=True)
valid=df.code.str.fullmatch(r'[1-9]\d{3}') & ~df.get('name',pd.Series('',index=df.index)).astype(str).str.contains('KY',case=False,na=False)
df=df[valid].copy(); g=df.groupby('code',group_keys=False)
df['amt']=df.close*df.volume
df['amt20']=g.amt.transform(lambda s:s.rolling(20,min_periods=20).mean())
df['ma60']=g.close.transform(lambda s:s.rolling(60,min_periods=60).mean())
df['r20']=g.close.transform(lambda s:s.pct_change(20)); df['r60']=g.close.transform(lambda s:s.pct_change(60))
df['prior60']=g.close.transform(lambda s:s.shift(1).rolling(60,min_periods=60).max())
df['fwd20']=g.close.transform(lambda s:s.shift(-20)/s-1)
df['fwd10']=g.close.transform(lambda s:s.shift(-10)/s-1)
df['liquid']=df.amt20>=30_000_000
df['above60']=df.liquid & (df.close>df.ma60)
df['mom20pos']=df.liquid & (df.r20>0)
df['break60']=df.liquid & (df.close>df.prior60)
df['break20_success']=df.break60 & (df.fwd20>0)

rows=[]
for date,x in df.groupby('date'):
    q=x[x.liquid]
    if len(q)<50: continue
    br=q[q.break60]
    top=q.nlargest(max(1,int(len(q)*0.10)),'r20')
    rows.append({'date':date,'year':date//10000,'n_liquid':len(q),'breadth_ma60':q.above60.mean(),'breadth_r20_pos':q.mom20pos.mean(),
                 'median_r20':q.r20.median(),'median_r60':q.r60.median(),'cross_section_r20_std':q.r20.std(),
                 'break60_rate':q.break60.mean(),'break60_fwd20_mean':br.fwd20.mean() if len(br) else np.nan,
                 'break60_fwd20_win':(br.fwd20>0).mean() if len(br) else np.nan,'top10pct_r20_fwd10_mean':top.fwd10.mean(),
                 'top10pct_r20_fwd20_mean':top.fwd20.mean()})
panel=pd.DataFrame(rows); panel.to_csv(OUT/'daily_cross_section_panel.csv',index=False)
annual=panel.groupby('year').agg(days=('date','count'),breadth_ma60=('breadth_ma60','mean'),breadth_r20_pos=('breadth_r20_pos','mean'),median_r20=('median_r20','mean'),cross_section_r20_std=('cross_section_r20_std','mean'),break60_rate=('break60_rate','mean'),break60_fwd20_mean=('break60_fwd20_mean','mean'),break60_fwd20_win=('break60_fwd20_win','mean'),top_mom_fwd10=('top10pct_r20_fwd10_mean','mean'),top_mom_fwd20=('top10pct_r20_fwd20_mean','mean')).reset_index()
annual.to_csv(OUT/'annual_cross_section.csv',index=False)

def era(a,b):
    z=panel[(panel.year>=a)&(panel.year<=b)]
    return {k:float(z[k].mean()) for k in ['breadth_ma60','breadth_r20_pos','median_r20','cross_section_r20_std','break60_rate','break60_fwd20_mean','break60_fwd20_win','top10pct_r20_fwd10_mean','top10pct_r20_fwd20_mean']}
summary={'design':'diagnostic_only_no_threshold_selection','warmup':2015,'calm_hypothesis_years':[2016,2017,2018],'bridge_years':[2019,2020],'r10_baseline_years':[2021,2022,2023,2024,2025], 'era_2016_2018':era(2016,2018),'era_2019_2020':era(2019,2020),'era_2021_2025':era(2021,2025)}
(OUT/'phase2_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('PHASE2_COMPLETE')
print(annual.to_string(index=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
