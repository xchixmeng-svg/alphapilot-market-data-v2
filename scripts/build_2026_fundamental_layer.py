#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json, os, re, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'history'/'2026-fundamental'
OUT.mkdir(parents=True, exist_ok=True)
API='https://api.finmindtrade.com/api/v4/data'
TOKEN=os.environ.get('FINMIND_TOKEN','').strip()
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot-Fundamental/1.0','Accept':'application/json'})

def get_dataset(dataset,start,end,timeout=300):
    params={'dataset':dataset,'start_date':start,'end_date':end}
    if TOKEN: params['token']=TOKEN
    last=None
    for i in range(5):
        try:
            r=S.get(API,params=params,timeout=timeout); r.raise_for_status(); j=r.json()
            status=j.get('status')
            if status not in (200,None): raise RuntimeError(f"FinMind status={status} msg={j.get('msg')}")
            data=j.get('data')
            if not isinstance(data,list): raise RuntimeError(f'FinMind {dataset}: data is not list')
            if not data: raise RuntimeError(f'FinMind {dataset}: empty data')
            print(f'[DATA] {dataset} rows={len(data)}',flush=True)
            return data
        except Exception as e:
            last=e
            if i<4: time.sleep(min(20,2**i))
    raise RuntimeError(f'{dataset} failed: {last}')

def write_gz_csv(path, rows, fields=None):
    if not rows: raise RuntimeError(f'empty rows: {path}')
    if fields is None:
        seen=[]; ss=set()
        for r in rows:
            for k in r:
                if k not in ss: ss.add(k); seen.append(k)
        fields=seen
    with gzip.open(path,'wt',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    return fields

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def code_ok(v): return bool(re.fullmatch(r'\d{4}',str(v or '').strip()))

# We deliberately fetch enough history to calculate YoY revenue and trailing valuation without future leakage.
rev=get_dataset('TaiwanStockMonthRevenue','2024-01-01','2026-08-31')
fs=get_dataset('TaiwanStockFinancialStatements','2024-01-01','2026-08-31')
per=get_dataset('TaiwanStockPER','2026-01-01','2026-08-31')

# Preserve 4-digit stock universe only, as strings; no numeric conversion of stock codes.
rev=[r for r in rev if code_ok(r.get('stock_id'))]
fs=[r for r in fs if code_ok(r.get('stock_id'))]
per=[r for r in per if code_ok(r.get('stock_id'))]

p_rev=OUT/'month_revenue_2024_2026.csv.gz'; f_rev=write_gz_csv(p_rev,rev)
p_fs=OUT/'financial_statements_2024_2026.csv.gz'; f_fs=write_gz_csv(p_fs,fs)
p_per=OUT/'daily_valuation_2026.csv.gz'; f_per=write_gz_csv(p_per,per)

# EPS extract: retain any financial-statement row whose standardized or original label denotes EPS.
eps=[]
for r in fs:
    blob='|'.join(str(r.get(k,'')) for k in ('type','origin_name','name','statement'))
    low=blob.lower()
    if ('eps' in low) or ('每股盈餘' in blob) or ('每股盈余' in blob) or ('基本每股' in blob) or ('稀釋每股' in blob):
        eps.append(r)
p_eps=OUT/'eps_actual_2024_2026.csv.gz'
if eps: f_eps=write_gz_csv(p_eps,eps)
else:
    # Keep explicit audit instead of silently fabricating an EPS file.
    p_eps=None; f_eps=[]

# Revenue growth derived table. Use only rows as published in source; no future values are backfilled.
# We calculate YoY/MoM if fields are available directly from source, otherwise leave derivation for analysis layer.
rev_der=[]
by={}
for r in rev:
    sid=str(r.get('stock_id'))
    y=r.get('revenue_year'); m=r.get('revenue_month'); val=r.get('revenue')
    try: key=(sid,int(y),int(m)); v=float(val)
    except Exception: continue
    by[key]=v
for (sid,y,m),v in sorted(by.items()):
    py=by.get((sid,y-1,m)); pm=by.get((sid,y-1,12)) if m==1 else by.get((sid,y,m-1))
    rev_der.append({'stock_id':sid,'revenue_year':y,'revenue_month':m,'revenue':v,
                    'yoy_pct':None if py in (None,0) else (v/py-1)*100,
                    'mom_pct':None if pm in (None,0) else (v/pm-1)*100})
p_revd=OUT/'month_revenue_growth_2024_2026.csv.gz'; f_revd=write_gz_csv(p_revd,rev_der)

files=[]
for p in [p_rev,p_revd,p_fs,p_eps,p_per]:
    if p and p.exists(): files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)})
manifest={
 'dataset':'AlphaPilot 2026 Fundamental Layer',
 'generated_at_utc':datetime.now(timezone.utc).isoformat(),
 'source':'FinMind API v4 public datasets (token used only if repository secret exists)',
 'source_datasets':['TaiwanStockMonthRevenue','TaiwanStockFinancialStatements','TaiwanStockPER'],
 'coverage':{'revenue':'2024-01 through 2026-08 requested','financial_statements':'2024-01-01 through 2026-08-31 requested','valuation':'2026-01-01 through 2026-08-31 requested'},
 'rows':{'month_revenue':len(rev),'month_revenue_growth':len(rev_der),'financial_statements':len(fs),'eps_extract':len(eps),'daily_valuation':len(per)},
 'stock_code_policy':'stock_id preserved as string; only exact 4-digit codes retained; leading-zero ETFs/securities excluded from stock universe',
 'point_in_time_note':'Raw source dates are preserved. Backtests must apply publication/availability dates and must not use reports before they were public.',
 'eps_revision_note':'This package contains actual reported EPS/fundamentals, not historical analyst-consensus EPS revisions. AlphaPilot EPS Revision Proxy must be derived point-in-time from revenue/financial trends.',
 'files':files,
 'fields':{'month_revenue':f_rev,'financial_statements':f_fs,'eps_extract':f_eps,'daily_valuation':f_per}
}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
readme=OUT/'README.txt'
readme.write_text('AlphaPilot 2026 Fundamental Layer\n\nIncludes monthly revenue + YoY/MoM, reported financial statements/EPS extract, and daily PER/PBR valuation source rows.\nHistorical analyst consensus EPS revisions are NOT claimed; use the data to construct a point-in-time EPS Revision Proxy.\n',encoding='utf-8')
zip_path=ROOT/'AlphaPilot_2026_Fundamental_Layer.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname='2026-Fundamental/'+p.name)
print('[DONE]',zip_path,'bytes',zip_path.stat().st_size,flush=True)
print(json.dumps(manifest['rows'],ensure_ascii=False),flush=True)
