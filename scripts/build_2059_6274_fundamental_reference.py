#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json, os, re, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'reference'/'2059-6274-fundamental'; OUT.mkdir(parents=True,exist_ok=True)
API='https://api.finmindtrade.com/api/v4/data'; TOKEN=os.environ.get('FINMIND_TOKEN','').strip()
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot-Reference/1.0','Accept':'application/json'})

def fetch(dataset,stock,start,end):
    p={'dataset':dataset,'data_id':stock,'start_date':start,'end_date':end}
    if TOKEN:p['token']=TOKEN
    last=None
    for i in range(4):
        try:
            r=S.get(API,params=p,timeout=90); r.raise_for_status(); j=r.json()
            if j.get('status') not in (200,None): raise RuntimeError(f"{j.get('status')} {j.get('msg')}")
            d=j.get('data');
            if not isinstance(d,list) or not d: raise RuntimeError('empty data')
            print('[DATA]',dataset,stock,len(d),flush=True); return d
        except Exception as e:
            last=e; time.sleep(1+i)
    raise RuntimeError(f'{dataset} {stock}: {last}')

def write_gz(path,rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:seen.add(k);fields.append(k)
    with gzip.open(path,'wt',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

all_rows={}; errors=[]
for stock in ('2059','6274'):
    for ds,start,end in [
        ('TaiwanStockMonthRevenue','2024-01-01','2026-08-31'),
        ('TaiwanStockFinancialStatements','2024-01-01','2026-08-31'),
        ('TaiwanStockPER','2026-01-01','2026-08-31')]:
        try: all_rows[(stock,ds)]=fetch(ds,stock,start,end)
        except Exception as e: errors.append({'stock':stock,'dataset':ds,'error':str(e)})

# require all 6 queries; this package must not silently omit a layer
if errors: raise RuntimeError('reference fetch incomplete: '+json.dumps(errors,ensure_ascii=False))
files=[]
for (stock,ds),rows in all_rows.items():
    short={'TaiwanStockMonthRevenue':'month_revenue','TaiwanStockFinancialStatements':'financial_statements','TaiwanStockPER':'valuation'}[ds]
    p=OUT/f'{stock}_{short}.csv.gz'; write_gz(p,rows); files.append({'file':p.name,'rows':len(rows),'bytes':p.stat().st_size,'sha256':sha(p)})
# extract EPS rows
for stock in ('2059','6274'):
    src=all_rows[(stock,'TaiwanStockFinancialStatements')]; eps=[]
    for r in src:
        txt='|'.join(str(r.get(k,'')) for k in ('type','origin_name','name')).lower()
        if 'eps' in txt or '每股盈餘' in txt or '每股盈余' in txt or '基本每股' in txt: eps.append(r)
    if eps:
        p=OUT/f'{stock}_eps_extract.csv.gz';write_gz(p,eps);files.append({'file':p.name,'rows':len(eps),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'dataset':'AlphaPilot 2059 川湖 + 6274 台燿 fundamental reference','generated_at_utc':datetime.now(timezone.utc).isoformat(),
 'coverage':{'monthly_revenue':'2024-01 to 2026-08 requested','financial_statements':'2024-01 to 2026-08 requested','valuation':'2026-01 to 2026-08 requested'},
 'purpose':'Reconstruct the March-April 2026 signal using point-in-time revenue/EPS/valuation evidence.',
 'analyst_revision_note':'Actual reported fundamentals and valuation only; not paid historical analyst consensus revisions.','files':files}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
zip_path=ROOT/'AlphaPilot_2059_6274_Fundamental_Reference.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname='2059-6274-Fundamental/'+p.name)
print('[DONE]',zip_path,zip_path.stat().st_size,flush=True)
