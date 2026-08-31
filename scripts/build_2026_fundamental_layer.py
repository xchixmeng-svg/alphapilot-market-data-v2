#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, io, json, re, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'history'/'2026-fundamental'; OUT.mkdir(parents=True,exist_ok=True)
HEAD={'User-Agent':'Mozilla/5.0 AlphaPilot-Fundamental/2.0','Accept':'application/json,text/html,text/csv,*/*'}

def get(url,params=None,timeout=90):
    last=None
    for i in range(5):
        try:
            r=requests.get(url,params=params,headers=HEAD,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(min(8,2**i))
    raise RuntimeError(f'GET failed {url}: {last}')

def code4(x):
    s=str(x or '').strip().replace('=','').replace('"','')
    return s if re.fullmatch(r'\d{4}',s) else None

def num(x):
    s=str(x or '').strip().replace(',','').replace('%','')
    if s in ('','-','--','N/A','nan','None'): return None
    try:return float(s)
    except:return None

def flatcols(df):
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=['|'.join(str(v) for v in c if str(v)!='nan') for c in df.columns]
    else: df.columns=[str(c) for c in df.columns]
    return df

def write_gz(path,rows,fields=None):
    if not rows: raise RuntimeError(f'empty {path}')
    if fields is None:
        fields=[]; seen=set()
        for r in rows:
            for k in r:
                if k not in seen: seen.add(k); fields.append(k)
    with gzip.open(path,'wt',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    return fields

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

# ---------- 1) Monthly revenue history: official MOPS static archive ----------
def fetch_month(y,m,market):
    roc=y-1911
    url=f'https://mops.twse.com.tw/nas/t21/{market}/t21sc03_{roc}_{m}_0.html'
    r=get(url,timeout=60); r.encoding='big5'
    tables=pd.read_html(io.StringIO(r.text))
    out=[]
    for t in tables:
        t=flatcols(t)
        cols=list(t.columns)
        code_col=next((c for c in cols if '公司代號' in c),None)
        name_col=next((c for c in cols if '公司名稱' in c),None)
        rev_col=next((c for c in cols if ('當月營收' in c and '去年' not in c)),None)
        prev_col=next((c for c in cols if '上月營收' in c),None)
        last_col=next((c for c in cols if '去年當月營收' in c),None)
        yoy_col=next((c for c in cols if '去年同月增減' in c or '去年同月增減%' in c),None)
        mom_col=next((c for c in cols if '上月比較增減' in c or '上月比較增減%' in c),None)
        if not code_col or not rev_col: continue
        for _,row in t.iterrows():
            c=code4(row.get(code_col))
            if not c: continue
            out.append({'year':y,'month':m,'market':'TWSE' if market=='sii' else 'TPEX','code':c,
                        'name':str(row.get(name_col,'')) if name_col else '',
                        'revenue_thousand_n':num(row.get(rev_col)),
                        'prev_month_revenue_thousand_n':num(row.get(prev_col)) if prev_col else None,
                        'last_year_month_revenue_thousand_n':num(row.get(last_col)) if last_col else None,
                        'mops_mom_pct':num(row.get(mom_col)) if mom_col else None,
                        'mops_yoy_pct':num(row.get(yoy_col)) if yoy_col else None,
                        'source_url':url})
    return out

rev=[]; rev_fail=[]
for y in (2024,2025,2026):
    maxm=12 if y<2026 else 7  # Aug revenue is not fully reported on Aug 31.
    for m in range(1,maxm+1):
        for market in ('sii','otc'):
            try:
                rows=fetch_month(y,m,market); rev.extend(rows)
                print('[REV]',y,m,market,len(rows),flush=True)
            except Exception as e:
                rev_fail.append({'year':y,'month':m,'market':market,'error':str(e)}); print('[REV FAIL]',y,m,market,e,flush=True)
# dedupe authoritative key
rev=list({(r['year'],r['month'],r['market'],r['code']):r for r in rev}.values())
# derive clean YoY/MoM from raw revenue
idx={(r['code'],r['year'],r['month']):r['revenue_thousand_n'] for r in rev if r['revenue_thousand_n'] is not None}
for r in rev:
    v=r['revenue_thousand_n']; y=r['year']; m=r['month']; c=r['code']
    py=idx.get((c,y-1,m)); pm=idx.get((c,y-1,12)) if m==1 else idx.get((c,y,m-1))
    r['yoy_pct']=None if v is None or py in (None,0) else (v/py-1)*100
    r['mom_pct']=None if v is None or pm in (None,0) else (v/pm-1)*100
rev=sorted(rev,key=lambda r:(r['year'],r['month'],r['market'],r['code']))
p_rev=OUT/'monthly_revenue_2024_2026.csv.gz'; write_gz(p_rev,rev)

# ---------- 2) Latest reported EPS / income statement snapshot: official MOPS open data ----------
def fetch_eps_csv(suffix):
    url=f'https://mopsfin.twse.com.tw/opendata/t187ap14_{suffix}.csv'
    r=get(url,timeout=90)
    text=r.content.decode('utf-8-sig','ignore')
    rows=[]
    for x in csv.DictReader(io.StringIO(text)):
        c=code4(x.get('公司代號'))
        if not c: continue
        rows.append({'market':'TWSE' if suffix=='L' else 'TPEX','report_date':x.get('出表日期'),'year_roc':x.get('年度'),'quarter':x.get('季別'),
                     'code':c,'name':x.get('公司名稱',''),'industry':x.get('產業別',''),'eps_basic':num(x.get('基本每股盈餘(元)')),
                     'revenue':num(x.get('營業收入')),'operating_income':num(x.get('營業利益')),'non_operating':num(x.get('營業外收入及支出')),'net_income':num(x.get('稅後淨利')),
                     'source_url':url})
    if not rows: raise RuntimeError(f'empty EPS {suffix}')
    return rows

eps=[]; eps_fail=[]
for suffix in ('L','O'):
    try:
        rows=fetch_eps_csv(suffix); eps.extend(rows); print('[EPS]',suffix,len(rows),flush=True)
    except Exception as e:
        eps_fail.append({'market':suffix,'error':str(e)}); print('[EPS FAIL]',suffix,e,flush=True)
eps=sorted(eps,key=lambda r:(r['market'],r['code']))
p_eps=OUT/'eps_latest_2026.csv.gz'; write_gz(p_eps,eps)

# ---------- 3) Daily P/E, P/B, dividend yield history 2026: official TWSE + TPEx ----------
def weekdays(start,end):
    d=start
    while d<=end:
        if d.weekday()<5: yield d
        d+=timedelta(days=1)

def twse_val(d):
    ds=d.strftime('%Y%m%d')
    j=get('https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d',{'response':'json','date':ds,'selectType':'ALL'},timeout=60).json()
    fields=j.get('fields') or []; data=j.get('data') or []
    out=[]
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(r.get('證券代號'))
        if not c: continue
        out.append({'date':d.isoformat(),'market':'TWSE','code':c,'name':r.get('證券名稱',''),
                    'pe':num(r.get('本益比')),'pb':num(r.get('股價淨值比')),'dividend_yield_pct':num(r.get('殖利率(%)'))})
    return out

def tpex_val(d):
    roc=f'{d.year-1911}/{d.month:02d}/{d.day:02d}'
    j=get('https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php',{'l':'zh-tw','o':'json','d':roc,'c':''},timeout=60).json()
    data=j.get('aaData') or (j.get('tables',[{}])[0].get('data') if j.get('tables') else []) or []
    out=[]
    for row in data:
        if isinstance(row,dict):
            c=code4(row.get('SecuritiesCompanyCode') or row.get('股票代號') or row.get('代號'))
            if not c: continue
            out.append({'date':d.isoformat(),'market':'TPEX','code':c,'name':row.get('CompanyName') or row.get('名稱') or '',
                        'pe':num(row.get('PriceEarningRatio') or row.get('本益比')),'pb':num(row.get('PriceBookRatio') or row.get('股價淨值比')),
                        'dividend_yield_pct':num(row.get('DividendYield') or row.get('殖利率(%)'))})
        elif isinstance(row,list) and len(row)>=7:
            c=code4(row[0])
            if c: out.append({'date':d.isoformat(),'market':'TPEX','code':c,'name':str(row[1]),'pe':num(row[2]),'pb':num(row[6]),'dividend_yield_pct':num(row[5])})
    return out

dates=list(weekdays(date(2026,1,1),date(2026,8,28)))
vals=[]; val_fail=[]
def task(m,d): return m,d,(twse_val(d) if m=='TWSE' else tpex_val(d))
with ThreadPoolExecutor(max_workers=6) as ex:
    fut=[ex.submit(task,m,d) for d in dates for m in ('TWSE','TPEX')]
    for i,f in enumerate(as_completed(fut),1):
        try:
            m,d,rows=f.result()
            if rows: vals.extend(rows)
        except Exception as e:
            val_fail.append({'error':str(e)})
        if i%40==0 or i==len(fut): print('[VAL]',i,'/',len(fut),'rows',len(vals),'fails',len(val_fail),flush=True)
vals=list({(r['date'],r['market'],r['code']):r for r in vals}.values()); vals=sorted(vals,key=lambda r:(r['date'],r['market'],r['code']))
p_val=OUT/'daily_valuation_2026.csv.gz'; write_gz(p_val,vals)

# Audit coverage and package.
coverage={'revenue_rows':len(rev),'revenue_months':len({(r['year'],r['month']) for r in rev}),'eps_rows':len(eps),
          'eps_markets':sorted({r['market'] for r in eps}),'valuation_rows':len(vals),
          'valuation_dates_twse':len({r['date'] for r in vals if r['market']=='TWSE'}),'valuation_dates_tpex':len({r['date'] for r in vals if r['market']=='TPEX'})}
if len(rev)<20000: raise RuntimeError(f'revenue coverage too low: {coverage}')
if len(eps)<1000: raise RuntimeError(f'EPS coverage too low: {coverage}, failures={eps_fail}')
if coverage['valuation_dates_twse']<140 or coverage['valuation_dates_tpex']<140: raise RuntimeError(f'valuation date coverage too low: {coverage}')
files=[]
for p in (p_rev,p_eps,p_val): files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)})
manifest={'dataset':'AlphaPilot 2026 Fundamental Layer','generated_at_utc':datetime.now(timezone.utc).isoformat(),'sources':{
 'monthly_revenue':'Official MOPS static monthly revenue archive (listed + OTC)','eps':'Official MOPS financial open data t187ap14_L/O','valuation':'Official TWSE BWIBBU_d + TPEx peratio_analysis'},
 'coverage':coverage,'failures':{'revenue':rev_fail,'eps':eps_fail,'valuation_count':len(val_fail)},
 'point_in_time_note':'Revenue rows are monthly historical publications. EPS file is the latest official reported-quarter snapshot as of build time. Daily valuation rows preserve historical dates. For backtests, never expose latest EPS snapshot to dates before its report date.',
 'analyst_revision_note':'This is actual public fundamental/valuation data, not paid historical analyst-consensus EPS revisions. Use monthly revenue acceleration + reported EPS + historical P/E/P/B to construct the AlphaPilot EPS Revision / Re-rating proxy.',
 'stock_code_policy':'4-digit stock codes retained as strings; leading-zero ETFs excluded.','files':files}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.txt').write_text('AlphaPilot 2026 Fundamental Layer\nOfficial monthly revenue history, latest reported EPS snapshot, and 2026 daily P/E/P/B/dividend-yield history.\n',encoding='utf-8')
zip_path=ROOT/'AlphaPilot_2026_Fundamental_Layer.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname='2026-Fundamental/'+p.name)
print('[DONE]',zip_path,'bytes',zip_path.stat().st_size,flush=True)
print(json.dumps(coverage,ensure_ascii=False),flush=True)
