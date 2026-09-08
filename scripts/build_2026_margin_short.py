#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, random, re, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent.parent
YTD=ROOT/'data'/'history'/'2026-YTD'
OUT=YTD/'official_margin_short_2026_ytd.csv.gz'
HEAD={'User-Agent':'Mozilla/5.0 AlphaPilot-Sentinel/1.0','Accept':'application/json,text/plain,*/*'}
TWSE=('https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN','https://www.twse.com.tw/exchangeReport/MI_MARGN')
TPEX='https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php'

def num(v):
    s=str(v or '').strip().replace(',','').replace('+','')
    try:return float(s)
    except:return None
def code4(v):
    s=str(v or '').strip(); return s if re.fullmatch(r'\d{4}',s) else None
def req(url,params,attempts=7):
    last=None
    for i in range(attempts):
        try:
            r=requests.get(url,params=params,headers=HEAD,timeout=60); r.raise_for_status(); j=r.json()
            if j and '查詢過於頻繁' not in str(j.get('stat','')): return j
        except Exception as e:last=e
        time.sleep(min(60,3*2**min(i,4))+random.random())
    raise RuntimeError(f'{url}: {last}')
def twse(ds):
    errs=[]
    for url in TWSE:
        try:
            j=req(url,{'date':ds.replace('-',''),'selectType':'ALL','response':'json'},5)
            t=next((x for x in j.get('tables',[]) if '融資融券彙總' in str(x.get('title',''))),None)
            out=[]
            for r in (t or {}).get('data',[]):
                c=code4(r[0] if r else None)
                if c and len(r)>=14: out.append({'date':ds,'market':'TWSE','code':c,'name':str(r[1]).strip(),'margin_buy':num(r[2]),'margin_sell':num(r[3]),'margin_repay':num(r[4]),'margin_prev':num(r[5]),'margin_balance':num(r[6]),'margin_limit':num(r[7]),'short_sell':num(r[8]),'short_buy':num(r[9]),'short_repay':num(r[10]),'short_prev':num(r[11]),'short_balance':num(r[12]),'short_limit':num(r[13])})
            if out:return out
        except Exception as e:errs.append(str(e))
    raise RuntimeError(' | '.join(errs) or 'empty TWSE')
def tpex(ds):
    d=datetime.strptime(ds,'%Y-%m-%d'); roc=f'{d.year-1911:03d}/{d.month:02d}/{d.day:02d}'
    j=req(TPEX,{'l':'zh-tw','o':'json','d':roc,'s':'0,asc'},6); out=[]
    for r in ((j.get('tables') or [{}])[0].get('data') or []):
        c=code4(r[0] if r else None)
        if c and len(r)>=19: out.append({'date':ds,'market':'TPEX','code':c,'name':str(r[1]).strip(),'margin_prev':num(r[2]),'margin_buy':num(r[3]),'margin_sell':num(r[4]),'margin_repay':num(r[5]),'margin_balance':num(r[6]),'margin_usage_pct':num(r[8]),'margin_limit':num(r[9]),'short_prev':num(r[10]),'short_sell':num(r[11]),'short_buy':num(r[12]),'short_repay':num(r[13]),'short_balance':num(r[14]),'short_usage_pct':num(r[16]),'short_limit':num(r[17])})
    if not out:raise RuntimeError('empty TPEX')
    return out

dates=sorted(pd.read_csv(YTD/'ohlcv_2026_ytd.csv',usecols=['date'])['date'].drop_duplicates())
rows=[]; failures=[]
for market,fn,pause in [('TPEX',tpex,.25),('TWSE',twse,1.8)]:
    for i,ds in enumerate(dates,1):
        try:rows.extend(fn(ds))
        except Exception as e:failures.append({'date':ds,'market':market,'error':str(e)})
        time.sleep(pause+random.random()*.4)
        if i%25==0 or i==len(dates):print(f'[{market}] {i}/{len(dates)}',flush=True)
rows.sort(key=lambda r:(r['date'],r['market'],r['code']))
pd.DataFrame(rows).to_csv(OUT,index=False,compression='gzip')
coverage={m:{'expected':len(dates),'present':len({r['date'] for r in rows if r['market']==m})} for m in ('TWSE','TPEX')}
for v in coverage.values():v['ratio']=v['present']/v['expected']
manifest={'dataset':'AlphaPilot official margin/short 2026 YTD','generated_at_utc':datetime.now(timezone.utc).isoformat(),'date_start':dates[0],'date_end':dates[-1],'rows':len(rows),'coverage':coverage,'failures':failures,'sources':{'TWSE':TWSE,'TPEX':TPEX},'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}
(YTD/'margin_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(coverage,ensure_ascii=False),flush=True)
if min(x['ratio'] for x in coverage.values())<.98:raise RuntimeError('margin coverage below 98%')
