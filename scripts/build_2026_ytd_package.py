#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json, re, time, zipfile, hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'history'/'2026-YTD'; OUT.mkdir(parents=True,exist_ok=True)
HEAD={'User-Agent':'Mozilla/5.0 AlphaPilot/2026YTD','Accept':'application/json,text/plain,*/*'}

def get(url,params=None,timeout=90):
    last=None
    for i in range(5):
        try:
            r=requests.get(url,params=params,headers=HEAD,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(min(4,2**i))
    raise RuntimeError(f'GET failed {url}: {last}')

def n(x):
    if x is None:return None
    s=str(x).strip().replace(',','').replace('+','')
    if s in ('','--','---','null','None'):return None
    try:return float(s)
    except:return None

def code4(x):
    s=str(x or '').strip().strip('=').strip('"'); return s if re.fullmatch(r'\d{4}',s) else None

def parse_date(v):
    s=re.sub(r'[^0-9]','',str(v or ''))
    try:
        if len(s)==8:return datetime.strptime(s,'%Y%m%d').date()
        if len(s)==7:return datetime.strptime(str(int(s[:3])+1911)+s[3:],'%Y%m%d').date()
    except: pass
    return None

def write_csv(path,rows,fields=None):
    if not rows: raise RuntimeError(f'empty {path}')
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

# OHLCV from 2026 weekly GitHub release assets.
rel=get('https://api.github.com/repos/yukishirotsubasa/tw-stock-data-release/releases/tags/daily-close-csv').json()
assets={a['name']:a for a in rel.get('assets',[]) if a['name'].startswith('weekly_2026_W') and a['name'].endswith('.zip')}
if not assets: raise RuntimeError('no 2026 weekly assets')
ohlcv={}; used=[]
for name in sorted(assets):
    a=assets[name]; blob=get(a['browser_download_url'],timeout=180).content
    digest=hashlib.sha256(blob).hexdigest(); exp=(a.get('digest') or '').replace('sha256:','')
    if exp and digest!=exp: raise RuntimeError(f'{name} sha mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            if not member.lower().endswith('.csv'):continue
            text=io.TextIOWrapper(z.open(member),encoding='utf-8-sig',newline='')
            for r in csv.DictReader(text):
                d=parse_date(r.get('date')); c=code4(r.get('code'))
                if not d or d.year!=2026 or not c:continue
                row={'date':d.isoformat(),'code':c,'name':r.get('name',''),'volume':r.get('volume'),'open':r.get('open'),'high':r.get('high'),'low':r.get('low'),'close':r.get('close')}
                ohlcv[(row['date'],c)]=row
    used.append({'asset':name,'sha256':digest,'bytes':len(blob)})
    print('[OHLCV]',name,'rows_total',len(ohlcv),flush=True)
rows_ohlcv=sorted(ohlcv.values(),key=lambda r:(r['date'],r['code']))
trade_dates=sorted({r['date'] for r in rows_ohlcv})
write_csv(OUT/'ohlcv_2026_ytd.csv',rows_ohlcv,['date','code','name','volume','open','high','low','close'])

# Official TWSE T86.
def twse_t86(ds):
    j=get('https://www.twse.com.tw/rwd/zh/fund/T86',{'response':'json','date':ds.replace('-',''),'selectType':'ALLBUT0999'},timeout=60).json()
    fields=j.get('fields') or []; data=j.get('data') or []; out=[]
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(r.get('證券代號'))
        if not c:continue
        out.append({'date':ds,'market':'TWSE','code':c,'name':r.get('證券名稱',''),
          'foreign_buy':n(r.get('外陸資買進股數(不含外資自營商)')),'foreign_sell':n(r.get('外陸資賣出股數(不含外資自營商)')),'foreign_net':n(r.get('外陸資買賣超股數(不含外資自營商)')),
          'trust_buy':n(r.get('投信買進股數')),'trust_sell':n(r.get('投信賣出股數')),'trust_net':n(r.get('投信買賣超股數')),
          'dealer_buy':n(r.get('自營商買進股數(自行買賣)')),'dealer_sell':n(r.get('自營商賣出股數(自行買賣)')),'dealer_net':n(r.get('自營商買賣超股數(自行買賣)')),
          'total_net':n(r.get('三大法人買賣超股數'))})
    if not out: raise RuntimeError(f'TWSE T86 empty {ds}')
    return out

def roc(ds):
    d=datetime.strptime(ds,'%Y-%m-%d').date(); return f'{d.year-1911:03d}/{d.month:02d}/{d.day:02d}'

# Official TPEx new dailyTrade JSON API (old 3itrade endpoint retired in 2025/12).
def tpex_daily(ds):
    params={'type':'Daily','sect':'EW','date':roc(ds),'id':'','response':'json'}
    j=get('https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade',params,timeout=60).json()
    tables=j.get('tables') or []
    if not tables: raise RuntimeError(f'TPEX empty tables {ds}')
    rows=tables[0].get('data') or []
    if not rows: raise RuntimeError(f'TPEX empty data {ds}')
    out=[]
    for r in rows:
        if not isinstance(r,list) or len(r)<24: continue
        c=code4(r[0])
        if not c:continue
        out.append({'date':ds,'market':'TPEX','code':c,'name':str(r[1]).strip(),
          'foreign_buy':n(r[8]),'foreign_sell':n(r[9]),'foreign_net':n(r[10]),
          'trust_buy':n(r[11]),'trust_sell':n(r[12]),'trust_net':n(r[13]),
          'dealer_buy':n(r[20]),'dealer_sell':n(r[21]),'dealer_net':n(r[22]),'total_net':n(r[23])})
    if not out: raise RuntimeError(f'TPEX no parsed rows {ds}')
    return out

def fetch_all(fn,market):
    rows=[]; fails=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(fn,ds):ds for ds in trade_dates}
        for i,f in enumerate(as_completed(fut),1):
            ds=fut[f]
            try: rows.extend(f.result())
            except Exception as e: fails.append({'date':ds,'market':market,'error':str(e)})
            if i%20==0 or i==len(fut): print(f'[{market} INST]',i,'/',len(fut),'rows',len(rows),'failures',len(fails),flush=True)
    return rows,fails

twse,f1=fetch_all(twse_t86,'TWSE')
tpex,f2=fetch_all(tpex_daily,'TPEX')
failures=f1+f2
inst=twse+tpex
cov={m:len({r['date'] for r in inst if r['market']==m})/len(trade_dates) for m in ('TWSE','TPEX')}
if min(cov.values())<0.95: raise RuntimeError(f'institutional coverage too low {cov}; failures={len(failures)}')
inst=sorted(inst,key=lambda r:(r['date'],r['market'],r['code']))
write_csv(OUT/'institutional_2026_ytd.csv',inst)
if failures:(OUT/'institutional_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
manifest={'dataset':'AlphaPilot 2026 YTD Taiwan market package','generated_at_utc':datetime.utcnow().isoformat()+'Z','coverage':{'start':trade_dates[0],'end':trade_dates[-1],'trading_days':len(trade_dates),'ohlcv_rows':len(rows_ohlcv),'institutional_rows':len(inst),'twse_institutional_rows':len(twse),'tpex_institutional_rows':len(tpex),'institutional_market_date_coverage':cov},'ohlcv_source':'GitHub release yukishirotsubasa/tw-stock-data-release (TWSE MI_INDEX + TPEx daily close)','institutional_source':'Official TWSE T86 + official TPEx /www/zh-tw/insti/dailyTrade','weekly_assets':used,'failures':failures}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
zip_path=ROOT/'AlphaPilot_2026_YTD_Data_Package.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname='2026-YTD/'+p.name)
print('[DONE]',zip_path,'bytes',zip_path.stat().st_size,flush=True)
print(json.dumps(manifest['coverage'],ensure_ascii=False),flush=True)
