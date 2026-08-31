#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json, re, time, zipfile, hashlib
from datetime import date, datetime
from pathlib import Path
import requests

YEAR=2026
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'history'/'2026-YTD'
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot/2026YTD','Accept':'application/json,text/plain,*/*'})

def get(url,params=None,timeout=90):
    last=None
    for i in range(5):
        try:
            r=S.get(url,params=params,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(min(8,2**i))
    raise RuntimeError(f'GET failed {url}: {last}')

def n(x):
    if x is None: return None
    s=str(x).strip().replace(',','').replace('+','')
    if s in ('','--','---','null','None'): return None
    try:return float(s)
    except:return None

def code4(x):
    s=str(x or '').strip()
    return s if re.fullmatch(r'\d{4}',s) else None

def write_csv(path,rows,fields=None):
    if not rows: raise RuntimeError(f'empty {path}')
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def parse_any_date(v):
    s=re.sub(r'[^0-9]','',str(v or ''))
    if len(s)==8:
        try:return datetime.strptime(s,'%Y%m%d').date()
        except: pass
    if len(s)==7:
        try:return datetime.strptime(str(int(s[:3])+1911)+s[3:],'%Y%m%d').date()
        except: pass
    return None

# 1) OHLCV: GitHub Release weekly files, latest available through current week.
rel=get('https://api.github.com/repos/yukishirotsubasa/tw-stock-data-release/releases/tags/daily-close-csv').json()
assets={a['name']:a for a in rel.get('assets',[]) if a['name'].startswith('weekly_2026_W') and a['name'].endswith('.zip')}
if not assets: raise RuntimeError('no 2026 weekly assets')
ohlcv={}; used=[]
for name in sorted(assets):
    a=assets[name]; blob=get(a['browser_download_url'],timeout=180).content
    digest=hashlib.sha256(blob).hexdigest()
    exp=(a.get('digest') or '').replace('sha256:','')
    if exp and digest!=exp: raise RuntimeError(f'{name} sha mismatch')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            if not member.lower().endswith('.csv'): continue
            text=io.TextIOWrapper(z.open(member),encoding='utf-8-sig',newline='')
            for r in csv.DictReader(text):
                d=parse_any_date(r.get('date')); c=code4(r.get('code'))
                if not d or d.year!=2026 or not c: continue
                row={'date':d.isoformat(),'code':c,'name':r.get('name',''),'volume':r.get('volume'),'open':r.get('open'),'high':r.get('high'),'low':r.get('low'),'close':r.get('close')}
                ohlcv[(row['date'],c)]=row
    used.append({'asset':name,'sha256':digest,'bytes':len(blob)})
    print('[OHLCV]',name,'rows_total',len(ohlcv),flush=True)
rows_ohlcv=sorted(ohlcv.values(),key=lambda r:(r['date'],r['code']))
write_csv(OUT/'ohlcv_2026_ytd.csv',rows_ohlcv,['date','code','name','volume','open','high','low','close'])
trade_dates=sorted({r['date'] for r in rows_ohlcv})

# Generic TWSE T86 parser
def twse_t86(ds):
    p={'response':'json','date':ds.replace('-',''),'selectType':'ALLBUT0999'}
    j=get('https://www.twse.com.tw/rwd/zh/fund/T86',p).json()
    fields=j.get('fields') or [] ; data=j.get('data') or []
    out=[]
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(r.get('證券代號'))
        if not c: continue
        out.append({'date':ds,'market':'TWSE','code':c,'name':r.get('證券名稱',''),
          'foreign_buy':n(r.get('外陸資買進股數(不含外資自營商)')),'foreign_sell':n(r.get('外陸資賣出股數(不含外資自營商)')),'foreign_net':n(r.get('外陸資買賣超股數(不含外資自營商)')),
          'trust_buy':n(r.get('投信買進股數')),'trust_sell':n(r.get('投信賣出股數')),'trust_net':n(r.get('投信買賣超股數')),
          'dealer_buy':n(r.get('自營商買進股數(自行買賣)')),'dealer_sell':n(r.get('自營商賣出股數(自行買賣)')),'dealer_net':n(r.get('自營商買賣超股數(自行買賣)')),
          'total_net':n(r.get('三大法人買賣超股數'))})
    return out

# TPEx historical official JSON; generic field matching for current site schema.
def tpex_hist(ds):
    j=get('https://www.tpex.org.tw/www/zh-tw/insti/daily',{'date':ds.replace('-','/'),'id':'','response':'json'}).json()
    tables=[]
    def walk(x):
        if isinstance(x,dict):
            if isinstance(x.get('fields'),list) and isinstance(x.get('data'),list): tables.append((x['fields'],x['data']))
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(j)
    if not tables: raise RuntimeError(f'TPEx no table {ds}')
    fields,data=max(tables,key=lambda t:len(t[1]))
    out=[]
    def pick(r,*tokens):
        for k,v in r.items():
            sk=re.sub(r'\s+','',str(k))
            if all(t in sk for t in tokens): return v
        return None
    for vals in data:
        r=dict(zip(fields,vals)); c=code4(pick(r,'代號'))
        if not c: continue
        out.append({'date':ds,'market':'TPEX','code':c,'name':pick(r,'名稱') or '',
          'foreign_buy':n(pick(r,'外資','買進')),'foreign_sell':n(pick(r,'外資','賣出')),'foreign_net':n(pick(r,'外資','買賣超')),
          'trust_buy':n(pick(r,'投信','買進')),'trust_sell':n(pick(r,'投信','賣出')),'trust_net':n(pick(r,'投信','買賣超')),
          'dealer_buy':n(pick(r,'自營商','買進')),'dealer_sell':n(pick(r,'自營商','賣出')),'dealer_net':n(pick(r,'自營商','買賣超')),
          'total_net':n(pick(r,'合計','買賣超') or pick(r,'三大法人','買賣超'))})
    return out

inst=[]; failures=[]
for idx,ds in enumerate(trade_dates,1):
    day=[]
    try: day += twse_t86(ds)
    except Exception as e: failures.append({'date':ds,'market':'TWSE','error':str(e)})
    time.sleep(0.08)
    try: day += tpex_hist(ds)
    except Exception as e: failures.append({'date':ds,'market':'TPEX','error':str(e)})
    inst.extend(day)
    if idx%20==0 or idx==len(trade_dates): print('[INST]',idx,'/',len(trade_dates),'rows',len(inst),'failures',len(failures),flush=True)
    time.sleep(0.08)
if failures:
    (OUT/'institutional_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
# require at least 95% market-date coverage for each market
cov={m:len({r['date'] for r in inst if r['market']==m})/max(1,len(trade_dates)) for m in ('TWSE','TPEX')}
if min(cov.values())<0.95: raise RuntimeError(f'institutional coverage too low {cov}, failures={len(failures)}')
inst=sorted(inst,key=lambda r:(r['date'],r['market'],r['code']))
write_csv(OUT/'institutional_2026_ytd.csv',inst)
manifest={'dataset':'AlphaPilot 2026 YTD Taiwan market package','generated_at_utc':datetime.utcnow().isoformat()+'Z','coverage':{'start':trade_dates[0],'end':trade_dates[-1],'trading_days':len(trade_dates),'ohlcv_rows':len(rows_ohlcv),'institutional_rows':len(inst),'institutional_market_date_coverage':cov},'ohlcv_source':'GitHub release yukishirotsubasa/tw-stock-data-release (TWSE MI_INDEX + TPEx daily close)','institutional_source':'Official TWSE T86 + TPEx historical institutional daily JSON','weekly_assets':used,'failures':failures}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# zip package
zip_path=ROOT/'AlphaPilot_2026_YTD_Data_Package.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname='2026-YTD/'+p.name)
print('[DONE]',zip_path,'bytes',zip_path.stat().st_size,flush=True)
print(json.dumps(manifest['coverage'],ensure_ascii=False),flush=True)
