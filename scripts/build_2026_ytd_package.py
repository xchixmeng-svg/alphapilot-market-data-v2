#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json, re, time, zipfile, hashlib, os
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
    s=str(x or '').strip(); return s if re.fullmatch(r'\d{4}',s) else None

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

# Official TWSE T86, parallel by trading date.
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

twse=[]; failures=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    fut={ex.submit(twse_t86,ds):ds for ds in trade_dates}
    for i,f in enumerate(as_completed(fut),1):
        ds=fut[f]
        try: twse.extend(f.result())
        except Exception as e: failures.append({'date':ds,'market':'TWSE','error':str(e)})
        if i%20==0 or i==len(fut): print('[TWSE INST]',i,'/',len(fut),'rows',len(twse),'failures',len(failures),flush=True)

# TPEx: use FinMind all-market institutional wide data, then keep non-TWSE 4-digit codes.
token=os.environ.get('FINMIND_TOKEN','').strip()
if not token: raise RuntimeError('FINMIND_TOKEN missing; cannot complete TPEx historical institutional data')
params={'dataset':'TaiwanStockInstitutionalInvestorsBuySellWide','start_date':trade_dates[0],'end_date':trade_dates[-1],'token':token}
fj=get('https://api.finmindtrade.com/api/v4/data',params,timeout=240).json()
if fj.get('status') not in (200,None) or not isinstance(fj.get('data'),list): raise RuntimeError(f'FinMind failed: {fj.get("status")} {fj.get("msg")}')
fm=fj['data']; print('[FINMIND] rows',len(fm),flush=True)
twse_codes={r['code'] for r in twse}
tpex=[]
for r in fm:
    c=code4(r.get('stock_id')); ds=str(r.get('date',''))[:10]
    if not c or c in twse_codes or ds not in trade_dates: continue
    fb=n(r.get('Foreign_Investor_buy')); fs=n(r.get('Foreign_Investor_sell'))
    tb=n(r.get('Investment_Trust_buy')); ts=n(r.get('Investment_Trust_sell'))
    db=n(r.get('Dealer_buy')); dsell=n(r.get('Dealer_sell'))
    if db is None:
        db=(n(r.get('Dealer_self_buy')) or 0)+(n(r.get('Dealer_Hedging_buy')) or 0)
    if dsell is None:
        dsell=(n(r.get('Dealer_self_sell')) or 0)+(n(r.get('Dealer_Hedging_sell')) or 0)
    tpex.append({'date':ds,'market':'TPEX','code':c,'name':'',
      'foreign_buy':fb,'foreign_sell':fs,'foreign_net':None if fb is None or fs is None else fb-fs,
      'trust_buy':tb,'trust_sell':ts,'trust_net':None if tb is None or ts is None else tb-ts,
      'dealer_buy':db,'dealer_sell':dsell,'dealer_net':None if db is None or dsell is None else db-dsell,
      'total_net':None})
# dedupe FinMind rows by date/code
idx={(r['date'],r['code']):r for r in tpex}; tpex=list(idx.values())
inst=twse+tpex
cov={m:len({r['date'] for r in inst if r['market']==m})/len(trade_dates) for m in ('TWSE','TPEX')}
if min(cov.values())<0.95: raise RuntimeError(f'institutional coverage too low {cov}; TWSE failures={len(failures)}')
inst=sorted(inst,key=lambda r:(r['date'],r['market'],r['code']))
write_csv(OUT/'institutional_2026_ytd.csv',inst)
if failures:(OUT/'institutional_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
manifest={'dataset':'AlphaPilot 2026 YTD Taiwan market package','generated_at_utc':datetime.utcnow().isoformat()+'Z','coverage':{'start':trade_dates[0],'end':trade_dates[-1],'trading_days':len(trade_dates),'ohlcv_rows':len(rows_ohlcv),'institutional_rows':len(inst),'twse_institutional_rows':len(twse),'tpex_institutional_rows':len(tpex),'institutional_market_date_coverage':cov},'ohlcv_source':'GitHub release yukishirotsubasa/tw-stock-data-release (TWSE MI_INDEX + TPEx daily close)','institutional_source':'TWSE official T86; TPEx historical rows via FinMind TaiwanStockInstitutionalInvestorsBuySellWide fallback','weekly_assets':used,'failures':failures}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
zip_path=ROOT/'AlphaPilot_2026_YTD_Data_Package.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()): z.write(p,arcname='2026-YTD/'+p.name)
print('[DONE]',zip_path,'bytes',zip_path.stat().st_size,flush=True)
print(json.dumps(manifest['coverage'],ensure_ascii=False),flush=True)
