#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json, re, time, zipfile, hashlib
from datetime import datetime
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'data'/'history'/'2026-YTD'
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot/ForwardWindow','Accept':'application/json,text/plain,*/*'})
MIN_WEEK=10
INST_LOOKBACK_DAYS=18

def get(url,params=None,timeout=90):
    last=None
    for i in range(5):
        try:
            r=S.get(url,params=params,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e
            if i<4: time.sleep(min(8,2**i))
    raise RuntimeError(f'GET failed {url}: {last}')

def n(x):
    if x is None:return None
    s=str(x).strip().replace(',','').replace('+','').replace('−','-').replace('－','-')
    if s in ('','--','---','null','None'):return None
    try:return float(s)
    except:return None

def code4(x):
    s=str(x or '').strip().strip('=').strip('"'); return s if re.fullmatch(r'\d{4}',s) else None

def parse_date(v):
    s=re.sub(r'[^0-9]','',str(v or ''))
    if len(s)==8:
        try:return datetime.strptime(s,'%Y%m%d').date()
        except:pass
    if len(s)==7:
        try:return datetime.strptime(str(int(s[:3])+1911)+s[3:],'%Y%m%d').date()
        except:pass
    return None

def roc_date(ds):
    d=datetime.strptime(ds,'%Y-%m-%d').date(); return f'{d.year-1911:03d}/{d.month:02d}/{d.day:02d}'

def write_csv(path,rows,fields=None):
    if not rows:raise RuntimeError(f'empty {path}')
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def week_no(name):
    m=re.search(r'weekly_2026_W(\d+)',name);return int(m.group(1)) if m else -1

rel=get('https://api.github.com/repos/yukishirotsubasa/tw-stock-data-release/releases/tags/daily-close-csv').json()
assets=[a for a in rel.get('assets',[]) if a['name'].startswith('weekly_2026_W') and a['name'].endswith('.zip') and week_no(a['name'])>=MIN_WEEK]
if not assets:raise RuntimeError('no usable 2026 weekly assets')
ohlcv={};used=[]
for a in sorted(assets,key=lambda z:week_no(z['name'])):
    blob=get(a['browser_download_url'],timeout=180).content
    digest=hashlib.sha256(blob).hexdigest(); exp=(a.get('digest') or '').replace('sha256:','')
    if exp and digest!=exp:raise RuntimeError(f"{a['name']} sha mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            if not member.lower().endswith('.csv'):continue
            reader=csv.DictReader(io.TextIOWrapper(z.open(member),encoding='utf-8-sig',newline=''))
            for r in reader:
                d=parse_date(r.get('date'));c=code4(r.get('code'))
                if not d or d.year!=2026 or not c:continue
                row={'date':d.isoformat(),'code':c,'name':r.get('name',''),'volume':r.get('volume'),'open':r.get('open'),'high':r.get('high'),'low':r.get('low'),'close':r.get('close')}
                ohlcv[(row['date'],c)]=row
    used.append({'asset':a['name'],'sha256':digest,'bytes':len(blob)})
    print('[OHLCV]',a['name'],len(ohlcv),flush=True)
rows=sorted(ohlcv.values(),key=lambda r:(r['date'],r['code']))
write_csv(OUT/'ohlcv_2026_ytd.csv',rows,['date','code','name','volume','open','high','low','close'])
trade_dates=sorted({r['date'] for r in rows}); recent=trade_dates[-INST_LOOKBACK_DAYS:]

def twse(ds):
    j=get('https://www.twse.com.tw/rwd/zh/fund/T86',{'response':'json','date':ds.replace('-',''),'selectType':'ALLBUT0999'}).json()
    fields=j.get('fields') or [];out=[]
    for vals in j.get('data') or []:
        r=dict(zip(fields,vals));c=code4(r.get('證券代號'))
        if not c:continue
        out.append({'date':ds,'market':'TWSE','code':c,'name':r.get('證券名稱',''),'foreign_net':n(r.get('外陸資買賣超股數(不含外資自營商)')),'trust_net':n(r.get('投信買賣超股數'))})
    if len(out)<700:raise RuntimeError(f'TWSE T86 too few rows {ds}: {len(out)}')
    return out

def tpex(ds):
    j=get('https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade',{'type':'Daily','sect':'EW','date':roc_date(ds),'id':'','response':'json'}).json()
    tables=j.get('tables') or []
    if not tables:raise RuntimeError(f'TPEx no table {ds}')
    resp=str(tables[0].get('date') or j.get('date') or '').strip()
    if resp and resp!=roc_date(ds):raise RuntimeError(f'TPEx wrong date {ds}: {resp}')
    data=tables[0].get('data') or [];out=[]; corrupt=0
    for r in data:
        if not isinstance(r,list) or len(r)<24:continue
        c=code4(r[0])
        if not c:continue
        foreign=n(r[10]); trust=n(r[13]); dealer=n(r[22]); total=n(r[23])
        if None in (foreign,trust,dealer,total):continue
        if abs((foreign+trust+dealer)-total)>0.5:
            corrupt+=1; continue
        out.append({'date':ds,'market':'TPEX','code':c,'name':str(r[1]).strip(),'foreign_net':foreign,'trust_net':trust})
    if corrupt:print('[WARN] TPEx corrupt rows',ds,corrupt,flush=True)
    if len(out)<400:raise RuntimeError(f'TPEx inst too few rows {ds}: {len(out)}')
    return out

inst=[];fail=[]
for i,ds in enumerate(recent,1):
    try:inst+=twse(ds)
    except Exception as e:fail.append({'date':ds,'market':'TWSE','error':str(e)})
    try:inst+=tpex(ds)
    except Exception as e:fail.append({'date':ds,'market':'TPEX','error':str(e)})
    print('[INST]',i,'/',len(recent),'rows',len(inst),'fail',len(fail),flush=True)
if fail:raise RuntimeError('recent institutional gaps: '+json.dumps(fail,ensure_ascii=False))
inst=sorted(inst,key=lambda r:(r['date'],r['market'],r['code']))
write_csv(OUT/'institutional_2026_ytd.csv',inst)
manifest={'dataset':'AlphaPilot 2026 Forward rolling input window','generated_at_utc':datetime.utcnow().isoformat()+'Z','coverage':{'price_start':trade_dates[0],'price_end':trade_dates[-1],'price_days':len(trade_dates),'institutional_start':recent[0],'institutional_end':recent[-1],'institutional_days':len(recent),'ohlcv_rows':len(rows),'institutional_rows':len(inst)},'weekly_assets':used,'failures':fail}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('[DONE]',json.dumps(manifest['coverage'],ensure_ascii=False),flush=True)
