#!/usr/bin/env python3
from __future__ import annotations
import io, json, re, time, zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np
import requests

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'.clean_cache'
CACHE.mkdir(exist_ok=True)
YEARS=range(2020,2026)
RELEASE='https://github.com/yukishirotsubasa/tw-stock-data-release/releases/download/daily-close-csv/yearly_{year}.zip'
TWSE_T86='https://www.twse.com.tw/rwd/zh/fund/T86'
TPEX_OLD='https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php'
TPEX_NEW='https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade'
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot-Clean-Research/1.0','Accept':'application/json,text/plain,*/*','Accept-Language':'zh-TW,zh;q=0.9,en;q=0.5'})

def _get(url, params=None, tries=5):
    last=None
    for i in range(tries):
        try:
            r=S.get(url,params=params or {},timeout=(20,120)); r.raise_for_status()
            if not r.content: raise RuntimeError('empty response')
            return r
        except Exception as e:
            last=e
            if i+1<tries: time.sleep(min(20,2**i))
    raise RuntimeError(f'download failed {url}: {last}')

def _nk(x): return re.sub(r'[\s_\-()/（）％%]+','',str(x or '')).lower()
def _num(x):
    s=str(x or '').strip().replace(',','').replace('+','').replace('−','-')
    if s in {'','--','---','----','null','None'}: return np.nan
    try: return float(s)
    except: return np.nan

def _nint(x):
    z=_num(x); return np.nan if not np.isfinite(z) else int(round(z))
def _roc(d): return f'{d.year-1911}/{d:%m/%d}'

def _tables(payload):
    out=[]
    def walk(x):
        if isinstance(x,dict):
            f=x.get('fields'); d=x.get('data')
            if isinstance(f,list) and isinstance(d,list) and d: out.append((f,d))
            aa=x.get('aaData')
            if isinstance(aa,list) and aa: out.append((f or [],aa))
            for k,v in x.items():
                if str(k).startswith('fields') and isinstance(v,list):
                    dd=x.get('data'+str(k)[6:])
                    if isinstance(dd,list) and dd: out.append((v,dd))
                if isinstance(v,(dict,list)): walk(v)
        elif isinstance(x,list):
            if x and isinstance(x[0],dict): out.append((list(x[0].keys()),x))
            for v in x[:10]:
                if isinstance(v,(dict,list)): walk(v)
    walk(payload); return out

def _dicts(fields, rows):
    return [r if isinstance(r,dict) else {fields[i]:r[i] if i<len(r) else None for i in range(len(fields))} for r in rows]
def _pick(row,*names):
    m={_nk(k):v for k,v in row.items()}
    for n in names:
        if _nk(n) in m: return m[_nk(n)]
    return None

def _code(row):
    c=str(_pick(row,'Code','SecuritiesCompanyCode','證券代號','代號','股票代號') or '').strip().replace('"','').replace('=','')
    return c if re.fullmatch(r'\d{4}',c) else None

def _match(row, inst, actions, reject=()):
    c=[]
    for k,v in row.items():
        z=_nk(k)
        if any(z.startswith(_nk(x)) for x in reject): continue
        if any(_nk(x) in z for x in inst) and any(_nk(x) in z for x in actions): c.append((-len(z),v))
    return sorted(c,reverse=True)[0][1] if c else None

def _fval(r,a):
    v=_match(r,['外陸資','外資及陸資'],a)
    return v if v is not None else _match(r,['Foreign','外資'],a,['外資自營商'])
def _tval(r,a): return _match(r,['InvestmentTrust','Trust','投信'],a)
def _dval(r,a):
    c=[]
    for k,v in r.items():
        z=_nk(k)
        if z.startswith(_nk('外資自營商')): continue
        if not(z.startswith(_nk('自營商')) or 'dealer' in z) or not any(_nk(x) in z for x in a): continue
        score=-len(z)+(10000 if '自行買賣' not in z and '避險' not in z else 0); c.append((score,v))
    return sorted(c,reverse=True)[0][1] if c else None

def load_ohlcv():
    out=[]
    for y in YEARS:
        p=CACHE/f'ohlcv_{y}.parquet'
        if p.exists(): q=pd.read_parquet(p)
        else:
            r=_get(RELEASE.format(year=y))
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names=[n for n in z.namelist() if n.lower().endswith('.csv')]
                if not names: raise RuntimeError(f'yearly_{y}.zip has no CSV')
                q=pd.read_csv(z.open(names[0]),dtype={'code':str})
            need=['date','code','name','volume','open','high','low','close']
            if any(c not in q.columns for c in need): raise RuntimeError(f'OHLCV {y} missing columns')
            q=q[need].copy(); q['code']=q.code.astype(str).str.strip().str.replace(r'\.0$','',regex=True).str.zfill(4)
            for c in ['date','volume','open','high','low','close']: q[c]=pd.to_numeric(q[c],errors='coerce')
            q=q.dropna(subset=['date','code','open','high','low','close']); q['date']=q.date.astype(int); q['volume']=q.volume.fillna(0).astype(int)
            q=q.drop_duplicates(['date','code'],keep='last').sort_values(['date','code']); q.to_parquet(p,index=False)
        out.append(q)
    q=pd.concat(out,ignore_index=True).drop_duplicates(['date','code'],keep='last').sort_values(['date','code']).reset_index(drop=True)
    if '0050' not in set(q.code): raise RuntimeError('0050 missing from OHLCV')
    return q

def _norm_inst(rows,d,market):
    out=[]
    for r in rows:
        c=_code(r)
        if not c: continue
        fb,fs,fn=_nint(_fval(r,['Buy','買進'])),_nint(_fval(r,['Sell','賣出'])),_nint(_fval(r,['Net','Difference','買賣超','差額']))
        tb,ts,tn=_nint(_tval(r,['Buy','買進'])),_nint(_tval(r,['Sell','賣出'])),_nint(_tval(r,['Net','Difference','買賣超','差額']))
        db,ds,dn=_nint(_dval(r,['Buy','買進'])),_nint(_dval(r,['Sell','賣出'])),_nint(_dval(r,['Net','Difference','買賣超','差額']))
        if not np.isfinite(fn) and np.isfinite(fb) and np.isfinite(fs): fn=int(fb-fs)
        if not np.isfinite(tn) and np.isfinite(tb) and np.isfinite(ts): tn=int(tb-ts)
        if not np.isfinite(dn) and np.isfinite(db) and np.isfinite(ds): dn=int(db-ds)
        out.append({'date':int(d.strftime('%Y%m%d')),'market':market,'code':c,'foreign_net':fn,'trust_net':tn,'dealer_net':dn})
    return out

def _twse_inst(d):
    p=_get(TWSE_T86,{'date':d.strftime('%Y%m%d'),'selectType':'ALLBUT0999','response':'json'}).json()
    best=[]
    for f,r in _tables(p):
        z=_norm_inst(_dicts(f,r),d,'TWSE')
        if len(z)>len(best): best=z
    return best

def _tpex_inst(d):
    opts=[(TPEX_OLD,{'l':'zh-tw','o':'json','se':'EW','t':'D','d':_roc(d)}),(TPEX_NEW,{'date':d.strftime('%Y/%m/%d'),'response':'json','type':'Daily'}),(TPEX_NEW,{'date':d.strftime('%Y/%m/%d'),'response':'json'})]
    errs=[]
    for u,p in opts:
        try:
            payload=_get(u,p,3).json(); best=[]
            for f,r in _tables(payload):
                z=_norm_inst(_dicts(f,r),d,'TPEX')
                if len(z)>len(best): best=z
            if len(best)>=50: return best
            errs.append(f'{u} rows={len(best)}')
        except Exception as e: errs.append(str(e))
    raise RuntimeError('; '.join(errs))

def load_institutional(trading_dates: Iterable[int]):
    p=CACHE/'institutional_2020_2025.parquet'
    have=pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=['date','market','code','foreign_net','trust_net','dealer_net'])
    done=set(pd.to_numeric(have.get('date'),errors='coerce').dropna().astype(int).unique())
    target=sorted(int(x) for x in trading_dates)
    rows=[]; failures=[]
    for n,di in enumerate(target,1):
        if di in done: continue
        d=datetime.strptime(str(di),'%Y%m%d').date(); a=b=[]
        try: a=_twse_inst(d)
        except Exception as e: failures.append(f'{di} TWSE {e}')
        time.sleep(.05)
        try: b=_tpex_inst(d)
        except Exception as e: failures.append(f'{di} TPEX {e}')
        rows.extend(a); rows.extend(b)
        if rows and (n%20==0 or n==len(target)):
            have=pd.concat([have,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(['date','market','code'],keep='last').sort_values(['date','market','code'])
            have.to_parquet(p,index=False); rows=[]
        time.sleep(.05)
    if rows:
        have=pd.concat([have,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(['date','market','code'],keep='last').sort_values(['date','market','code']); have.to_parquet(p,index=False)
    coverage=set(have.date.astype(int).unique()) & set(target)
    ratio=len(coverage)/len(target) if target else 0
    if ratio<0.98: raise RuntimeError(f'institutional coverage {ratio:.2%}; failures={failures[:10]}')
    return have[have.date.astype(int).isin(target)].copy()

def main():
    o=load_ohlcv(); dates=sorted(o.date.unique())
    ins=load_institutional(dates)
    report={'status':'PASS','ohlcv_rows':int(len(o)),'ohlcv_dates':int(o.date.nunique()),'ohlcv_min':int(o.date.min()),'ohlcv_max':int(o.date.max()),'institutional_rows':int(len(ins)),'institutional_dates':int(ins.date.nunique()),'0050_rows':int((o.code=='0050').sum())}
    if report['ohlcv_min']>20200102 or report['ohlcv_max']<20251231: raise RuntimeError(report)
    if report['institutional_dates'] < int(report['ohlcv_dates']*.98): raise RuntimeError(report)
    (CACHE/'history_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
