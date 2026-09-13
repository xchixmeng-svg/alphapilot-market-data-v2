#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,os,re,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
import requests

ROOT=Path(__file__).resolve().parent.parent
API='https://api.finmindtrade.com/api/v4/data'
TOKEN=os.environ.get('FINMIND_TOKEN','').strip()
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot-FullMarket/1.0','Accept':'application/json,text/csv,*/*'})

DATASETS=[
 ('month_revenue','TaiwanStockMonthRevenue','2024-01-01','2026-08-31'),
 ('financial_statements','TaiwanStockFinancialStatements','2024-01-01','2026-08-31'),
 ('valuation','TaiwanStockPER','2026-01-01','2026-08-31'),
]

def get(url,params=None,timeout=90,tries=5):
    last=None
    for i in range(tries):
        try:
            r=S.get(url,params=params,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e
            if i<tries-1: time.sleep(min(20,2**i))
    raise RuntimeError(f'GET failed {url}: {last}')

def code_ok(x):
    s=str(x or '').strip().replace('=','').replace('"','')
    return s if re.fullmatch(r'[1-9]\d{3}',s) else None

def universe():
    out={}
    for suffix,market in [('L','TWSE'),('O','TPEX')]:
        url=f'https://mopsfin.twse.com.tw/opendata/t187ap03_{suffix}.csv'
        r=get(url,timeout=90)
        text=r.content.decode('utf-8-sig','ignore')
        for row in csv.DictReader(io.StringIO(text)):
            c=code_ok(row.get('公司代號'))
            if not c: continue
            out[c]={'stock_id':c,'market':market,'name':row.get('公司簡稱') or row.get('公司名稱') or ''}
    if len(out)<1000: raise RuntimeError(f'universe too small: {len(out)}')
    return [out[k] for k in sorted(out)]

def finmind(dataset,stock,start,end):
    p={'dataset':dataset,'data_id':stock,'start_date':start,'end_date':end}
    if TOKEN: p['token']=TOKEN
    last=None
    for i in range(6):
        try:
            r=S.get(API,params=p,timeout=90); r.raise_for_status(); j=r.json()
            if j.get('status') not in (200,None): raise RuntimeError(f"status={j.get('status')} msg={j.get('msg')}")
            d=j.get('data')
            if not isinstance(d,list): raise RuntimeError('data not list')
            return d
        except Exception as e:
            last=e
            if i<5: time.sleep(min(30,2**i))
    raise RuntimeError(str(last))

def write_jsonl_gz(path,rows):
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def read_jsonl_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def mode_shard(idx,total):
    uni=universe(); stocks=[x for i,x in enumerate(uni) if i%total==idx]
    out=ROOT/'shard_out'/f'{idx:02d}'; out.mkdir(parents=True,exist_ok=True)
    rows={k:[] for k,_,_,_ in DATASETS}; errors=[]; success={k:0 for k,_,_,_ in DATASETS}
    for n,meta in enumerate(stocks,1):
        sid=meta['stock_id']
        for short,ds,start,end in DATASETS:
            try:
                d=finmind(ds,sid,start,end)
                for r in d:
                    r=dict(r); r['stock_id']=str(r.get('stock_id') or sid); r['_market']=meta['market']; r['_name']=meta['name']
                    rows[short].append(r)
                if d: success[short]+=1
            except Exception as e:
                errors.append({'stock_id':sid,'market':meta['market'],'dataset':ds,'error':str(e)})
        if n%10==0 or n==len(stocks): print(f'[SHARD {idx}] {n}/{len(stocks)} success={success} errors={len(errors)}',flush=True)
    for short in rows: write_jsonl_gz(out/f'{short}.jsonl.gz',rows[short])
    manifest={'shard':idx,'total_shards':total,'stocks':len(stocks),'success_stock_counts':success,'row_counts':{k:len(v) for k,v in rows.items()},'errors':errors,'generated_at_utc':datetime.now(timezone.utc).isoformat()}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[DONE SHARD]',idx,manifest['row_counts'],flush=True)

def write_csv_gz(path,rows):
    rows=list(rows)
    if not rows: raise RuntimeError(f'empty {path}')
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with gzip.open(path,'wt',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    return len(rows)

def mode_aggregate(shard_root,total):
    base=Path(shard_root); out=ROOT/'data'/'history'/'full-market-fundamental-2024-2026'; out.mkdir(parents=True,exist_ok=True)
    manifests=[]; allrows={k:[] for k,_,_,_ in DATASETS}
    for idx in range(total):
        candidates=list(base.glob(f'**/{idx:02d}/manifest.json'))+list(base.glob(f'**/shard-{idx:02d}/**/manifest.json'))
        if not candidates: candidates=list(base.glob(f'**/{idx:02d}/**/manifest.json'))
        if not candidates: raise RuntimeError(f'missing shard {idx}')
        mp=candidates[0]; manifests.append(json.loads(mp.read_text(encoding='utf-8')))
        d=mp.parent
        for short in allrows:
            p=d/f'{short}.jsonl.gz'
            if not p.exists(): raise RuntimeError(f'missing {p}')
            allrows[short].extend(read_jsonl_gz(p))
    files=[]; row_counts={}
    for short in allrows:
        p=out/f'{short}.csv.gz'; row_counts[short]=write_csv_gz(p,allrows[short]); files.append({'file':p.name,'rows':row_counts[short],'bytes':p.stat().st_size,'sha256':sha256(p)})
    eps=[]
    for r in allrows['financial_statements']:
        blob='|'.join(str(r.get(k,'')) for k in ('type','origin_name','name')).lower()
        if 'eps' in blob or '每股盈餘' in blob or '每股盈余' in blob or '基本每股' in blob: eps.append(r)
    p_eps=out/'eps_actual_extract.csv.gz'; eps_rows=write_csv_gz(p_eps,eps); files.append({'file':p_eps.name,'rows':eps_rows,'bytes':p_eps.stat().st_size,'sha256':sha256(p_eps)})
    uni=universe(); p_uni=out/'stock_universe.csv.gz'; write_csv_gz(p_uni,uni); files.append({'file':p_uni.name,'rows':len(uni),'bytes':p_uni.stat().st_size,'sha256':sha256(p_uni)})
    errors=[e for m in manifests for e in m.get('errors',[])]
    success_counts={k:sum(m.get('success_stock_counts',{}).get(k,0) for m in manifests) for k in allrows}
    manifest={'dataset':'AlphaPilot Full-Market Fundamental 2024-2026','generated_at_utc':datetime.now(timezone.utc).isoformat(),'universe_stocks':len(uni),'row_counts':row_counts|{'eps_actual_extract':eps_rows},'success_stock_counts':success_counts,'error_count':len(errors),'errors':errors[:5000],'sources':['MOPS company universe','FinMind TaiwanStockMonthRevenue','FinMind TaiwanStockFinancialStatements','FinMind TaiwanStockPER'],'coverage':{'month_revenue':'2024-01-01..2026-08-31 requested per stock','financial_statements':'2024-01-01..2026-08-31 requested per stock','valuation':'2026-01-01..2026-08-31 requested per stock'},'point_in_time_note':'Raw source dates preserved. Any backtest must apply report/publication availability and never expose future reports.','analyst_revision_note':'Contains actual public fundamentals, not paid historical analyst-consensus revisions. EPS Revision Proxy should be derived from monthly revenue acceleration + reported financial trends + historical valuation.','files':files}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.txt').write_text('AlphaPilot full-market fundamental package for blind market-wide screening. Includes listed+OTC universe, monthly revenue, financial statements/EPS extract, and daily valuation.\n',encoding='utf-8')
    zpath=ROOT/'AlphaPilot_Full_Market_Fundamental_2024_2026.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p,arcname='Full-Market-Fundamental/'+p.name)
    print('[DONE AGG]',zpath,'bytes',zpath.stat().st_size,'stocks',len(uni),'rows',manifest['row_counts'],'errors',len(errors),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('shard'); s.add_argument('--index',type=int,required=True); s.add_argument('--total',type=int,required=True)
    a=sub.add_parser('aggregate'); a.add_argument('--root',required=True); a.add_argument('--total',type=int,required=True)
    x=ap.parse_args()
    if x.mode=='shard': mode_shard(x.index,x.total)
    else: mode_aggregate(x.root,x.total)
