#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,os,re,time,zipfile
from collections import defaultdict
from datetime import date,datetime,timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent.parent
API='https://api.finmindtrade.com/api/v4/data'
TOKEN=os.environ.get('FINMIND_TOKEN','').strip()
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot-Fundamentals2019/1.0','Accept':'application/json,text/csv,*/*'})
START='2019-01-01'; END=date.today().isoformat()


def get(url,params=None,timeout=90,tries=6):
    last=None
    for i in range(tries):
        try:
            r=S.get(url,params=params,timeout=timeout); r.raise_for_status(); return r
        except Exception as e:
            last=e
            if i<tries-1: time.sleep(min(30,2**i))
    raise RuntimeError(f'GET failed {url}: {last}')

def code_ok(x):
    s=str(x or '').strip().replace('=','').replace('"','')
    return s if re.fullmatch(r'[1-9]\d{3}',s) else None

def universe():
    out={}
    for suffix,market in [('L','TWSE'),('O','TPEX')]:
        url=f'https://mopsfin.twse.com.tw/opendata/t187ap03_{suffix}.csv'
        text=get(url).content.decode('utf-8-sig','ignore')
        for r in csv.DictReader(io.StringIO(text)):
            c=code_ok(r.get('公司代號'))
            if c: out[c]={'stock_id':c,'market':market,'name':r.get('公司簡稱') or r.get('公司名稱') or ''}
    if len(out)<1000: raise RuntimeError(f'universe too small {len(out)}')
    return [out[k] for k in sorted(out)]

def finmind(ds,sid):
    p={'dataset':ds,'data_id':sid,'start_date':START,'end_date':END}
    if TOKEN:p['token']=TOKEN
    last=None
    for i in range(6):
        try:
            r=S.get(API,params=p,timeout=90); r.raise_for_status(); j=r.json()
            if j.get('status') not in (200,None): raise RuntimeError(f"{j.get('status')} {j.get('msg')}")
            d=j.get('data')
            if not isinstance(d,list): raise RuntimeError('data not list')
            return d
        except Exception as e:
            last=e
            if i<5:time.sleep(min(30,2**i))
    raise RuntimeError(str(last))

def write_jsonl_gz(p,rows):
    with gzip.open(p,'wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def read_jsonl_gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def mode_shard(idx,total):
    uni=universe(); stocks=[x for i,x in enumerate(uni) if i%total==idx]
    out=ROOT/'fund2019_shards'/f'{idx:02d}';out.mkdir(parents=True,exist_ok=True)
    rev=[];fin=[];errors=[];ok={'revenue':0,'financial':0}
    for n,m in enumerate(stocks,1):
        sid=m['stock_id']
        for short,ds,target in [('revenue','TaiwanStockMonthRevenue',rev),('financial','TaiwanStockFinancialStatements',fin)]:
            try:
                rows=finmind(ds,sid)
                for r in rows:
                    x=dict(r);x['stock_id']=str(x.get('stock_id') or sid);x['_market']=m['market'];x['_name']=m['name'];target.append(x)
                if rows:ok[short]+=1
            except Exception as e:errors.append({'stock_id':sid,'dataset':ds,'error':str(e)})
        if n%10==0 or n==len(stocks):print(f'[SHARD {idx}] {n}/{len(stocks)} ok={ok} errors={len(errors)}',flush=True)
    write_jsonl_gz(out/'revenue.jsonl.gz',rev);write_jsonl_gz(out/'financial.jsonl.gz',fin)
    (out/'manifest.json').write_text(json.dumps({'shard':idx,'stocks':len(stocks),'ok':ok,'rows':{'revenue':len(rev),'financial':len(fin)},'errors':errors},ensure_ascii=False,indent=2),encoding='utf-8')

def num(v):
    try:return float(v)
    except:return None

def quarter_from_date(s):
    try:
        d=datetime.strptime(str(s)[:10],'%Y-%m-%d');return f'{d.year}Q{(d.month-1)//3+1}'
    except:return ''

def write_csv(p,rows,fields):
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def mode_aggregate(root,total):
    base=Path(root); manifests=[]; rev=[]; fin=[]
    for i in range(total):
        ms=list(base.glob(f'**/{i:02d}/manifest.json'))
        if not ms:raise RuntimeError(f'missing shard {i}')
        mp=ms[0];manifests.append(json.loads(mp.read_text(encoding='utf-8')))
        rev.extend(read_jsonl_gz(mp.parent/'revenue.jsonl.gz'));fin.extend(read_jsonl_gz(mp.parent/'financial.jsonl.gz'))
    out=ROOT/'data'/'fundamentals_2019_latest';out.mkdir(parents=True,exist_ok=True)

    # monthly revenue normalized
    monthly=[]
    for r in rev:
        sid=str(r.get('stock_id','')); d=str(r.get('date',''))
        try:dt=datetime.strptime(d[:10],'%Y-%m-%d')
        except:continue
        cur=num(r.get('revenue')); prev_y=num(r.get('revenue_lastyear')); mom=num(r.get('revenue_month'))
        yoy=num(r.get('revenue_year'))
        # FinMind fields revenue_year / revenue_month are usually percentages; keep source values.
        monthly.append({'stock_id':sid,'year':dt.year,'month':dt.month,'revenue':cur,'last_year_revenue':prev_y,'yoy_pct':yoy,'mom_pct':mom,'market':r.get('_market',''),'name':r.get('_name',''),'source_date':d})
    monthly.sort(key=lambda x:(x['stock_id'],x['year'],x['month']))
    p1=out/'monthly_revenue_2019_2026.csv'
    write_csv(p1,monthly,['stock_id','year','month','revenue','last_year_revenue','yoy_pct','mom_pct','market','name','source_date'])

    # quarterly statement pivot for selected useful metrics + all raw rows separately
    aliases={
      'EPS':'eps','Revenue':'revenue','GrossProfit':'gross_profit','OperatingIncome':'operating_income','IncomeAfterTaxes':'net_income',
      'Equity':'equity','TotalAssets':'total_assets','TotalLiabilities':'total_liabilities','Liabilities':'total_liabilities'
    }
    by=defaultdict(dict); meta={}
    for r in fin:
        sid=str(r.get('stock_id',''));q=quarter_from_date(r.get('date'))
        if not q:continue
        k=(sid,q);meta[k]={'stock_id':sid,'quarter':q,'market':r.get('_market',''),'name':r.get('_name',''),'report_date':str(r.get('date',''))}
        typ=str(r.get('type','')); val=num(r.get('value'))
        if typ in aliases and val is not None:by[k][aliases[typ]]=val
    qrows=[]
    for k in sorted(meta):
        x=dict(meta[k]);x.update(by.get(k,{}))
        revv=x.get('revenue');gp=x.get('gross_profit');ni=x.get('net_income');eq=x.get('equity');li=x.get('total_liabilities');ta=x.get('total_assets')
        x['gross_margin_pct']=None if gp is None or revv in (None,0) else gp/revv*100
        x['roe_pct']=None if ni is None or eq in (None,0) else ni/eq*100
        x['debt_ratio_pct']=None if li is None or ta in (None,0) else li/ta*100
        qrows.append(x)
    p2=out/'quarterly_eps_2019_2026.csv'
    fields=['stock_id','quarter','eps','revenue','gross_profit','gross_margin_pct','operating_income','net_income','equity','total_assets','total_liabilities','roe_pct','debt_ratio_pct','market','name','report_date']
    write_csv(p2,qrows,fields)

    # keep raw financial rows for future factor engineering
    rawp=out/'quarterly_financial_raw_2019_2026.jsonl.gz';write_jsonl_gz(rawp,fin)
    errors=[e for m in manifests for e in m.get('errors',[])]
    manifest={'dataset':'AlphaPilot monthly revenue + quarterly EPS 2019-latest','generated_at_utc':datetime.now(timezone.utc).isoformat(),'requested_start':START,'requested_end':END,'universe':'all current TWSE+TPEX normal 4-digit stocks','row_counts':{'monthly_revenue':len(monthly),'quarterly_eps':len(qrows),'raw_financial_rows':len(fin)},'error_count':len(errors),'errors':errors[:5000],'sources':['MOPS listed/OTC company universe','FinMind TaiwanStockMonthRevenue','FinMind TaiwanStockFinancialStatements'],'files':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in (p1,p2,rawp)]}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    z=ROOT/'AlphaPilot_Fundamentals_2019_Latest.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in sorted(out.iterdir()):zz.write(p,arcname=p.name)
    print('[DONE]',z,z.stat().st_size,manifest['row_counts'],'errors',len(errors),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('shard');s.add_argument('--index',type=int,required=True);s.add_argument('--total',type=int,required=True)
    a=sub.add_parser('aggregate');a.add_argument('--root',required=True);a.add_argument('--total',type=int,required=True)
    x=ap.parse_args(); mode_shard(x.index,x.total) if x.mode=='shard' else mode_aggregate(x.root,x.total)
