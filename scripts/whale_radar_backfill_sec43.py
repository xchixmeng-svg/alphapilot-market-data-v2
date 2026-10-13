#!/usr/bin/env python3
"""Backfill Article 43-1 disclosures from MOPS full-text announcement search (t51sb10)."""
from __future__ import annotations
import argparse, calendar, hashlib, html, json, re, time
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

URL='https://mops.twse.com.tw/mops/web/ajax_t51sb10'
UA='Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.4)'
POSITIVE_CONTROL={'roc_year':115,'month':10,'stock_code':'4194','subject_token':'隆欣投資','date':'2026-10-12'}
KEYWORDS=('43條之1','第43條之1','第四十三條之一')
PATTERNS=('43條之1','第43條之1','第四十三條之一','43 條之 1')

class P(HTMLParser):
 def __init__(self): super().__init__(); self.rows=[]; self.r=None; self.c=None
 def handle_starttag(self,t,a):
  if t=='tr': self.r=[]
  elif t in ('td','th') and self.r is not None:self.c=[]
  elif t=='br' and self.c is not None:self.c.append(' ')
 def handle_data(self,d):
  if self.c is not None:self.c.append(d)
 def handle_endtag(self,t):
  if t in ('td','th') and self.c is not None and self.r is not None:self.r.append(re.sub(r'\s+',' ',html.unescape(''.join(self.c))).strip());self.c=None
  elif t=='tr' and self.r is not None:
   if any(self.r):self.rows.append(self.r)
   self.r=None;self.c=None

def has43(s):return any(x.replace(' ','') in s.replace(' ','') for x in PATTERNS)
def rocdate(s):
 m=re.search(r'(?<!\d)(\d{2,3})/(\d{1,2})/(\d{1,2})(?!\d)',s)
 if not m:return ''
 try:return date(int(m[1])+1911,int(m[2]),int(m[3])).isoformat()
 except:return ''
def actor(s):
 for p in (r'代取得人(.+?)依(?:證券交易法|證交法)',r'共同取得人(.+?)依(?:證券交易法|證交法)',r'取得人(.+?)依(?:證券交易法|證交法)'):
  m=re.search(p,s)
  if m:return m[1].strip(' ：:，,。()（）')
 return ''
def parse(page):
 p=P();p.feed(page);out={}
 for cells in p.rows:
  joined=' | '.join(cells)
  if not has43(joined):continue
  code=next((m.group(1) for c in cells if (m:=re.fullmatch(r'\s*(\d{4,6})\s*',c))), '')
  d=rocdate(joined); subj=next((c for c in cells if has43(c)),joined)
  if not code or not d:continue
  a=actor(subj);eid=hashlib.sha256(f'{code}|{d}|{subj}|{a}'.encode()).hexdigest()[:24]
  name=''
  for i,c in enumerate(cells):
   if c.strip()==code and i+1<len(cells):name=cells[i+1].strip();break
  out[eid]={'event_id':eid,'event_type':'sec43_1_disclosure','dataset':'mops_t51sb10_fulltext','stock_code':code,'company_name':name,'actor':a,'event_date':d,'available_at':d,'signal_score':50,'direction':'bullish_watch','subject':subj,'raw_cells':cells,'source':'MOPS announcement full-text search'}
 return list(out.values())

def params(ry,mo,kw):
 days=calendar.monthrange(ry+1911,mo)[1]
 return {'encodeURIComponent':'1','step':'1','firstin':'true','id':'','key':'','TYPEK':'','Stp':'4','go':'false','co_id':'','r1':'1','KIND':'C','CODE':'','keyWord':kw,'Condition2':'2','keyWord2':'','year':str(ry),'month1':str(mo),'begin_day':'1','end_day':str(days),'Orderby':'1'}
def fetch(ry,mo,kw):
 body=urlencode(params(ry,mo,kw)).encode();req=Request(URL,data=body,headers={'User-Agent':UA,'Content-Type':'application/x-www-form-urlencoded','Referer':'https://mops.twse.com.tw/mops/web/t51sb10'},method='POST')
 with urlopen(req,timeout=35) as r:raw=r.read()
 text=raw.decode('utf-8','replace')
 if len(text)<200:raise RuntimeError(f'short response {len(text)}')
 if '查詢過於頻繁' in text or 'FOR SECURITY' in text:raise RuntimeError('MOPS throttle/security')
 return text

def fetch_month(ry,mo):
 all={};diag=[]
 for kw in KEYWORDS:
  try:
   page=fetch(ry,mo,kw);rows=parse(page);diag.append({'keyword':kw,'bytes':len(page),'events':len(rows),'ok':True})
   for r in rows:all[r['event_id']]=r
  except Exception as e:diag.append({'keyword':kw,'events':0,'ok':False,'error':str(e)})
 if not any(x['ok'] for x in diag):raise RuntimeError('all keyword requests failed')
 return list(all.values()),diag

def self_test():
 s='<table><tr><td>4194</td><td>禾生技</td><td>115/10/12</td><td>代取得人隆欣投資(股)公司依證券交易法第43條之1第1項規定辦理公告</td></tr></table>';r=parse(s);assert len(r)==1 and '隆欣投資' in r[0]['actor'];print('sec43-backfill self-test: OK')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='data/whale_radar');ap.add_argument('--start-year',type=int,default=113);ap.add_argument('--end-year',type=int,default=datetime.now().year-1911);ap.add_argument('--sleep',type=float,default=.15);ap.add_argument('--self-test',action='store_true');ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args()
 if a.self_test:self_test();return 0
 root=Path(a.root);latest=root/'latest';latest.mkdir(parents=True,exist_ok=True)
 rows,diag=fetch_month(115,10);c=POSITIVE_CONTROL;hit=any(x['stock_code']==c['stock_code'] and c['subject_token'] in x['subject'] and x['event_date']==c['date'] for x in rows)
 (latest/'sec43_positive_control_probe.json').write_text(json.dumps({'control':c,'positive_control':hit,'diagnostics':diag},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if not hit:
  (latest/'sec43_backfill_manifest.json').write_text(json.dumps({'status':'BROKEN_SOURCE','events':0,'named_actors':0,'positive_control':c},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('historical sec43: BROKEN_SOURCE; positive control not retrieved');return 3
 print('positive control OK via ajax_t51sb10 KIND=C full-text search')
 all={};coverage=[];fails=0;now=date.today()
 for ry in range(a.start_year,a.end_year+1):
  last=now.month if ry+1911==now.year else 12
  for mo in range(1,last+1):
   try:
    rr,d=fetch_month(ry,mo)
    for x in rr:all[x['event_id']]=x
    coverage.append({'roc_year':ry,'month':mo,'status':'OK','events':len(rr),'attempts':d});print(f'MOPS {ry}/{mo:02d}: 43-1={len(rr)}')
   except Exception as e:
    fails+=1;coverage.append({'roc_year':ry,'month':mo,'status':'FAILED','events':0,'error':str(e)});print(f'MOPS {ry}/{mo:02d}: FAILED {e}')
    if not a.allow_partial:raise
   time.sleep(max(0,a.sleep))
 out=sorted(all.values(),key=lambda x:(x['event_date'],x['stock_code']));root.mkdir(parents=True,exist_ok=True)
 with (root/'historical_sec43_events.jsonl').open('w',encoding='utf-8') as f:
  for x in out:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
 manifest={'generated_at':datetime.now().astimezone().isoformat(),'source':'MOPS ajax_t51sb10 full-text announcement search','start_roc_year':a.start_year,'end_roc_year':a.end_year,'months_attempted':len(coverage),'months_failed':fails,'status':'OK' if fails==0 else 'PARTIAL','events':len(out),'named_actors':len({x['actor'] for x in out if x['actor']}),'positive_control':c,'coverage':coverage}
 (latest/'sec43_backfill_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(latest/'historical_sec43_events.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(f"historical sec43: {len(out)} events; named actors={manifest['named_actors']}; failures={fails}/{len(coverage)}");return 0
if __name__=='__main__':raise SystemExit(main())
