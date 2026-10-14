#!/usr/bin/env python3
"""Backfill Article 43-1 disclosures from the dedicated MOPS announcement route.

The old t51 full-text search returns a valid HTML shell but not the 43-1 corpus.
This collector uses t146sb10, discovers the current 43-1 noticeKind from the form
when possible, and refuses all historical research until a known positive control
(4194 禾生技 / 隆欣投資 / 115-10-12) is actually retrieved.
"""
from __future__ import annotations
import argparse, hashlib, html, json, re, time, subprocess, tempfile, os
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

HOSTS=('https://mopsov.twse.com.tw','https://mops.twse.com.tw')
PAGE='/mops/web/t146sb10'; PATH='/mops/web/ajax_t146sb10'
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36'
POSITIVE_CONTROL={'roc_year':115,'month':10,'stock_code':'4194','subject_token':'隆欣投資','date':'2026-10-12'}

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

def rocdate(s):
 for pat in (r'(?<!\d)(\d{2,3})/(\d{1,2})/(\d{1,2})(?!\d)',r'(?<!\d)(\d{3})(\d{2})(\d{2})(?!\d)'):
  m=re.search(pat,s)
  if m:
   try:return date(int(m[1])+1911,int(m[2]),int(m[3])).isoformat()
   except:return ''
 return ''
def actor(s):
 for p in (r'代取得人\s*[:：]?\s*(.+?)(?:依|$)',r'共同取得人\s*[:：]?\s*(.+?)(?:依|$)',r'取得人\s*[:：]?\s*(.+?)(?:依|$)'):
  m=re.search(p,s)
  if m:return m[1].strip(' ：:，,。()（）')
 return ''
def parse(page):
 p=P();p.feed(page);out={}
 for cells in p.rows:
  joined=' | '.join(cells)
  code=next((m.group(1) for c in cells if (m:=re.fullmatch(r'\s*(\d{4,6})\s*',c))), '')
  d=rocdate(joined)
  if not code or not d:continue
  # t146 is already category-scoped; preserve the longest descriptive cell.
  subj=max(cells,key=len) if cells else joined
  a=actor(joined)
  name=''
  for i,c in enumerate(cells):
   if c.strip()==code and i+1<len(cells):name=cells[i+1].strip();break
  eid=hashlib.sha256(f'{code}|{d}|{joined}'.encode()).hexdigest()[:24]
  out[eid]={'event_id':eid,'event_type':'sec43_1_disclosure','dataset':'mops_t146sb10','stock_code':code,'company_name':name,'actor':a,'event_date':d,'available_at':d,'signal_score':50,'direction':'bullish_watch','subject':subj,'raw_cells':cells,'source':'MOPS t146sb10 announcement search'}
 return list(out.values())

def curl(host,path,payload=None,jar=None,max_time=20):
 cmd=['curl','-sS','-L','--http1.1','--compressed','--connect-timeout','5','--max-time',str(max_time),'--retry','1','--retry-delay','1','-A',UA]
 if jar:cmd += ['-c',jar,'-b',jar]
 if payload is not None:
  cmd += ['-H','Content-Type: application/x-www-form-urlencoded','-H','Origin: '+host,'-e',host+PAGE]
  for k,v in payload.items():cmd += ['--data-urlencode',f'{k}={v}']
 cmd.append(host+path)
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=max_time+15)
 if p.returncode:raise RuntimeError((p.stderr or '')[-300:])
 if len(p.stdout)<200:raise RuntimeError(f'short response {len(p.stdout)}')
 if '查詢過於頻繁' in p.stdout or 'FOR SECURITY' in p.stdout:raise RuntimeError('MOPS throttle/security')
 return p.stdout

def discover_notice_kinds(page):
 # Prefer options whose visible label explicitly identifies Article 43-1.
 kinds=[]
 for val,label in re.findall(r'<option[^>]*value=["\']?([^"\' >]+)["\']?[^>]*>(.*?)</option>',page,re.I|re.S):
  txt=re.sub('<[^>]+>','',html.unescape(label)); compact=re.sub(r'\s+','',txt)
  if '43條之1' in compact or '第四十三條之一' in compact:kinds.append(val)
 return list(dict.fromkeys(kinds))

def payload(typek,d1,d2,kind):
 return {'step':'1','firstin':'true','off':'1','keyword4':'','code1':'','TYPEK2':typek,'checkbtn':'','queryName':'co_id_1','inpuType':'co_id','scope':'1','co_id_1':'','co_id_2':'','typek':typek,'TYPEK':typek,'selecttype':'0','date':'4','noticeDate':'1','yymmdd1':d1,'yymmdd2':d2,'noticeKind':kind,'sort':'1'}

def query_range(d1,d2,company=''):
 diagnostics=[]; allrows={}
 for host in HOSTS:
  fd,jar=tempfile.mkstemp(prefix='whale-mops-',suffix='.cookie');os.close(fd)
  try:
   try: form=curl(host,PAGE,jar=jar,max_time=12)
   except Exception as e: diagnostics.append({'host':host,'stage':'warmup','error':repr(e)}); form=''
   kinds=discover_notice_kinds(form)
   # Empty category is a bounded fallback used only to discover/validate the positive control.
   candidates=kinds or ['']
   for typek in ('rotc','pub','otc','sii'):
    for kind in candidates:
     q=payload(typek,d1,d2,kind)
     if company:q['code1']=company;q['co_id_1']=company;q['co_id_2']=company
     try:
      text=curl(host,PATH,q,jar=jar,max_time=18); rows=parse(text)
      row={'host':host,'typek':typek,'noticeKind':kind,'bytes':len(text),'events':len(rows),'has4194':'4194' in text,'hasActor':'隆欣投資' in text}
      diagnostics.append(row)
      for r in rows:allrows[r['event_id']]=r
      if company and row['has4194'] and row['hasActor']:
       return list(allrows.values()),diagnostics,kind
     except Exception as e:diagnostics.append({'host':host,'typek':typek,'noticeKind':kind,'error':repr(e)})
     time.sleep(.15)
   if allrows:return list(allrows.values()),diagnostics,(kinds[0] if kinds else '')
  finally:
   try:os.unlink(jar)
   except OSError:pass
 return list(allrows.values()),diagnostics,''

def month_range(ry,mo):
 import calendar
 last=calendar.monthrange(ry+1911,mo)[1]
 return f'{ry:03d}{mo:02d}01',f'{ry:03d}{mo:02d}{last:02d}'

def self_test():
 s='<table><tr><td>4194</td><td>禾生技</td><td>115/10/12</td><td>取得人：隆欣投資股份有限公司依證券交易法第43條之1辦理公告</td></tr></table>';r=parse(s);assert len(r)==1 and r[0]['stock_code']=='4194' and '隆欣投資' in r[0]['actor'];print('sec43-backfill self-test: OK')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='data/whale_radar');ap.add_argument('--start-year',type=int,default=113);ap.add_argument('--end-year',type=int,default=datetime.now().year-1911);ap.add_argument('--sleep',type=float,default=.25);ap.add_argument('--self-test',action='store_true');ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args()
 if a.self_test:self_test();return 0
 root=Path(a.root);latest=root/'latest';latest.mkdir(parents=True,exist_ok=True)
 rows,diag,kind=query_range('1151012','1151012','4194')
 c=POSITIVE_CONTROL
 hit=any(x['stock_code']==c['stock_code'] and x['event_date']==c['date'] and (c['subject_token'] in (' | '.join(x.get('raw_cells',[])))) for x in rows)
 probe={'control':c,'positive_control':hit,'route':'t146sb10','noticeKind':kind,'diagnostics':diag}
 (latest/'sec43_positive_control_probe.json').write_text(json.dumps(probe,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if not hit:
  (latest/'sec43_backfill_manifest.json').write_text(json.dumps({'status':'BROKEN_SOURCE','events':0,'named_actors':0,**probe},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('historical sec43: BROKEN_SOURCE; positive control not retrieved; '+json.dumps(diag[-12:],ensure_ascii=False));return 3
 print(f'positive control OK via t146sb10; noticeKind={kind!r}')
 allrows={};coverage=[];fails=0;now=date.today()
 for ry in range(a.start_year,a.end_year+1):
  last=now.month if ry+1911==now.year else 12
  for mo in range(1,last+1):
   d1,d2=month_range(ry,mo)
   try:
    rr,d,_=query_range(d1,d2)
    for x in rr:allrows[x['event_id']]=x
    coverage.append({'roc_year':ry,'month':mo,'status':'OK','events':len(rr),'attempts':d});print(f'MOPS t146 {ry}/{mo:02d}: 43-1={len(rr)}')
   except Exception as e:
    fails+=1;coverage.append({'roc_year':ry,'month':mo,'status':'FAILED','events':0,'error':str(e)});print(f'MOPS t146 {ry}/{mo:02d}: FAILED {e}')
    if not a.allow_partial:raise
   time.sleep(max(0,a.sleep))
 out=sorted(allrows.values(),key=lambda x:(x['event_date'],x['stock_code']));root.mkdir(parents=True,exist_ok=True)
 with (root/'historical_sec43_events.jsonl').open('w',encoding='utf-8') as f:
  for x in out:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
 manifest={'generated_at':datetime.now().astimezone().isoformat(),'source':'MOPS t146sb10 dedicated announcement search','noticeKind':kind,'start_roc_year':a.start_year,'end_roc_year':a.end_year,'months_attempted':len(coverage),'months_failed':fails,'status':'OK' if fails==0 else 'PARTIAL','events':len(out),'named_actors':len({x['actor'] for x in out if x['actor']}),'positive_control':c,'coverage':coverage}
 (latest/'sec43_backfill_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(latest/'historical_sec43_events.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(f"historical sec43: {len(out)} events; named actors={manifest['named_actors']}; failures={fails}/{len(coverage)}");return 0
if __name__=='__main__':raise SystemExit(main())
