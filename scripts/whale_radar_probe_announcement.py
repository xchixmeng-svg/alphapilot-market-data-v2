#!/usr/bin/env python3
"""Positive-control probe for MOPS Article 43-1 announcement search."""
import html, json, re, urllib.parse, urllib.request, http.cookiejar

HOSTS=('https://mopsov.twse.com.tw','https://mops.twse.com.tw')
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}
UA='Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.4)'

def clean(text): return re.sub(r'\s+','',html.unescape(re.sub(r'<[^>]+>',' ',text)))
def req(opener,host,path,fields):
    data=urllib.parse.urlencode(fields).encode('utf-8')
    r=urllib.request.Request(host+path,data=data,headers={'User-Agent':UA,'Accept-Language':'zh-TW,zh;q=0.9','Content-Type':'application/x-www-form-urlencoded','Referer':host+'/mops/web/t51sb10_q1'},method='POST')
    with opener.open(r,timeout=35) as x: return x.read().decode('utf-8','replace')
def hit(text):
    c=clean(text)
    return '4194' in c and '隆欣投資' in c and ('43條之1' in c or '第四十三條之一' in c) and ('115/10/12' in c or '1151012' in c)

def main():
  # MOPS 公告查詢 / 主旨全文檢索. Historical examples of this form use KIND=C,
  # month1=0 (whole year), blank company selector, and Condition2/Orderby fields.
  # Probe both the statutory phrase and the known actor to prove retrieval.
  for host in HOSTS:
    jar=http.cookiejar.CookieJar(); opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try: opener.open(urllib.request.Request(host+'/mops/web/t51sb10_q1',headers={'User-Agent':UA}),timeout=15).read(8192)
    except Exception: pass
    for keyword in ('43條之1','隆欣投資','證券交易法第43條之1第1項'):
      for kind in ('C','','R','P'):
        for month1 in ('0','10'):
          fields={'encodeURIComponent':'1','step':'1','firstin':'true','id':'','key':'','TYPEK':'','Stp':'4','go':'false','co_id':'','COMPANY_ID':'','r1':'1','KIND':kind,'CODE':'','keyWord':keyword,'Condition2':'2','keyWord2':'','year':'115','month1':month1,'begin_day':'1','end_day':'31','Orderby':'1'}
          try:
            text=req(opener,host,'/mops/web/ajax_t51sb10',fields); c=clean(text); ok=hit(text)
            print(json.dumps({'host':host,'kind':kind,'month1':month1,'keyword':keyword,'bytes':len(text),'has4194':'4194' in c,'hasActor':'隆欣投資' in c,'has43':('43條之1' in c or '第四十三條之一' in c),'hasDate':('115/10/12' in c or '1151012' in c),'positive_control':ok},ensure_ascii=False))
            if ok:
              print('POSITIVE_CONTROL_OK',json.dumps({**CONTROL,'host':host,'kind':kind,'month1':month1,'keyword':keyword},ensure_ascii=False)); return 0
          except Exception as e: print(json.dumps({'host':host,'kind':kind,'month1':month1,'keyword':keyword,'error':repr(e)},ensure_ascii=False))
  print('POSITIVE_CONTROL_FAILED',json.dumps(CONTROL,ensure_ascii=False)); return 3
if __name__=='__main__': raise SystemExit(main())
