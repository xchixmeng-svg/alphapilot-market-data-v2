#!/usr/bin/env python3
"""Positive-control probe for MOPS full-text announcement search."""
import html,json,re,urllib.parse,urllib.request,http.cookiejar
HOSTS=('https://mopsov.twse.com.tw','https://mops.twse.com.tw')
KINDS=('R','L','O','P','C','')
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}

def main():
 for host in HOSTS:
  jar=http.cookiejar.CookieJar();opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
  try: opener.open(urllib.request.Request(host+'/mops/web/t51sb10_q1',headers={'User-Agent':'Mozilla/5.0'}),timeout=15).read(2048)
  except Exception: pass
  for kind in KINDS:
   fields={'encodeURIComponent':'1','step':'1','firstin':'true','id':'','key':'','TYPEK':'','Stp':'4','go':'false','COMPANY_ID':'','co_id':'','r1':'1','KIND':kind,'CODE':'','keyWord':'43條之1','Condition2':'2','keyWord2':'','year':'115','month1':'10','begin_day':'1','end_day':'31','Orderby':'1'}
   for method in ('GET','POST'):
    url=host+'/mops/web/ajax_t51sb10'; data=None
    if method=='GET': url+='?'+urllib.parse.urlencode(fields)
    else: data=urllib.parse.urlencode(fields).encode()
    req=urllib.request.Request(url,data=data,headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded','Referer':host+'/mops/web/t51sb10_q1'},method=method)
    try:
     with opener.open(req,timeout=35) as r:text=r.read().decode('utf-8','replace')
    except Exception as e:
     print(json.dumps({'host':host,'kind':kind,'method':method,'error':repr(e)},ensure_ascii=False));continue
    compact=re.sub(r'\s+','',html.unescape(re.sub(r'<[^>]+>',' ',text)))
    hit='4194' in compact and '隆欣投資' in compact and '43條之1' in compact and ('115/10/12' in compact or '1151012' in compact)
    print(json.dumps({'host':host,'kind':kind,'method':method,'bytes':len(text),'has4194':'4194' in compact,'hasActor':'隆欣投資' in compact,'has43':'43條之1' in compact,'hasDate':'115/10/12' in compact or '1151012' in compact,'positive_control':hit},ensure_ascii=False))
    if hit:
     print('POSITIVE_CONTROL_OK',json.dumps({**CONTROL,'host':host,'kind':kind,'method':method},ensure_ascii=False));return 0
 print('POSITIVE_CONTROL_FAILED',json.dumps(CONTROL,ensure_ascii=False));return 3
if __name__=='__main__':raise SystemExit(main())
