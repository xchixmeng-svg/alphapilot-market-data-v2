#!/usr/bin/env python3
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import json

HOSTS=['https://mops.twse.com.tw','https://mopsov.twse.com.tw']
PATHS=['/mops/web/t05st01','/mops/web/ajax_t05st01']
PARAM_VARIANTS=[
 {'TYPEK':'all','co_id':'4194','encodeURIComponent':'1','firstin':'1','inpuType':'co_id','off':'1','step':'1','year':'115'},
 {'TYPEK':'all','co_id':'4194','firstin':'true','step':'1','year':'115'},
 {'TYPEK':'all','co_id':'4194','firstin':'true','step':'1','year':'115','month':'10'},
]
needle=['4194','隆欣投資','43條之1']
out=[]
for host in HOSTS:
  for path in PATHS:
    for method in ['GET','POST']:
      for i,p in enumerate(PARAM_VARIANTS):
        try:
          headers={'User-Agent':'Mozilla/5.0','Accept-Language':'zh-TW,zh;q=0.9'}
          if method=='GET':
            req=Request(host+path+'?'+urlencode(p),headers=headers)
          else:
            req=Request(host+path,data=urlencode(p).encode(),headers={**headers,'Content-Type':'application/x-www-form-urlencoded'},method='POST')
          with urlopen(req,timeout=25) as r: raw=r.read()
          text=raw.decode('utf-8','replace')
          hit=all(x in text for x in needle)
          row={'host':host,'path':path,'method':method,'variant':i,'bytes':len(raw),'hit':hit,'has4194':'4194' in text,'has43':'43條之1' in text,'hasActor':'隆欣投資' in text}
          out.append(row); print(row)
          if hit:
            print('POSITIVE_CONTROL_OK',json.dumps(row,ensure_ascii=False)); raise SystemExit(0)
        except SystemExit: raise
        except Exception as e:
          row={'host':host,'path':path,'method':method,'variant':i,'error':repr(e)}; out.append(row); print(row)
print('POSITIVE_CONTROL_FAILED')
raise SystemExit(3)
