#!/usr/bin/env python3
"""Positive-control probe for MOPS full-text announcement search."""
import html,json,re,urllib.parse,urllib.request
URL='https://mops.twse.com.tw/mops/web/ajax_t51sb10'
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}

def main():
 fields={'encodeURIComponent':'1','step':'1','firstin':'true','id':'','key':'','TYPEK':'','Stp':'4','go':'false','co_id':'','r1':'1','KIND':'C','CODE':'','keyWord':'43條之1','Condition2':'2','keyWord2':'','year':'115','month1':'10','begin_day':'1','end_day':'31','Orderby':'1'}
 req=urllib.request.Request(URL,data=urllib.parse.urlencode(fields).encode(),headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded','Referer':'https://mops.twse.com.tw/mops/web/t51sb10'},method='POST')
 try:
  with urllib.request.urlopen(req,timeout=35) as r:text=r.read().decode('utf-8','replace')
 except Exception as e:print('POSITIVE_CONTROL_FAILED',repr(e));return 3
 compact=re.sub(r'\s+','',html.unescape(re.sub(r'<[^>]+>',' ',text)))
 hit='4194' in compact and '隆欣投資' in compact and ('43條之1' in compact or '第43條之1' in compact) and ('115/10/12' in compact or '1151012' in compact)
 print(json.dumps({'bytes':len(text),'has4194':'4194' in compact,'hasActor':'隆欣投資' in compact,'has43':'43條之1' in compact or '第43條之1' in compact,'hasDate':'115/10/12' in compact or '1151012' in compact,'positive_control':hit},ensure_ascii=False))
 if hit:print('POSITIVE_CONTROL_OK',json.dumps(CONTROL,ensure_ascii=False));return 0
 print('POSITIVE_CONTROL_FAILED',json.dumps(CONTROL,ensure_ascii=False));return 3
if __name__=='__main__':raise SystemExit(main())
