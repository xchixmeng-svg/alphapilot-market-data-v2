#!/usr/bin/env python3
"""Positive-control probe for the current MOPS JSON material-info API.

Try the exact date before broad historical queries.  MOPS currently truncates some
company/year queries, so a broad `month=all` response is not a valid negative control.
"""
import json, urllib.request

URL='https://mops.twse.com.tw/mops/api/t05st01'
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}

def post(payload):
    body=json.dumps(payload,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json','Accept':'application/json, text/plain, */*','User-Agent':'Mozilla/5.0'},method='POST')
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

def main():
    variants=[
      {'companyId':'4194','year':'115','month':'10','firstDay':'12','lastDay':'12'},
      {'companyId':'4194','year':'115','month':'10','firstDay':'1','lastDay':'31'},
      {'companyId':'4194','year':'115','month':'all','firstDay':'','lastDay':''},
      {'companyId':'','year':'115','month':'10','firstDay':'12','lastDay':'12'},
    ]
    diagnostics=[]
    for payload in variants:
      try:
        obj=post(payload); rows=((obj.get('result') or {}).get('data') or [])
        hitrow=None
        for row in rows:
          text=' '.join(str(x) for x in row)
          if '4194' in text and '隆欣投資' in text and ('43條之1' in text or '43 條之 1' in text): hitrow=row; break
        diagnostics.append({'payload':payload,'row_count':len(rows),'hit':bool(hitrow),'sample':rows[:2]})
        if hitrow:
          print('POSITIVE_CONTROL_OK',json.dumps({'payload':payload,'row':hitrow,'diagnostics':diagnostics},ensure_ascii=False)); return 0
      except Exception as exc:
        diagnostics.append({'payload':payload,'error':repr(exc)})
    print('POSITIVE_CONTROL_FAILED',json.dumps({'control':CONTROL,'diagnostics':diagnostics},ensure_ascii=False)); return 3
if __name__=='__main__': raise SystemExit(main())
