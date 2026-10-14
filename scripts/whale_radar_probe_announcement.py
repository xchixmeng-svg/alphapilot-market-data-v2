#!/usr/bin/env python3
"""Positive-control probe for the current MOPS JSON API."""
import json, urllib.request

URL='https://mops.twse.com.tw/mops/api/t05st01'
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}

def post(payload):
    body=json.dumps(payload,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json','User-Agent':'Mozilla/5.0'},method='POST')
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))

def main():
    payload={'companyId':'4194','year':'115','month':'10','firstDay':'12','lastDay':'12'}
    try:
        obj=post(payload)
    except Exception as exc:
        print('POSITIVE_CONTROL_TRANSPORT_FAILED',repr(exc)); return 3
    rows=((obj.get('result') or {}).get('data') or [])
    for row in rows:
        text=' '.join(str(x) for x in row[:5])
        if '4194' in text and '隆欣投資' in text and ('43條之1' in text or '43 條之 1' in text):
            print('POSITIVE_CONTROL_OK',json.dumps({'payload':payload,'row':row[:5]},ensure_ascii=False)); return 0
    print('POSITIVE_CONTROL_FAILED',json.dumps({'payload':payload,'rows':rows,'control':CONTROL},ensure_ascii=False)); return 3
if __name__=='__main__': raise SystemExit(main())
