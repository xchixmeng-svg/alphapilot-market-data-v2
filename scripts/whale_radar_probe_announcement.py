#!/usr/bin/env python3
"""Positive-control probe for the actual MOPS *announcement* corpus.

43-1 filings are announcements (公告查詢 / t146sb10), not ordinary historical
material-information rows (t05st01).  This probe intentionally uses the official
t146sb10 form contract and refuses to green-light the backfill until the known
4194 / 隆欣投資 filing is observed.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import http.cookiejar

HOST = "https://mopsov.twse.com.tw"
PAGE = HOST + "/mops/web/t146sb10"
ACTION = HOST + "/mops/web/ajax_t146sb10"
CONTROL = {"stock_code":"4194","company":"禾生技","actor":"隆欣投資","roc_date":"1151012"}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36"


def fetch(fields):
    jar=http.cookiejar.CookieJar(); op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    h={"User-Agent":UA,"Accept-Language":"zh-TW,zh;q=0.9,en;q=0.6"}
    try: op.open(urllib.request.Request(PAGE,headers=h),timeout=20).read()
    except Exception: pass
    body=urllib.parse.urlencode(fields).encode()
    req=urllib.request.Request(ACTION,data=body,headers={**h,"Content-Type":"application/x-www-form-urlencoded","Referer":PAGE,"Origin":HOST},method="POST")
    with op.open(req,timeout=30) as r: return r.read().decode("utf-8","replace")


def main():
    base={"step":"1","firstin":"ture","off":"1","keyword4":"","code1":"4194","TYPEK2":"","checkbtn":"",
          "queryName":"co_id_1","inpuType":"co_id","scope":"1","co_id_1":"4194","co_id_2":"4194",
          "selecttype":"0","date":"4","yymmdd1":"1151012","yymmdd2":"1151012","sort":"1"}
    diags=[]
    for typek in ("pub","rotc","otc","sii","all",""):
      for notice_date in ("1","2"):
       for notice_kind in ("","0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"):
        f={**base,"typek":typek,"TYPEK":typek,"noticeDate":notice_date,"noticeKind":notice_kind}
        try:
            text=fetch(f); compact=re.sub(r"\s+","",text)
            hit="4194" in text and "隆欣投資" in text and ("43條之1" in compact or "第43條之1" in compact)
            row={"typek":typek,"noticeDate":notice_date,"noticeKind":notice_kind,"bytes":len(text),"has4194":"4194" in text,"hasActor":"隆欣投資" in text,"has43":"43條之1" in compact,"hit":hit}
            diags.append(row); print(json.dumps(row,ensure_ascii=False,sort_keys=True))
            if hit:
                print("POSITIVE_CONTROL_OK",json.dumps({"control":CONTROL,"fields":f},ensure_ascii=False,sort_keys=True)); return 0
        except Exception as e:
            row={"typek":typek,"noticeDate":notice_date,"noticeKind":notice_kind,"error":repr(e),"hit":False}; diags.append(row); print(json.dumps(row,ensure_ascii=False,sort_keys=True))
    print("POSITIVE_CONTROL_FAILED",json.dumps(CONTROL,ensure_ascii=False)); return 3

if __name__=="__main__": raise SystemExit(main())
