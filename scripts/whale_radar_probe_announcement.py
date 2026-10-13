#!/usr/bin/env python3
"""Positive-control probe for the official MOPS announcement-query form.

The 43-1 disclosure is an *announcement* category, not the material-news subject
search.  MOPS currently exposes that UI at t146sb10 and posts to ajax_t146sb10.
We discover the category id (noticeKind) against the frozen 4194 control rather
than silently trusting an undocumented numeric selector.
"""
import html, json, re, urllib.parse, urllib.request, http.cookiejar

HOST='https://mopsov.twse.com.tw'
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}
UA='Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.5)'

def clean(text): return re.sub(r'\s+','',html.unescape(re.sub(r'<[^>]+>',' ',text)))
def hit(text):
    c=clean(text)
    return '4194' in c and '隆欣投資' in c and ('43條之1' in c or '第四十三條之一' in c) and ('115/10/12' in c or '1151012' in c)
def post(opener, fields):
    data=urllib.parse.urlencode(fields).encode('utf-8')
    req=urllib.request.Request(HOST+'/mops/web/ajax_t146sb10',data=data,headers={'User-Agent':UA,'Accept-Language':'zh-TW,zh;q=0.9','Content-Type':'application/x-www-form-urlencoded','Referer':HOST+'/mops/web/t146sb10'},method='POST')
    with opener.open(req,timeout=30) as r: return r.read().decode('utf-8','replace')

def main():
    jar=http.cookiejar.CookieJar(); opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try: opener.open(urllib.request.Request(HOST+'/mops/web/t146sb10',headers={'User-Agent':UA}),timeout=15).read()
    except Exception: pass
    # Physically verified t146sb10 contract: TYPEK, co_id_1, noticeKind,
    # noticeDate, yymmdd1/yymmdd2.  Probe bounded 2026-10-12 date encodings and
    # noticeKind ids; stop immediately on the known positive control.
    date_variants=(('1','20261012','20261012'),('2','20261012','20261012'),('1','1151012','1151012'),('2','1151012','1151012'))
    for notice_kind in range(1,81):
      for notice_date,d1,d2 in date_variants:
        fields={'encodeURIComponent':'1','step':'1','firstin':'true','TYPEK':'pub','co_id_1':'4194','noticeKind':str(notice_kind),'noticeDate':notice_date,'yymmdd1':d1,'yymmdd2':d2,'order':'1'}
        try:
          text=post(opener,fields); c=clean(text); ok=hit(text)
          if ok:
            out={**CONTROL,'noticeKind':notice_kind,'noticeDate':notice_date,'yymmdd1':d1,'yymmdd2':d2,'bytes':len(text)}
            print('POSITIVE_CONTROL_OK',json.dumps(out,ensure_ascii=False)); return 0
          if notice_kind in (1,10,20,30,40,50,60,70,80):
            print(json.dumps({'noticeKind':notice_kind,'noticeDate':notice_date,'bytes':len(text),'has4194':'4194' in c,'hasActor':'隆欣投資' in c},ensure_ascii=False))
        except Exception as e:
          print(json.dumps({'noticeKind':notice_kind,'noticeDate':notice_date,'error':repr(e)},ensure_ascii=False))
    print('POSITIVE_CONTROL_FAILED',json.dumps(CONTROL,ensure_ascii=False)); return 3
if __name__=='__main__': raise SystemExit(main())
