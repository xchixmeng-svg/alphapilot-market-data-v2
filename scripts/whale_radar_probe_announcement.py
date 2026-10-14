#!/usr/bin/env python3
"""Positive-control probe for Article 43-1 on the MOPS *announcement* route.

43-1 acquisition filings are announcements (公告), not ordinary t05st01 material news.
The known control is 4194 禾生技 / 隆欣投資 / 115-10-12.  Probe the official
legacy t146sb10 form contract and discover the noticeKind that surfaces it.
"""
import json
import urllib.parse
import urllib.request

URL = 'https://mopsov.twse.com.tw/mops/web/ajax_t146sb10'
CONTROL = {'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}
UA = 'Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.2)'


def post(payload):
    body = urllib.parse.urlencode(payload).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={
        'Content-Type':'application/x-www-form-urlencoded',
        'Accept':'text/html,application/xhtml+xml',
        'User-Agent':UA,
        'Referer':'https://mopsov.twse.com.tw/mops/web/t146sb10',
    }, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', 'replace')


def main():
    diagnostics=[]
    # Official t146sb10 form fields are TYPEK, co_id_1, noticeKind,
    # noticeDate, yymmdd1, yymmdd2.  4194 is emerging (rotc).
    # noticeKind has changed across MOPS revisions, so discover it from
    # a bounded known-positive query instead of hard-coding an unverified value.
    for kind in range(1, 31):
        for notice_date, d1, d2 in [
            ('1','1151012','1151012'), ('2','1151012','1151012'),
            ('1','20261012','20261012'), ('2','20261012','20261012'),
        ]:
            payload={
                'firstin':'true','step':'1','TYPEK':'rotc','co_id_1':'4194','co_id_2':'4194',
                'noticeKind':str(kind),'noticeDate':notice_date,'yymmdd1':d1,'yymmdd2':d2,
                'date1':d1,'date2':d2,'sort':'1','num':'100',
            }
            try:
                text=post(payload)
                compact=''.join(text.split())
                hit=('4194' in compact and '隆欣投資' in compact and
                     ('43條之1' in compact or '第四十三條之一' in compact))
                diagnostics.append({'noticeKind':kind,'noticeDate':notice_date,'d1':d1,
                                    'bytes':len(text),'hit':hit,
                                    'has4194':'4194' in compact,'hasActor':'隆欣投資' in compact,
                                    'has43':'43條之1' in compact or '第四十三條之一' in compact})
                if hit:
                    print('POSITIVE_CONTROL_OK', json.dumps({
                        'control':CONTROL,'route':URL,'payload':payload,
                        'diagnostics_tail':diagnostics[-8:]
                    },ensure_ascii=False))
                    return 0
            except Exception as exc:
                diagnostics.append({'noticeKind':kind,'noticeDate':notice_date,'d1':d1,'error':repr(exc)})
    print('POSITIVE_CONTROL_FAILED', json.dumps({'control':CONTROL,'route':URL,
          'diagnostics_tail':diagnostics[-20:]},ensure_ascii=False))
    return 3

if __name__=='__main__':
    raise SystemExit(main())
