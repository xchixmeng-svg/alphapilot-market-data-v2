#!/usr/bin/env python3
"""Positive-control probe for MOPS historical material information.

4194 禾生技 is a public/possibly emerging-company case, so probing only the
full-text form is not sufficient.  Probe the actual historical-material-info
endpoint by company across every MOPS market selector and parameter spelling.
"""
import html, json, re, urllib.parse, urllib.request, http.cookiejar

HOSTS = ('https://mopsov.twse.com.tw', 'https://mops.twse.com.tw')
TYPEKS = ('pub', 'rotc', 'otc', 'sii', 'all', '')
CONTROL = {'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}
UA = 'Mozilla/5.0 (compatible; AlphaPilot-WhaleRadar/1.3)'

def clean(text):
    return re.sub(r'\s+', '', html.unescape(re.sub(r'<[^>]+>', ' ', text)))

def request(opener, host, path, fields, method):
    url = host + path
    data = None
    if method == 'GET':
        url += '?' + urllib.parse.urlencode(fields)
    else:
        data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, headers={
        'User-Agent': UA,
        'Accept-Language': 'zh-TW,zh;q=0.9',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': host + '/mops/web/t05st01',
    }, method=method)
    with opener.open(req, timeout=35) as r:
        return r.read().decode('utf-8', 'replace')

def is_hit(text):
    c = clean(text)
    return ('4194' in c and '隆欣投資' in c and
            ('43條之1' in c or '第43條之1' in c or '第四十三條之一' in c) and
            ('115/10/12' in c or '1151012' in c or '2026/10/12' in c or '20261012' in c))

def main():
    for host in HOSTS:
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        for bootstrap in ('/mops/web/t05st01', '/mops/web/t51sb10_q1'):
            try:
                opener.open(urllib.request.Request(host+bootstrap, headers={'User-Agent':UA}), timeout=15).read(4096)
            except Exception:
                pass
        # Historical material-information form: query the known company directly.
        for typek in TYPEKS:
            variants = (
                {'encodeURIComponent':'1','step':'1','firstin':'1','off':'1','TYPEK':typek,'co_id':'4194','year':'115','month':'10','b_date':'01','e_date':'31','queryName':'co_id','inpuType':'co_id','key_word':'','keyword4':'','code1':'','TYPEK2':'','checkbtn':''},
                {'encodeURIComponent':'1','step':'1','firstin':'true','off':'1','TYPEK':typek,'co_id':'4194','year':'115','month':'10','b_date':'01','e_date':'31','queryName':'co_id','inpuType':'co_id'},
                {'encodeURIComponent':'1','step':'1','firstin':'1','off':'1','TYPEK':typek,'co_id':'4194','year':'115','month':'10'},
            )
            for vi, fields in enumerate(variants):
                for method in ('POST','GET'):
                    try:
                        text = request(opener, host, '/mops/web/ajax_t05st01', fields, method)
                        c = clean(text)
                        hit = is_hit(text)
                        print(json.dumps({'endpoint':'t05st01','host':host,'typek':typek,'variant':vi,'method':method,'bytes':len(text),'has4194':'4194' in c,'hasActor':'隆欣投資' in c,'has43':('43條之1' in c or '第43條之1' in c),'hasDate':('115/10/12' in c or '1151012' in c or '20261012' in c),'positive_control':hit}, ensure_ascii=False))
                        if hit:
                            print('POSITIVE_CONTROL_OK', json.dumps({**CONTROL,'endpoint':'ajax_t05st01','host':host,'typek':typek,'variant':vi,'method':method},ensure_ascii=False)); return 0
                    except Exception as e:
                        print(json.dumps({'endpoint':'t05st01','host':host,'typek':typek,'variant':vi,'method':method,'error':repr(e)},ensure_ascii=False))
        # Fallback: old full-text search, retained only as a secondary probe.
        for kind in ('R','L','O','P','C',''):
            fields={'encodeURIComponent':'1','step':'1','firstin':'true','id':'','key':'','TYPEK':'','Stp':'4','go':'false','COMPANY_ID':'4194','co_id':'4194','r1':'1','KIND':kind,'CODE':'4194','keyWord':'43條之1','Condition2':'2','keyWord2':'','year':'115','month1':'10','begin_day':'1','end_day':'31','Orderby':'1'}
            for method in ('POST','GET'):
                try:
                    text=request(opener,host,'/mops/web/ajax_t51sb10',fields,method); c=clean(text); hit=is_hit(text)
                    print(json.dumps({'endpoint':'t51sb10','host':host,'kind':kind,'method':method,'bytes':len(text),'has4194':'4194' in c,'hasActor':'隆欣投資' in c,'has43':'43條之1' in c,'positive_control':hit},ensure_ascii=False))
                    if hit:
                        print('POSITIVE_CONTROL_OK',json.dumps({**CONTROL,'endpoint':'ajax_t51sb10','host':host,'kind':kind,'method':method},ensure_ascii=False)); return 0
                except Exception as e:
                    print(json.dumps({'endpoint':'t51sb10','host':host,'kind':kind,'method':method,'error':repr(e)},ensure_ascii=False))
    print('POSITIVE_CONTROL_FAILED',json.dumps(CONTROL,ensure_ascii=False)); return 3

if __name__=='__main__': raise SystemExit(main())
