#!/usr/bin/env python3
"""Positive-control probe for Article 43-1 on MOPS announcement search."""
import json, subprocess, tempfile, os, urllib.parse

HOST='https://mopsov.twse.com.tw'
PAGE=HOST+'/mops/web/t146sb10'
URL=HOST+'/mops/web/ajax_t146sb10'
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36'

def request(payload):
    fd,jar=tempfile.mkstemp(prefix='whale-mops-',suffix='.cookie'); os.close(fd)
    try:
        subprocess.run(['curl','-sS','-L','--http1.1','--compressed','--connect-timeout','10','--max-time','30','-A',UA,'-c',jar,'-b',jar,PAGE],stdout=subprocess.DEVNULL,check=False)
        cmd=['curl','-sS','-L','--http1.1','--compressed','--connect-timeout','10','--max-time','30','-A',UA,'-c',jar,'-b',jar,'-H','Content-Type: application/x-www-form-urlencoded','-H','Origin: '+HOST,'-e',PAGE]
        for k,v in payload.items(): cmd += ['--data-urlencode',f'{k}={v}']
        cmd.append(URL)
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        if p.returncode: raise RuntimeError(p.stderr[-500:])
        return p.stdout
    finally:
        try: os.unlink(jar)
        except OSError: pass

def main():
    diagnostics=[]
    # These are the actual visible/JS-assigned t146sb10 fields.  Previous probe
    # omitted scope/selecttype/date/lowercase typek/queryName, so MOPS returned
    # a company shell rather than the bounded announcement list.
    for typek in ('rotc','pub','otc','sii'):
      for selecttype in ('0','1','2'):
       for kind in range(1,61):
        payload={'step':'1','firstin':'ture','off':'1','keyword4':'','code1':'4194','TYPEK2':typek,'checkbtn':'',
          'queryName':'co_id_1','inpuType':'co_id','scope':'1','co_id_1':'4194','co_id_2':'4194',
          'typek':typek,'TYPEK':typek,'selecttype':selecttype,'date':'4','noticeDate':'1',
          'yymmdd1':'1151012','yymmdd2':'1151012','noticeKind':str(kind),'sort':'1'}
        try:
            text=request(payload); compact=''.join(text.split())
            hit=('4194' in compact and '隆欣投資' in compact and ('43條之1' in compact or '第四十三條之一' in compact))
            diagnostics.append({'typek':typek,'selecttype':selecttype,'noticeKind':kind,'bytes':len(text),'hit':hit,'has4194':'4194' in compact,'hasActor':'隆欣投資' in compact,'has43':'43條之1' in compact or '第四十三條之一' in compact})
            if hit:
                print('POSITIVE_CONTROL_OK',json.dumps({'control':CONTROL,'route':URL,'payload':payload,'diagnostics_tail':diagnostics[-10:]},ensure_ascii=False)); return 0
        except Exception as exc: diagnostics.append({'typek':typek,'selecttype':selecttype,'noticeKind':kind,'error':repr(exc)})
    print('POSITIVE_CONTROL_FAILED',json.dumps({'control':CONTROL,'route':URL,'diagnostics_tail':diagnostics[-30:]},ensure_ascii=False)); return 3
if __name__=='__main__': raise SystemExit(main())
