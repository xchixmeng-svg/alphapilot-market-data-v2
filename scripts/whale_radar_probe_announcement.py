#!/usr/bin/env python3
"""Bounded positive-control probe for Article 43-1 on MOPS announcement search.

Never brute-force hundreds of requests: MOPS can be slow from GitHub runners.
The control must be found before historical zeroes are trusted.
"""
import json, subprocess, tempfile, os, time

HOST='https://mopsov.twse.com.tw'
PAGE=HOST+'/mops/web/t146sb10'
URL=HOST+'/mops/web/ajax_t146sb10'
CONTROL={'stock_code':'4194','company':'禾生技','actor':'隆欣投資','roc_date':'115/10/12'}
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36'

def request(payload, jar):
    cmd=['curl','-sS','-L','--http1.1','--compressed','--connect-timeout','5','--max-time','12','--retry','1','--retry-delay','1','-A',UA,'-c',jar,'-b',jar,'-H','Content-Type: application/x-www-form-urlencoded','-H','Origin: '+HOST,'-e',PAGE]
    for k,v in payload.items(): cmd += ['--data-urlencode',f'{k}={v}']
    cmd.append(URL)
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=28)
    if p.returncode: raise RuntimeError((p.stderr or '')[-300:])
    return p.stdout

def main():
    diagnostics=[]
    fd,jar=tempfile.mkstemp(prefix='whale-mops-',suffix='.cookie'); os.close(fd)
    try:
        # One warm-up only. Failure is non-fatal; POST may still work.
        subprocess.run(['curl','-sS','-L','--http1.1','--connect-timeout','5','--max-time','10','-A',UA,'-c',jar,'-b',jar,PAGE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,timeout=15)
        # Blank noticeKind is intentional: bounded company+date query should return
        # all announcement categories and avoids guessing undocumented category ids.
        variants=[]
        for typek in ('rotc','pub','otc','sii'):
            for selecttype in ('0','1','2'):
                variants.append({'step':'1','firstin':'true','off':'1','keyword4':'','code1':'4194','TYPEK2':typek,'checkbtn':'','queryName':'co_id_1','inpuType':'co_id','scope':'1','co_id_1':'4194','co_id_2':'4194','typek':typek,'TYPEK':typek,'selecttype':selecttype,'date':'4','noticeDate':'1','yymmdd1':'1151012','yymmdd2':'1151012','noticeKind':'','sort':'1'})
        for payload in variants:
            try:
                text=request(payload,jar); compact=''.join(text.split())
                row={'typek':payload['typek'],'selecttype':payload['selecttype'],'bytes':len(text),'has4194':'4194' in compact,'hasActor':'隆欣投資' in compact,'has43':('43條之1' in compact or '第四十三條之一' in compact)}
                row['hit']=row['has4194'] and row['hasActor'] and row['has43']; diagnostics.append(row); print('PROBE',json.dumps(row,ensure_ascii=False),flush=True)
                if row['hit']:
                    print('POSITIVE_CONTROL_OK',json.dumps({'control':CONTROL,'route':URL,'payload':payload},ensure_ascii=False)); return 0
            except Exception as exc:
                diagnostics.append({'typek':payload['typek'],'selecttype':payload['selecttype'],'error':repr(exc)}); print('PROBE_ERROR',repr(exc),flush=True)
            time.sleep(.25)
    finally:
        try: os.unlink(jar)
        except OSError: pass
    print('POSITIVE_CONTROL_FAILED',json.dumps({'control':CONTROL,'route':URL,'diagnostics':diagnostics},ensure_ascii=False)); return 3
if __name__=='__main__': raise SystemExit(main())
