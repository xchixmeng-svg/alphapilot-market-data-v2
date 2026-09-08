#!/usr/bin/env python3
"""Run the existing YTD builder with a paced, retrying TWSE institutional fetch."""
from pathlib import Path

path=Path(__file__).with_name('build_2026_ytd_package.py')
src=path.read_text(encoding='utf-8')
src=src.replace(
    'import csv, io, json, re, time, zipfile, hashlib',
    'import csv, io, json, random, re, time, zipfile, hashlib'
)
old='''def fetch_all(fn,market):
    rows=[]; fails=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(fn,ds):ds for ds in trade_dates}
        for i,f in enumerate(as_completed(fut),1):
            ds=fut[f]
            try: rows.extend(f.result())
            except Exception as e: fails.append({'date':ds,'market':market,'error':str(e)})
            if i%20==0 or i==len(fut): print(f'[{market} INST]',i,'/',len(fut),'rows',len(rows),'failures',len(fails),flush=True)
    return rows,fails

twse,f1=fetch_all(twse_t86,'TWSE')
tpex,f2=fetch_all(tpex_daily,'TPEX')
'''
new='''def fetch_parallel(fn,market):
    rows=[]; fails=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut={ex.submit(fn,ds):ds for ds in trade_dates}
        for i,f in enumerate(as_completed(fut),1):
            ds=fut[f]
            try: rows.extend(f.result())
            except Exception as e: fails.append({'date':ds,'market':market,'error':str(e)})
            if i%20==0 or i==len(fut): print(f'[{market} INST]',i,'/',len(fut),'rows',len(rows),'failures',len(fails),flush=True)
    return rows,fails

def fetch_twse_resilient():
    by_date={}; errors={}; pending=list(trade_dates)
    for round_no in range(3):
        if round_no:
            cool=90*round_no
            print(f'[TWSE INST] repair round {round_no+1}; cooldown={cool}s pending={len(pending)}',flush=True)
            time.sleep(cool)
        retry=[]
        for i,ds in enumerate(pending,1):
            try:
                by_date[ds]=twse_t86(ds); errors.pop(ds,None)
            except Exception as e:
                retry.append(ds); errors[ds]=str(e)
            time.sleep(1.8+random.uniform(.1,.7))
            if i%30==0 or i==len(pending):
                print(f'[TWSE INST] round={round_no+1} {i}/{len(pending)} complete={len(by_date)} retry={len(retry)}',flush=True)
                if i%30==0: time.sleep(15)
        pending=retry
        if not pending: break
    return [r for ds in sorted(by_date) for r in by_date[ds]], [{'date':ds,'market':'TWSE','error':errors[ds]} for ds in pending]

twse,f1=fetch_twse_resilient()
tpex,f2=fetch_parallel(tpex_daily,'TPEX')
'''
if old not in src:
    raise RuntimeError('upstream builder changed; refusing unsafe patch')
src=src.replace(old,new).replace(
    "if min(cov.values())<0.95:",
    "if min(cov.values())<0.98:"
)
exec(compile(src,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
