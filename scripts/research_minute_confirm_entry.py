"""Research-only minute-confirmed T+1 entry discovery.

This is a distinct entry family, not a modification of locked R10-MAX. Candidate intent
is fixed at T close from the frozen formal BUY orders. On T+1 we use only completed
1-minute bars before a checkpoint, then model entry at the NEXT minute open. No same-bar
lookahead. This stage is diagnostic: it tests whether causal intraday confirmation can
improve trade quality before any full common-cash portfolio promotion.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

OUT=Path('research_out_minute_confirm'); OUT.mkdir(exist_ok=True)
ORD=Path('formal_run/r10max_formal_orders.csv'); TRD=Path('formal_run/r10max_formal_trades.csv'); DAILY=Path('formal_run/ohlcv_causal_2020_2025.csv.gz')
for p in (ORD,TRD,DAILY):
    if not p.exists(): raise FileNotFoundError(p)
START=20230523; END=20251231; DEV_END=20240630
CHECKPOINTS=[10,15,30]
API_KEY=os.getenv('FUGLE_API_KEY','').strip(); BASE='https://api.fugle.tw/marketdata/v1.0/stock/historical/candles'
SPACING=1.25; BACKOFF=65.0
if not API_KEY: raise RuntimeError('FUGLE_API_KEY missing')

orders=pd.read_csv(ORD,dtype={'code':str}); trades=pd.read_csv(TRD,dtype={'code':str}); daily=pd.read_csv(DAILY,dtype={'code':str},low_memory=False)
for x in (orders,trades,daily): x['code']=x.code.astype(str).str.zfill(4)
b=orders[(orders.side=='BUY')&(orders.status=='FILLED')].copy()
for c in ('scheduled_date','signal_date','fill_date'): b[c]=b[c].astype(int)
b=b[(b.scheduled_date>=START)&(b.scheduled_date<=END)]
trades['entry_date']=trades.entry_date.astype(int); trades['exit_date']=trades.exit_date.astype(int)
keep=['code','strategy','entry_date','exit_date','exit_raw_price','return','reason']
b=b.merge(trades[keep],left_on=['code','strategy','fill_date'],right_on=['code','strategy','entry_date'],how='left',validate='many_to_one')
b=b[b.exit_date.notna()].copy()
ref=daily[['date','code','close']].copy(); ref.date=ref.date.astype(int); ref=ref.rename(columns={'date':'signal_date','close':'signal_close'})
b=b.merge(ref,on=['signal_date','code'],how='left',validate='many_to_one')

sess=requests.Session(); sess.headers.update({'X-API-KEY':API_KEY}); cache={}; errors=[]
def fetch(code,ymd):
    key=(code,int(ymd))
    if key in cache:return cache[key]
    d=pd.to_datetime(str(int(ymd))).strftime('%Y-%m-%d'); last=None
    for a in range(5):
        r=sess.get(f'{BASE}/{code}',params={'timeframe':'1','from':d,'to':d,'fields':'open,high,low,close,volume','sort':'asc'},timeout=30)
        last=(r.status_code,r.text[:240])
        if r.status_code==200:
            z=pd.DataFrame(r.json().get('data',[]))
            if z.empty: raise RuntimeError(f'empty minute data {code} {ymd}')
            z['ts']=pd.to_datetime(z.date,errors='coerce',utc=True).dt.tz_convert('Asia/Taipei'); z['minute']=z.ts.dt.strftime('%H:%M:%S')
            for c in ('open','high','low','close','volume'):z[c]=pd.to_numeric(z[c],errors='coerce')
            z=z.sort_values('ts').reset_index(drop=True);cache[key]=z;time.sleep(SPACING);return z
        if r.status_code in (401,402,403,404):raise RuntimeError(f'Fugle {r.status_code} {code} {ymd}: {r.text[:240]}')
        if r.status_code==429:time.sleep(BACKOFF+5*a)
        else:time.sleep(2+2*a)
    raise RuntimeError(f'Fugle retries exhausted {code} {ymd}: {last}')

def features(code,ymd,base,m):
    kb=fetch(code,ymd); cp=f'09:{m:02d}:00'; ent=f'09:{m+1:02d}:00'
    pre=kb[(kb.minute>='09:00:00')&(kb.minute<cp)]; nxt=kb[kb.minute==ent]
    if pre.empty or nxt.empty:return None
    op=float(pre.iloc[0].open); cl=float(pre.iloc[-1].close); lo=float(pre.low.min()); hi=float(pre.high.max()); vol=float(pre.volume.sum())
    vwap=float((pre.close*pre.volume).sum()/vol) if vol>0 else cl
    mk=fetch('0050',ymd); mpre=mk[(mk.minute>='09:00:00')&(mk.minute<cp)]
    mret=float(mpre.iloc[-1].close/mk.iloc[0].open-1) if len(mpre) else np.nan
    return {'checkpoint':m,'entry_time':ent,'entry_price':float(nxt.iloc[0].open),'gap':op/base-1,'ret':cl/base-1,'low_ret':lo/base-1,'high_ret':hi/base-1,'above_vwap':cl>=vwap,'above_open':cl>=op,'recovery':(cl-lo)/max(hi-lo,1e-9),'mkt_ret':mret,'rel_open':cl/op-1-mret if np.isfinite(mret) else np.nan}

rows=[]
for _,r in b.sort_values(['scheduled_date','code']).iterrows():
    try:
        base=float(r.signal_close)
        for m in CHECKPOINTS:
            f=features(r.code,int(r.scheduled_date),base,m)
            if f is None:continue
            # predeclared economically distinct confirmations
            policies={
              'trend_confirm': f['ret']>0 and f['above_vwap'] and f['above_open'] and f['rel_open']>=0,
              'pullback_reclaim': f['low_ret']<=-0.01 and f['ret']>=-0.002 and f['above_vwap'] and f['recovery']>=0.65 and f['rel_open']>=0,
              'controlled_gap_hold': -0.02<=f['gap']<=0.03 and f['ret']>=0 and f['above_vwap'] and f['rel_open']>=0,
              'relative_strength': f['rel_open']>=0.006 and f['above_vwap'] and f['recovery']>=0.50,
            }
            for p,ok in policies.items():
                if not ok:continue
                ep=f['entry_price']; gross=float(r.exit_raw_price)/ep-1
                # approximate round-trip costs: 0.0855% each side + 0.3% sell tax
                net=(float(r.exit_raw_price)*(1-0.000855-0.003))/(ep*(1+0.000855))-1
                rows.append({'code':r.code,'strategy':r.strategy,'date':int(r.scheduled_date),'policy':p,'checkpoint':m,'entry_price':ep,'exit_price':float(r.exit_raw_price),'return_net':net,'return_gross':gross,'original_return':float(r['return']),**{k:v for k,v in f.items() if k not in ('entry_price','checkpoint')}})
    except Exception as e:
        errors.append({'code':r.code,'date':int(r.scheduled_date),'error':repr(e)})
        if '401' in repr(e) or '402' in repr(e) or '403' in repr(e):break

res=pd.DataFrame(rows); res.to_csv(OUT/'confirmed_trades.csv',index=False)
summary=[]
if len(res):
  for (p,m),g in res.groupby(['policy','checkpoint']):
    for split,sg in [('all',g),('dev',g[g.date<=DEV_END]),('holdout',g[g.date>DEV_END])]:
      if len(sg)==0:continue
      summary.append({'policy':p,'checkpoint':m,'split':split,'n':len(sg),'wins':int((sg.return_net>0).sum()),'win_rate':float((sg.return_net>0).mean()),'mean_return':float(sg.return_net.mean()),'median_return':float(sg.return_net.median()),'worst_return':float(sg.return_net.min()),'sum_return':float(sg.return_net.sum())})
s=pd.DataFrame(summary); s.to_csv(OUT/'summary.csv',index=False)
# Promotion requires non-tiny holdout, >=70% WR in both dev/holdout, positive mean in both, and neighboring checkpoint support.
prom=[]
for p in sorted(res.policy.unique()) if len(res) else []:
  q=s[(s.policy==p)&(s.split.isin(['dev','holdout']))]
  good=[]
  for m in CHECKPOINTS:
    d=q[(q.checkpoint==m)&(q.split=='dev')]; h=q[(q.checkpoint==m)&(q.split=='holdout')]
    ok=len(d)==1 and len(h)==1 and int(d.iloc[0].n)>=10 and int(h.iloc[0].n)>=10 and float(d.iloc[0].win_rate)>=0.70 and float(h.iloc[0].win_rate)>=0.70 and float(d.iloc[0].mean_return)>0 and float(h.iloc[0].mean_return)>0
    if ok:good.append(m)
  prom.append({'policy':p,'good_checkpoints':good,'neighbor_stable': any(abs(a-b)<=5 for a in good for b in good if a!=b),'promote':len(good)>=2 and any(abs(a-b)<=5 for a in good for b in good if a!=b)})
payload={'status':'PASS' if not errors else 'PASS_WITH_ERRORS','study_start':START,'study_end':END,'dev_end':DEV_END,'completed_formal_buys':int(len(b)),'minute_pairs_cached':len(cache),'errors':errors[:20],'promotion':prom,'note':'Discovery diagnostic only; promotion requires later full common-cash portfolio simulation.'}
(OUT/'decision.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2)); print(s.sort_values(['split','win_rate','mean_return'],ascending=[True,False,False]).to_string(index=False) if len(s) else 'NO_CONFIRMED_TRADES')
