#!/usr/bin/env python3
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent; YTD=ROOT/'data'/'history'/'2026-YTD'
d=pd.read_csv(YTD/'ohlcv_2026_ytd.csv',dtype={'code':str}); d['close']=pd.to_numeric(d.close,errors='coerce'); d['volume']=pd.to_numeric(d.volume,errors='coerce')
d=d.sort_values(['code','date']); d['ret']=d.groupby('code').close.pct_change()
rows=[]
for day,g in d.groupby('date'):
    valid=g.ret.dropna(); etf=g[g.code=='0050']; r0050=None if etf.empty or pd.isna(etf.ret.iloc[-1]) else float(etf.ret.iloc[-1])
    rows.append({'date':day,'return_0050':r0050,'advance_ratio':float((valid>0).mean()) if len(valid) else None,'decline_5pct_ratio':float((valid<=-.05).mean()) if len(valid) else None,'near_limit_down_ratio':float((valid<=-.085).mean()) if len(valid) else None,'market_volume':float(g.volume.fillna(0).sum()),'symbols':int(g.code.nunique())})
out=YTD/'market_risk_features_2026_ytd.csv'; pd.DataFrame(rows).to_csv(out,index=False)
manifest={'dataset':'AlphaPilot 2026 YTD causal market-risk features','generated_at_utc':datetime.now(timezone.utc).isoformat(),'rows':len(rows),'date_start':rows[0]['date'],'date_end':rows[-1]['date'],'inputs':['ohlcv_2026_ytd.csv'],'fields':['0050 daily return','advance ratio','<=-5% ratio','near-limit-down ratio','market volume'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(YTD/'market_risk_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False),flush=True)
