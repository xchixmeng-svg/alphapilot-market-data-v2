#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/v6-stage0-industry/twse_old_daily'
OUT=ROOT/'artifacts/v6-stage0-0b-coverage-audit'
OUT.mkdir(parents=True,exist_ok=True)
TARGET='電子類指數'
files=sorted(CACHE.glob('*.json'))
completed=[]; target_dates=[]; all_dates=[]
for p in files:
    try: o=json.loads(p.read_text(encoding='utf-8'))
    except Exception: continue
    if o.get('status') not in {'SUCCESS','VERIFIED_NO_DATA'}: continue
    completed.append(o)
    if o.get('status')=='SUCCESS':
        d=str(o.get('date'))
        all_dates.append(d)
        if any(r.get('index_name')==TARGET for r in o.get('rows',[])): target_dates.append(d)
missing=sorted(set(all_dates)-set(target_dates))
by_year=Counter(x[:4] for x in missing)
present_by_year=Counter(x[:4] for x in target_dates)
progress={
 'lane':'0B_INDUSTRY_COVERAGE_AUDIT','status':'BLOCKED' if missing else 'PASS',
 'current_completed':len(target_dates),'target_total':len(all_dates),
 'completion_pct':round(100*len(target_dates)/len(all_dates),4) if all_dates else 'n/a',
 'newly_completed_this_run':0,'remaining':len(missing),
 'current_blocker':f'{TARGET} absent on {len(missing)} successful TWSE trading-date checkpoints' if missing else None,
 'last_successful_unit':max(target_dates) if target_dates else None,
 'artifact_name':'v6-stage0-0b-coverage-audit','updated_at_utc':datetime.now(timezone.utc).isoformat(),
 'checkpoint_files':len(files),'successful_trading_dates':len(all_dates),
 'missing_by_year':dict(sorted(by_year.items())),'present_by_year':dict(sorted(present_by_year.items())),
 'missing_date_sample':missing[:100],
 'diagnostic_rule':'No refetch and no zero-fill. Diagnose only immutable SUCCESS/VERIFIED_NO_DATA checkpoint cache.'
}
(OUT/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
summary='\n'.join([
 '# V6 Stage0 0B coverage blocker audit',
 f"status: {progress['status']}",f"target: {TARGET}",
 f"present: {len(target_dates)}/{len(all_dates)} ({progress['completion_pct']}%)",
 f"missing: {len(missing)}",f"missing_by_year: {dict(sorted(by_year.items()))}",
 f"last_successful_unit: {progress['last_successful_unit']}",
 'No historical refetch; no zero-fill; no gate relaxation.'
])+'\n'
(OUT/'summary.md').write_text(summary,encoding='utf-8')
print(json.dumps(progress,ensure_ascii=False))
print(summary)
