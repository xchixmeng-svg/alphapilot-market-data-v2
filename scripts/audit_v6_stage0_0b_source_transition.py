#!/usr/bin/env python3
from __future__ import annotations
import json,re
from datetime import datetime,timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v6-stage0-0b-source-transition-audit'; OUT.mkdir(parents=True,exist_ok=True)
DATES=['20190426','20190429','20190718','20201231']
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 AlphaPilot-V6-Stage0-Audit/1.0'})

def norm(x): return re.sub(r'\s+','',str(x or '')).replace('臺','台')
def get_json(url,params):
 r=S.get(url,params=params,timeout=30); r.raise_for_status(); ct=(r.headers.get('content-type') or '').lower()
 if 'json' not in ct: raise RuntimeError(f'non-json content-type={ct}')
 return r.json()
def names_from_mi(j):
 out=[]
 for t in j.get('tables') or []:
  fields=[str(x).strip() for x in (t.get('fields') or [])]; data=t.get('data') or []
  ni=next((i for i,x in enumerate(fields) if x in ('指數','指數名稱') or '指數' in x),None)
  if ni is None: continue
  for row in data:
   if isinstance(row,list) and ni<len(row): out.append(str(row[ni]).strip())
 return sorted(set(out))
def names_from_bf(j):
 out=[]
 for t in j.get('tables') or []:
  fields=[str(x).strip() for x in (t.get('fields') or [])]; data=t.get('data') or []
  ni=next((i for i,x in enumerate(fields) if '分類指數名稱' in x or '指數名稱' in x),None)
  if ni is None: continue
  for row in data:
   if isinstance(row,list) and ni<len(row): out.append(str(row[ni]).strip())
 return sorted(set(out))

ev=[]
for d in DATES:
 rec={'date':d}
 try:
  mi=get_json('https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX',{'response':'json','date':d,'type':'IND'})
  mn=names_from_mi(mi); rec['mi_stat']=mi.get('stat'); rec['mi_name_count']=len(mn); rec['mi_electronic_exact']=any(norm(x)==norm('電子類指數') for x in mn); rec['mi_electronic_like']=[x for x in mn if '電子' in norm(x)][:20]
 except Exception as e: rec['mi_error']=f'{type(e).__name__}: {e}'
 try:
  bf=get_json('https://www.twse.com.tw/rwd/zh/afterTrading/BFIAMU',{'response':'json','date':d})
  bn=names_from_bf(bf); rec['bf_stat']=bf.get('stat'); rec['bf_name_count']=len(bn); rec['bf_electronic_exact']=any(norm(x)==norm('電子類指數') for x in bn); rec['bf_electronic_like']=[x for x in bn if '電子' in norm(x)][:20]
 except Exception as e: rec['bf_error']=f'{type(e).__name__}: {e}'
 ev.append(rec)

post=[x for x in ev if x['date']>'20190426']
mi_post=sum(bool(x.get('mi_electronic_exact')) for x in post); bf_post=sum(bool(x.get('bf_electronic_exact')) for x in post)
if mi_post==len(post): conclusion='MI_INDEX_CONTAINS_TARGET_POST_TRANSITION__CHECK_CHECKPOINT_PARSER_OR_RESPONSE_SCHEMA'
elif bf_post==len(post) and mi_post==0: conclusion='BFIAMU_HAS_TARGET_WHILE_MI_INDEX_DOES_NOT__CROSS_SOURCE_FALLBACK_CANDIDATE_REQUIRES_EQUIVALENCE'
else: conclusion='MIXED__NEEDS_MORE_SOURCE_EQUIVALENCE_EVIDENCE'
progress={'lane':'0B_INDUSTRY_SOURCE_TRANSITION_AUDIT','status':'PASS_DIAGNOSTIC' if conclusion!='MIXED__NEEDS_MORE_SOURCE_EQUIVALENCE_EVIDENCE' else 'BLOCKED','current_completed':len(ev),'target_total':len(DATES),'completion_pct':round(100*len(ev)/len(DATES),2),'newly_completed_this_run':len(ev),'remaining':0,'current_blocker':conclusion,'last_successful_unit':DATES[-1],'artifact_name':'v6-stage0-0b-source-transition-audit','updated_at_utc':datetime.now(timezone.utc).isoformat(),'evidence':ev,'rules':['targeted audit only; no historical backfill','no zero-fill','no gate relaxation','formal OOS remains sealed']}
(OUT/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'summary.md').write_text('# V6 Stage0 0B source-transition audit\n\n'+f"status: {progress['status']}\nconclusion: {conclusion}\nMI post exact: {mi_post}/{len(post)}\nBFIAMU post exact: {bf_post}/{len(post)}\n",encoding='utf-8')
print(json.dumps(progress,ensure_ascii=False))
