#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('v6_2_evidence_bundle_v1'); ROOT.mkdir(exist_ok=True)
FROZEN=Path('inputs/0f/0F_FINAL_ASSEMBLY/decision_ticker_panel.parquet')
TWSE=Path('inputs/institutional/TWSE_INSTITUTIONAL_PIT.parquet')
TPEX=Path('inputs/institutional/TPEX_INSTITUTIONAL_PIT.parquet')
ADMISSION=Path('inputs/admission/INSTITUTIONAL_SUPPLEMENTAL_ADMISSION.json')
for p in [FROZEN,TWSE,TPEX,ADMISSION]:
    if not p.exists(): raise FileNotFoundError(p)
adm=json.loads(ADMISSION.read_text())
if adm.get('status')!='PASS' or adm.get('future_join_violations')!=0 or adm.get('duplicate_key_violations')!=0:
    raise RuntimeError('institutional source not admitted')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm_code(s): return s.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
sp=pd.read_parquet(FROZEN)
date_col='decision_date' if 'decision_date' in sp.columns else 'date'
code_col='code' if 'code' in sp.columns else ('ticker' if 'ticker' in sp.columns else None)
if code_col is None: raise RuntimeError('0F missing code/ticker')
sp[date_col]=pd.to_datetime(sp[date_col]); sp[code_col]=norm_code(sp[code_col])
if sp[date_col].dt.year.max()>2024: raise RuntimeError('sealed/live year exposure in 0F input')
if sp.duplicated([date_col,code_col]).any(): raise RuntimeError('duplicate 0F decision key')
parts=[]
for market,p in [('TWSE',TWSE),('TPEX',TPEX)]:
    x=pd.read_parquet(p).copy(); x['code']=norm_code(x['code']); x['available_session']=pd.to_datetime(x['available_session']); x['date']=pd.to_datetime(x['date']); x['market']=market
    if x.duplicated(['date','code']).any(): raise RuntimeError(f'duplicate source key {market}')
    if (x['available_session']<=x['date']).any(): raise RuntimeError(f'same-day visibility {market}')
    parts.append(x)
inst=pd.concat(parts,ignore_index=True)
# Join only exact next-session availability. This is stricter than as-of carry-forward and prevents accidental future joins.
keys=sp[[date_col,code_col]].copy(); keys['_rowid']=np.arange(len(keys)); keys=keys.rename(columns={date_col:'decision_date',code_col:'code'})
j=keys.merge(inst,left_on=['decision_date','code'],right_on=['available_session','code'],how='left',suffixes=('','_inst'))
if len(j)!=len(keys): raise RuntimeError('institutional join multiplied spine rows')
future=((j['available_session'].notna()) & (j['available_session']>j['decision_date'])).sum()
if future: raise RuntimeError(f'future_join_violations={future}')
cols=['foreign_net_ratio','trust_net_ratio','dealer_net_ratio','foreign_net','trust_net','dealer_net','volume','market','date','available_session']
ren={c:'inst_'+c for c in cols if c in j.columns}
j=j[['_rowid']+list(ren)].rename(columns=ren)
out=sp.copy(); out['_rowid']=np.arange(len(out)); out=out.merge(j,on='_rowid',how='left',validate='one_to_one').drop(columns='_rowid')
out_path=ROOT/'EVIDENCE_BUNDLE_V1.parquet'; out.to_parquet(out_path,index=False)
coverage=int(out['inst_available_session'].notna().sum()) if 'inst_available_session' in out else 0
manifest={
 'layer':'V6.2 Evidence Bundle v1','scope':'2020-2024 development only','status':'PASS',
 'spine_policy':'exact frozen 0F bytes; supplemental columns only','join_policy':'institutional source row admitted only on available_session (next trading session); no same-day use',
 'rows':len(out),'spine_rows':len(sp),'institutional_matched_rows':coverage,'institutional_missing_rows':len(out)-coverage,
 'future_join_violations':int(future),'duplicate_output_key_violations':int(out.duplicated([date_col,code_col]).sum()),
 'inputs':{'frozen_0f_sha256':sha(FROZEN),'twse_sha256':sha(TWSE),'tpex_sha256':sha(TPEX),'admission_sha256':sha(ADMISSION)},
 'output_sha256':sha(out_path),
 'missingness':{c:int(out[c].isna().sum()) for c in out.columns if c.startswith('inst_')}
}
if manifest['duplicate_output_key_violations']!=0: raise RuntimeError('duplicate output key')
(ROOT/'EVIDENCE_BUNDLE_V1_AUDIT.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps(manifest,ensure_ascii=False))
