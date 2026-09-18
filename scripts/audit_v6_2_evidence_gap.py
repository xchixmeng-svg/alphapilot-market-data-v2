#!/usr/bin/env python3
"""V6.2 evidence-family gap audit.
Development-only inventory: never synthesizes evidence and never reads sealed years.
"""
from pathlib import Path
import hashlib, json, re
import pandas as pd

ROOT=Path('inputs/evidence')
OUT=Path('v6_2_evidence_gap_audit'); OUT.mkdir(parents=True,exist_ok=True)
files=list(ROOT.rglob('EVIDENCE_BUNDLE_V1.parquet'))
assert len(files)==1, f'expected one Evidence Bundle parquet, got {files}'
p=files[0]
df=pd.read_parquet(p)
cols=list(map(str,df.columns)); lc={c.lower():c for c in cols}

def hits(patterns):
    out=[]
    for c in cols:
        x=c.lower()
        if any(re.search(p,x) for p in patterns): out.append(c)
    return sorted(set(out))

families={
 'institutional_flow': hits([r'^inst_',r'foreign.*net',r'trust.*net',r'dealer.*net']),
 'eps_revision': hits([r'eps.*rev',r'rev.*eps',r'earnings.*revision',r'consensus.*eps',r'forecast.*eps']),
 'news_disclosure_text': hits([r'news',r'disclosure',r'announcement',r'material.*event',r'headline',r'text']),
 'industry_pricing_supply_demand': hits([r'industry.*price',r'commodity.*price',r'supply',r'demand',r'freight',r'inventory']),
 'historical_company_industry_pit': hits([
     r'company.*industry', r'industry.*asof', r'industry.*mapping',
     r'sector.*asof', r'sector.*mapping', r'classification.*asof'
 ]),
}
# Presence by column name is inventory only, never source admission.
required=['eps_revision','news_disclosure_text','industry_pricing_supply_demand','historical_company_industry_pit']
missing=[k for k in required if not families[k]]
# hard guards
for dc in ['decision_date','date']:
    if dc in lc:
        s=pd.to_datetime(df[lc[dc]],errors='coerce')
        if s.notna().any():
            assert int(s.dt.year.max())<=2024, f'sealed-year violation in {dc}'
rows=len(df)
report={
 'layer':'V6.2 Evidence Family Gap Audit',
 'scope':'2020-2024 development only',
 'status':'PASS_INVENTORY' if families['institutional_flow'] else 'FAIL',
 'scientific_success':False,
 'note':'Column-name presence is inventory only; every supplemental family still requires independent source-accuracy/PIT admission before use. Global industry_ret1_* context does NOT count as historical company-industry PIT mapping.',
 'rows':rows,'columns':len(cols),
 'bundle_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'families':{k:{'columns':v,'present_by_name':bool(v)} for k,v in families.items()},
 'missing_priority_families':missing,
 'next_action':('build and admit EPS revision source' if 'eps_revision' in missing else 'audit EPS revision source provenance/PIT'),
 'sealed_years_opened':False,
}
(OUT/'EVIDENCE_GAP_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
if report['status']=='FAIL': raise SystemExit(2)
