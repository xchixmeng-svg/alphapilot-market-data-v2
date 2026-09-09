#!/usr/bin/env python3
"""Run the locked formal R10 engine on audited 2015-2020 inputs.
Only data-binding/evaluation-window substitutions are allowed. No strategy logic is changed.
"""
from pathlib import Path
import hashlib, json, re

ROOT=Path(__file__).resolve().parent.parent
locked=ROOT/'scripts'/'r10_max_formal.py'
src=locked.read_text(encoding='utf-8')
locked_sha=hashlib.sha256(locked.read_bytes()).hexdigest()
EXPECTED_LOCKED='2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0'
if locked_sha!=EXPECTED_LOCKED: raise RuntimeError(f'locked engine changed {locked_sha}')
m=json.loads(Path('input_manifest.json').read_text(encoding='utf-8'))
if m.get('status')!='PASS': raise RuntimeError('input manifest not PASS')
h={'institutional':m['institutional']['sha256']}
for r in m['ohlcv']: h[f"ohlcv_{r['year']}"]=r['sha256']

block=re.compile(r"EXPECTED_INPUT_HASHES = \{.*?\n\}\n\nfor filename, expected in EXPECTED_INPUT_HASHES.items\(\):.*?raise RuntimeError\(f'input SHA mismatch: \{filename\}: \{actual\} != \{expected\}'\)\n",re.S)
rep="EXPECTED_INPUT_HASHES = {\n"+"\n".join([f"    'ohlcv_{y}.parquet': {h[f'ohlcv_{y}']!r}," for y in range(2015,2021)])+f"\n    'institutional_2015_2020.parquet': {h['institutional']!r},\n}}\n\nfor filename, expected in EXPECTED_INPUT_HASHES.items():\n    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()\n    if actual != expected:\n        raise RuntimeError(f'input SHA mismatch: {{filename}}: {{actual}} != {{expected}}')\n"
if not block.search(src): raise RuntimeError('hash block signature changed')
src=block.sub(rep,src,count=1)

# Data-binding names can legitimately occur in multiple places in the locked engine
# (producer + consumer). Replace every occurrence, but only for these audited input/output names.
file_subs=[
("official_corporate_actions_2020_2025.csv","official_corporate_actions_2015_2020.csv"),
("ohlcv_causal_2020_2025.csv.gz","ohlcv_causal_2015_2020.csv.gz"),
("institutional_2020_2025.parquet","institutional_2015_2020.parquet")]
for old,new in file_subs:
    if old not in src: raise RuntimeError('required pure adapter file signature missing: '+old)
    src=src.replace(old,new)

single_subs=[
("for y in range(2020, 2026):","for y in range(2015, 2021):"),
("eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]","eval_dates = [d for d in all_dates if 20160104 <= int(d) <= 20201231]"),
("assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231","assert eval_dates[0] == 20160104 and eval_dates[-1] == 20201231"),
("'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231","'exact_evaluation_window': eval_dates[0] == 20160104 and eval_dates[-1] == 20201231"),
("for year in range(2021, 2026):","for year in range(2016, 2021):")]
for old,new in single_subs:
    if old not in src: raise RuntimeError('required pure adapter signature missing: '+old)
    src=src.replace(old,new,1)

# Prove old data bindings are gone before execution.
for old,_ in file_subs:
    if old in src: raise RuntimeError('stale pure adapter file binding remains: '+old)

# Guard against known-invalid V2/full-exposure mutations.
for forbidden in ["r05_slots_regime", "def dd_multiplier(dd):\n    return 1.0", "r7_exposure = 0.95", "if False and dd <= -FORCE_DD"]:
    if forbidden in src: raise RuntimeError('forbidden strategy mutation detected: '+forbidden)
Path('r10_pure_2016_2020_generated.py').write_text(src,encoding='utf-8')
Path('pure_adapter_audit.json').write_text(json.dumps({'status':'PASS','formal_strategy_modified':False,'locked_engine_sha256':locked_sha,'allowed_changes':['input hashes/files','evaluation window','annual reporting years']},indent=2),encoding='utf-8')
exec(compile(src,'r10_pure_2016_2020_generated.py','exec'),{'__name__':'__main__','__file__':'r10_pure_2016_2020_generated.py'})
