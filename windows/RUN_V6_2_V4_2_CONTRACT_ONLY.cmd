@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --disable-pip-version-check -r windows\requirements-v6-2-v4-2.txt
if errorlevel 1 goto :fail
python -m unittest -v tests.test_v6_2_v4_2_contract
if errorlevel 1 goto :fail
if not exist "inputs\EVIDENCE_BUNDLE_V2.parquet" goto :missing
if not exist "inputs\consumed\cases_shard_0.jsonl" goto :missing
python scripts\build_v6_2_causal_context_sidecar.py --evidence inputs\EVIDENCE_BUNDLE_V2.parquet --cases inputs\consumed\cases_shard_0.jsonl inputs\consumed\cases_shard_1.jsonl inputs\consumed\cases_shard_2.jsonl inputs\consumed\cases_shard_3.jsonl --out output\causal_context_sidecar.jsonl
if errorlevel 1 goto :fail
python scripts\upgrade_v6_2_v4_2_packets.py --cases inputs\consumed\cases_shard_0.jsonl inputs\consumed\cases_shard_1.jsonl inputs\consumed\cases_shard_2.jsonl inputs\consumed\cases_shard_3.jsonl --sidecar output\causal_context_sidecar.jsonl --out output\v6_2_v4_2_stage_c_packets.jsonl
if errorlevel 1 goto :fail
python scripts\validate_v6_2_v4_2_delivery.py output\v6_2_v4_2_stage_c_packets.jsonl --summary output\CONTRACT_ONLY_SUMMARY.json
if errorlevel 1 goto :fail
echo PASS: output\CONTRACT_ONLY_SUMMARY.json
exit /b 0
:missing
echo FAIL: Put the exact consumed packet JSONL files in inputs\consumed and EVIDENCE_BUNDLE_V2.parquet in inputs.
exit /b 2
:fail
echo FAIL: V6.2 V4.2 contract validation stopped closed.
exit /b 1
