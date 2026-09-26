@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --disable-pip-version-check -r windows\requirements-v6-2-v4-2.txt
if errorlevel 1 goto :fail
if not exist "output\v6_2_v4_2_stage_c_packets.jsonl" goto :missing
python scripts\validate_v6_2_v4_2_delivery.py output\v6_2_v4_2_stage_c_packets.jsonl --summary output\CONTRACT_ONLY_SUMMARY.json
if errorlevel 1 goto :fail
if "%V62_TRANSPORT_ONLY%"=="1" goto :transport
where ollama >nul 2>nul
if errorlevel 1 goto :no_ollama
ollama list | findstr /C:"qwen2.5:7b" >nul
if errorlevel 1 goto :no_model
if "%V62_CASE_LIMIT%"=="" set V62_CASE_LIMIT=1
python scripts\run_v6_2_v4_2_stage_c.py output\v6_2_v4_2_stage_c_packets.jsonl --out output\v6_2_v4_2_model_results.jsonl --model qwen2.5:7b --limit %V62_CASE_LIMIT%
if errorlevel 1 goto :fail
echo PASS: output\v6_2_v4_2_model_results.jsonl
exit /b 0
:transport
python scripts\run_v6_2_v4_2_stage_c.py output\v6_2_v4_2_stage_c_packets.jsonl --out output\STAGE_C_TRANSPORT_RECEIPTS.jsonl --transport-only
if errorlevel 1 goto :fail
python scripts\benchmark_v6_2_v4_2_transport.py output\STAGE_C_TRANSPORT_RECEIPTS.jsonl --expected 64 --out output\TRANSPORT_BENCHMARK_SUMMARY.json
if errorlevel 1 goto :fail
echo PASS: packaged Stage-C transport is runnable on Windows.
exit /b 0
:missing
echo FAIL: output\v6_2_v4_2_stage_c_packets.jsonl is missing.
exit /b 2
:no_ollama
echo FAIL: Ollama is not installed or not on PATH.
exit /b 3
:no_model
echo FAIL: qwen2.5:7b is missing. Run: ollama pull qwen2.5:7b
exit /b 4
:fail
echo FAIL: Local Stage-C model run stopped closed.
exit /b 1
