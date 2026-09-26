@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
call windows\RUN_V6_2_V4_2_CONTRACT_ONLY.cmd
if errorlevel 1 exit /b %errorlevel%
where ollama >nul 2>nul
if errorlevel 1 goto :no_ollama
ollama list | findstr /C:"qwen2.5:7b" >nul
if errorlevel 1 goto :no_model
if "%V62_CASE_LIMIT%"=="" set V62_CASE_LIMIT=1
python scripts\run_v6_2_v4_2_stage_c.py output\v6_2_v4_2_stage_c_packets.jsonl --out output\v6_2_v4_2_model_results.jsonl --model qwen2.5:7b --limit %V62_CASE_LIMIT%
if errorlevel 1 goto :fail
echo PASS: output\v6_2_v4_2_model_results.jsonl
exit /b 0
:no_ollama
echo FAIL: Ollama is not installed or not on PATH.
exit /b 3
:no_model
echo FAIL: qwen2.5:7b is missing. Run: ollama pull qwen2.5:7b
exit /b 4
:fail
echo FAIL: Local Stage-C model run stopped closed.
exit /b 1
