@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "CUDA_VISIBLE_DEVICES=-1"
cd /d "%~dp0\..\..\.."
"%~1" -u -m classifier.dino.lora_plus_tta --stage queue --chain-dir "%~2" > "%~2\console.log" 2>&1
set "CHAIN_EXIT=%ERRORLEVEL%"
echo [exit %CHAIN_EXIT%] >> "%~2\console.log"
exit /b %CHAIN_EXIT%
