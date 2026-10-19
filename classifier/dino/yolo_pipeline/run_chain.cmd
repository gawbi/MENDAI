@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "CUDA_VISIBLE_DEVICES=-1"
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
cd /d "%~dp0\..\..\.."
"%~1" -u -m classifier.dino.yolo_pipeline --stage queue --run-dir "%~2" > "%~2\console.log" 2>&1
set "PIPELINE_EXIT=%ERRORLEVEL%"
echo [exit %PIPELINE_EXIT%] >> "%~2\console.log"
exit /b %PIPELINE_EXIT%
