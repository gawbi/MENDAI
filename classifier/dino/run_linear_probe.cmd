@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "%~dp0\..\.."
"%~1" -u -m classifier.dino.linear_probe --stage extract --run-dir "%~2" > "%~2\console.log" 2>&1
set "PROBE_EXIT=%ERRORLEVEL%"
if not "%PROBE_EXIT%"=="0" goto finished
set "CUDA_VISIBLE_DEVICES=-1"
"%~1" -u -m classifier.dino.linear_probe --stage train --run-dir "%~2" >> "%~2\console.log" 2>&1
set "PROBE_EXIT=%ERRORLEVEL%"
if not "%PROBE_EXIT%"=="0" goto finished
"%~1" -u -m classifier.dino.linear_probe --stage evaluate --run-dir "%~2" >> "%~2\console.log" 2>&1
set "PROBE_EXIT=%ERRORLEVEL%"
:finished
echo [exit %PROBE_EXIT%] >> "%~2\console.log"
exit /b %PROBE_EXIT%
