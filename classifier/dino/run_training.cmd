@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
cd /d "%~dp0\..\.."
"%~1" -m classifier.dino.train --stage train --config "%~2\config.json" --run-dir "%~2" --backbone-path "%~3" > "%~2\console.log" 2>&1
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" goto training_failed
"%~1" -m classifier.dino.evaluate --run-dir "%~2" >> "%~2\evaluation_raw.log" 2>&1
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" goto training_failed
"%~1" classifier\dino\tools\report_combined.py --run-dir "%~2" >> "%~2\console.log" 2>&1
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" goto training_failed
if "%~4"=="" goto training_failed
"%~1" -m classifier.dino.tools.compare_lora --run-dir "%~2" --baseline-run "%~4" >> "%~2\console.log" 2>&1
set "TRAIN_EXIT=%ERRORLEVEL%"
:training_failed
echo [exit %TRAIN_EXIT%] >> "%~2\console.log"
exit /b %TRAIN_EXIT%
