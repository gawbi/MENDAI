@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "CUDA_VISIBLE_DEVICES=0"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "PAIR_PYTHON=%~1"
cd /d "%~dp0\..\..\.."
call :run_one "%~2"
if errorlevel 1 goto failed
call :run_one "%~3"
if errorlevel 1 goto failed
"%~1" -u -m classifier.dino.normalized_probe --stage compare --runs "%~2" "%~3" --output "%~4" >> "%~4\console.log" 2>&1
set "PAIR_EXIT=%ERRORLEVEL%"
goto finished
:failed
set "PAIR_EXIT=%ERRORLEVEL%"
:finished
echo [exit %PAIR_EXIT%] >> "%~4\console.log"
exit /b %PAIR_EXIT%
:run_one
"%PAIR_PYTHON%" -u -m classifier.dino.normalized_probe --stage extract --run-dir "%~1" >> "%~1\console.log" 2>&1
if errorlevel 1 exit /b 1
"%PAIR_PYTHON%" -u -m classifier.dino.normalized_probe --stage train --run-dir "%~1" >> "%~1\console.log" 2>&1
if errorlevel 1 exit /b 1
"%PAIR_PYTHON%" -u -m classifier.dino.normalized_probe --stage evaluate --run-dir "%~1" >> "%~1\console.log" 2>&1
exit /b %ERRORLEVEL%
