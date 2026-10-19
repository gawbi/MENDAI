@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "CUDA_VISIBLE_DEVICES=0"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
cd /d "%~dp0\..\..\..\.."
"%~1" -u -m classifier.dino.linear_probe.gpu_head --stage continue --run-dir "%~2" --wait-for-pid %~3 >> "%~2\gpu_head_console.log" 2>&1
set "GPU_HEAD_EXIT=%ERRORLEVEL%"
echo [exit %GPU_HEAD_EXIT%] >> "%~2\gpu_head_console.log"
exit /b %GPU_HEAD_EXIT%
