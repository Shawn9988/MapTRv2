@echo off
setlocal

set "DATA_ROOT=data\argoverse2\sensor"

set "DOWNLOAD_TRAIN=0"
set "DOWNLOAD_VAL=0"
set "DOWNLOAD_TEST=1"

set "TRAIN_LOG=00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"
set "VAL_LOG=070bbf42-31d3-3aa9-aca4-c262afc9077d"
set "TEST_LOG=0c6e62d7-bdfa-3061-8d3d-03b13aa21f68"

where s5cmd >nul 2>nul
if errorlevel 1 (
    echo [ERROR] s5cmd not found in PATH.
    exit /b 1
)

if "%DOWNLOAD_TRAIN%"=="1" call :download_split train "%TRAIN_LOG%"
if "%DOWNLOAD_VAL%"=="1" call :download_split val "%VAL_LOG%"
if "%DOWNLOAD_TEST%"=="1" call :download_split test "%TEST_LOG%"

echo [DONE] Requested downloads finished.
endlocal
exit /b 0

:download_split
set "SPLIT=%~1"
set "LOG_ID=%~2"
set "SRC=s3://argoverse/datasets/av2/sensor/%SPLIT%/%LOG_ID%"
set "DST=%DATA_ROOT%\%SPLIT%\%LOG_ID%"

echo [INFO] %SPLIT% %LOG_ID%
s5cmd --no-sign-request du -H "%SRC%/*"
mkdir "%DST%" 2>nul
s5cmd --no-sign-request cp "%SRC%/*" "%DST%"
if errorlevel 1 (
    echo [ERROR] Failed: %SPLIT% %LOG_ID%
    exit /b 1
)
exit /b 0
