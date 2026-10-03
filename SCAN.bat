@echo off
rem ============================================================
rem  Prism Lab - 3D scan, one click, nothing to type.
rem
rem  Drag a photo folder onto this file, or double-click and it
rem  uses the newest capture folder on the Desktop.
rem  Results go to <folder>\_scan\
rem
rem  Everything is decided from the data: the SAM prompt, the
rem  filters, the cleanup. See scan_run.py for the measurements
rem  behind each default.
rem
rem  Delayed expansion is REQUIRED - prompts may contain '|',
rem  and %VAR% expansion makes cmd read that as a pipe and die.
rem  (ASCII only on purpose - Korean text lives in Python.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Prism Lab - 3D scan
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "VPY=%USERPROFILE%\vggt_server\venv\Scripts\python.exe"
if not exist "!VPY!" (
    echo   [x] venv python not found:
    echo       !VPY!
    echo       Run VGGT-Server.bat once to install it.
    goto done
)

echo.
echo ============================================================
echo   Prism Lab  -  photos to point cloud / mesh / STL
echo ============================================================
echo.

rem ---- server must be up ----------------------------------------
"!VPY!" -c "import requests,sys; sys.exit(0 if requests.get('http://127.0.0.1:5000/health',timeout=5).ok else 1)" >nul 2>&1
if errorlevel 1 (
    echo   VGGT server is not running. Starting it now...
    start "VGGT server" /min "%~dp0VGGT-Server.bat"
    echo   Waiting for the model to load ^(about 60 seconds^)...
    "!VPY!" "%~dp0_wait_server.py" 200
    if errorlevel 1 (
        echo   [x] Server did not come up. Start VGGT-Server.bat by hand.
        goto done
    )
)
echo   Server ready.
echo.

rem ---- pick the folder ------------------------------------------
set "IMGDIR=%~1"
if not defined IMGDIR (
    echo   No folder given - using the newest capture folder on the Desktop.
    "!VPY!" "%~dp0_newest_folder.py" > "%TEMP%\_prism_newest.txt" 2>nul
    if exist "%TEMP%\_prism_newest.txt" set /p IMGDIR=<"%TEMP%\_prism_newest.txt"
    del "%TEMP%\_prism_newest.txt" >nul 2>&1
)
if not defined IMGDIR (
    echo   [x] No photo folder found. Drag one onto this file.
    goto done
)
echo   Folder: !IMGDIR!
echo.
echo ============================================================
echo   Running. Do not close this window.  Ctrl+C to abort.
echo ============================================================
echo.

"!VPY!" -u "%~dp0scan_run.py" "!IMGDIR!"

:done
echo.
pause
exit /b 0
