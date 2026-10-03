@echo off
rem ============================================================
rem  VGGT 3D Scan - measure the real single-pass frame limit
rem  Start VGGT-Server.bat first. Takes a few minutes.
rem  (ASCII only on purpose - Korean text lives in Python.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"
title VGGT 3D Scan - capacity test
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo.
echo   Measuring how many frames fit in ONE pass (no chaining).
echo   Drag a folder with at least one photo onto this window.
echo.
set "IMGDIR=%~1"
if not defined IMGDIR set /p IMGDIR="  Photo folder: "
if not defined IMGDIR goto done

call "%~dp0_findpy.bat"
if not defined PYEXE (
    echo   [!] Python not found. Run VGGT-Server.bat once first.
    goto done
)

echo.
%PYEXE% "%~dp0launcher.py" --test %IMGDIR% --capacity

:done
echo.
pause
exit /b 0
