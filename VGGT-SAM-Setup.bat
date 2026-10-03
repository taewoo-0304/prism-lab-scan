@echo off
rem ============================================================
rem  VGGT 3D Scan - SAM 3 setup (login + download weights)
rem  Run this once, after your facebook/sam3 access is approved.
rem  (ASCII only on purpose - Korean text lives in Python.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"
title VGGT 3D Scan - SAM 3 setup
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

call "%~dp0_findpy.bat"
if not defined PYEXE (
    echo   [!] Python not found. Run VGGT-Server.bat once first.
    goto done
)

%PYEXE% "%~dp0launcher.py" --sam-setup

:done
echo.
pause
exit /b 0
