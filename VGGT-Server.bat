@echo off
rem ============================================================
rem  VGGT 3D Scan - laptop compute server
rem  Double-click this file. It installs on first run, then just runs.
rem  (ASCII only on purpose - cmd.exe garbles non-ASCII batch files.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"
title VGGT 3D Scan Server
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo.
echo   VGGT 3D Scan - checking Python...
echo.

call "%~dp0_findpy.bat"
if defined PYEXE goto run

echo   Python 3.10-3.12 not found. Installing it for this user.
echo   (no administrator rights needed, 2-5 minutes)
echo.

rem A) new Python Install Manager can install runtimes itself
py install 3.12 >nul 2>&1
call "%~dp0_findpy.bat"
if defined PYEXE goto run

rem B) winget
where winget >nul 2>&1
if not errorlevel 1 (
    echo   Trying winget...
    winget install --id Python.Python.3.12 -e --source winget --scope user --accept-package-agreements --accept-source-agreements
)
call "%~dp0_findpy.bat"
if defined PYEXE goto run

rem C) official installer
echo.
echo   Downloading the Python installer...
curl -L --fail -o "%TEMP%\py312.exe" https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe
if not exist "%TEMP%\py312.exe" goto nopy
echo   Installing Python (2-3 minutes)...
"%TEMP%\py312.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
call "%~dp0_findpy.bat"
if defined PYEXE goto run

:nopy
echo.
echo   [!] Could not install Python automatically.
echo       Install Python 3.12 manually:
echo         https://www.python.org/downloads/release/python-3128/
echo       Check "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
exit /b 1

:run
echo   Using: %PYEXE%
%PYEXE% "%~dp0launcher.py" %*
set "RC=%errorlevel%"
echo.
pause
exit /b %RC%
