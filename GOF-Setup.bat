@echo off
rem ============================================================
rem  GOF (Gaussian Opacity Fields) - install / diagnose
rem  Run this once before GOF-Run.bat.
rem  (ASCII only on purpose - Korean text lives in Python.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"
title GOF - setup
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

call "%~dp0_findpy.bat"
if not defined PYEXE (
    echo   [!] Python not found. Run VGGT-Server.bat once first.
    goto done
)

echo.
%PYEXE% "%~dp0gof_setup.py"
echo.
set /p GO="  Download and build GOF now? (y/N): "
if /i not "%GO%"=="y" goto done

echo.
%PYEXE% "%~dp0gof_setup.py" --install --python "%PYEXE%"

:done
echo.
pause
exit /b 0
