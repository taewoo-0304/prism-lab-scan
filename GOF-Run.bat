@echo off
rem ============================================================
rem  GOF scan - photos -> poses+points -> GOF -> mesh -> STL
rem  Drag a photo folder onto this file.
rem  (ASCII only on purpose - Korean text lives in Python.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"
title GOF scan
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo.
echo   GOF scan  (VGGT/COLMAP -^> GOF -^> mesh -^> STL)
echo.
set "IMGDIR=%~1"
if not defined IMGDIR set /p IMGDIR="  Photo folder: "
if not defined IMGDIR goto done

call "%~dp0_findpy.bat"
if not defined PYEXE (
    echo   [!] Python not found. Run VGGT-Server.bat once first.
    goto done
)

set "POSE=colmap"
set /p POSETXT="  Pose source - colmap / vggt (blank = colmap): "
if defined POSETXT set "POSE=%POSETXT%"

set "PREPOPT="
set /p DENSETXT="  Use COLMAP dense cloud for init? slow, ~20min (y/N): "
if /i "%DENSETXT%"=="y" set "PREPOPT=%PREPOPT% --dense"

set "RUNOPT="
set /p MMTXT="  Real length of the object's longest side in mm (blank = no scale): "
if defined MMTXT set "RUNOPT=%RUNOPT% --size-mm %MMTXT%"
set /p RESTXT="  Training downscale, 8GB VRAM prefers 2 (blank = 2): "
if defined RESTXT set "RUNOPT=%RUNOPT% -r %RESTXT%"

echo.
echo   [1/2] building GOF dataset
%PYEXE% "%~dp0gof_prep.py" "%IMGDIR%" --pose %POSE%%PREPOPT%
if errorlevel 1 goto done

echo.
echo   [2/2] GOF training + mesh + STL
%PYEXE% "%~dp0gof_pipe.py" "%IMGDIR%\_gof"%RUNOPT%

:done
echo.
pause
exit /b 0
