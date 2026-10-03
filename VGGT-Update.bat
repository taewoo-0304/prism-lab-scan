@echo off
rem ============================================================
rem  VGGT 3D Scan - update the VGGT source from GitHub
rem  The May 2026 upstream fix lets 2-3x more frames fit in the
rem  same VRAM, which is what makes single-pass runs practical.
rem  Restart VGGT-Server.bat after this finishes.
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
title VGGT 3D Scan - update

set "VGGTDIR=%USERPROFILE%\vggt_server\vggt"
if not exist "%VGGTDIR%\.git" (
    echo.
    echo   [!] Not found: %VGGTDIR%
    echo       Run VGGT-Server.bat once to install first.
    goto done
)

set "GITEXE="
where git >nul 2>&1
if not errorlevel 1 set "GITEXE=git"
if not defined GITEXE if exist "%ProgramFiles%\Git\cmd\git.exe" set GITEXE="%ProgramFiles%\Git\cmd\git.exe"
if not defined GITEXE if exist "%ProgramFiles(x86)%\Git\cmd\git.exe" set GITEXE="%ProgramFiles(x86)%\Git\cmd\git.exe"
if not defined GITEXE if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set GITEXE="%LOCALAPPDATA%\Programs\Git\cmd\git.exe"
if not defined GITEXE (
    echo   [!] git not found. Install Git for Windows, then run this again.
    goto done
)

echo.
echo   Folder : %VGGTDIR%
echo.
echo   Before:
%GITEXE% -C "%VGGTDIR%" log -1 --date=short --format="     %%h  %%ad  %%s"

echo.
echo   Pulling...
%GITEXE% -C "%VGGTDIR%" pull

echo.
echo   After:
%GITEXE% -C "%VGGTDIR%" log -1 --date=short --format="     %%h  %%ad  %%s"

echo.
echo   Done. Close the server window and start VGGT-Server.bat again
echo   so the new code is loaded.

:done
echo.
pause
exit /b 0
