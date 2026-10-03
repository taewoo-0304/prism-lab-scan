@echo off
rem ============================================================
rem  Shared helper: find Python 3.10-3.12, set PYEXE
rem  Called by the other .bat files. Do not run this directly.
rem
rem  Note: newer py.exe (Python Install Manager) prints a loud
rem  "No runtime installed that matches 3.12" error that ignores
rem  output redirection. So we ask "py -0p" what exists FIRST and
rem  only probe versions it actually reports.
rem ============================================================
set "PYEXE="
set "PYLIST=%TEMP%\vggt_pylist.txt"

rem 1) plain "python" on PATH
call :try_py python
if defined PYEXE goto :eof

rem 2) only the versions the py launcher actually has
py -0p >"%PYLIST%" 2>nul
if exist "%PYLIST%" (
    findstr /c:"3.12" "%PYLIST%" >nul 2>&1 && call :try_py py -3.12
    findstr /c:"3.11" "%PYLIST%" >nul 2>&1 && call :try_py py -3.11
    findstr /c:"3.10" "%PYLIST%" >nul 2>&1 && call :try_py py -3.10
    del "%PYLIST%" >nul 2>&1
)
if defined PYEXE goto :eof

rem 3) common install locations
for %%V in (312 311 310) do call :try_exe "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
for %%V in (312 311 310) do call :try_exe "%ProgramFiles%\Python%%V\python.exe"
for %%V in (312 311 310) do call :try_exe "C:\Python%%V\python.exe"
goto :eof

:try_py
if defined PYEXE goto :eof
%* -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYEXE=%*"
goto :eof

:try_exe
if defined PYEXE goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set PYEXE="%~1"
goto :eof
