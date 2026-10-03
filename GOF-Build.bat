@echo off
rem ============================================================
rem  Build GOF CUDA extensions.
rem  Needs the MSVC environment (vcvars64) AND the CUDA toolkit,
rem  which is why this cannot be done from the plain Python script.
rem
rem  Plain (non-editable) install on purpose: simple_knn ships no
rem  __init__.py, and an editable install cannot resolve it as a PEP 420
rem  namespace package - you get "No module named simple_knn" even after a
rem  successful build. A normal install copies _C.pyd into site-packages,
rem  where namespace resolution works.
rem  (ASCII only on purpose - Korean text lives in Python.)
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"
title GOF - build CUDA extensions

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo   [!] vcvars64.bat not found: %VCVARS%
    goto done
)

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo   [!] nvcc not found under %CUDA_PATH%
    echo       Install the CUDA toolkit first, or set CUDA_PATH.
    goto done
)
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

if not defined GOF_DIR set "GOF_DIR=%USERPROFILE%\gaussian-opacity-fields"
if not defined GOF_PYTHON set "GOF_PYTHON=%USERPROFILE%\vggt_server\venv\Scripts\python.exe"

echo   MSVC   : %VCVARS%
echo   CUDA   : %CUDA_PATH%
echo   GOF    : %GOF_DIR%
echo   python : %GOF_PYTHON%
echo.
call "%VCVARS%" >nul 2>&1

rem nvcc rejects host compilers it does not know. MSVC 14.4x is newer than
rem what CUDA 12.6 lists, so allow it explicitly rather than downgrading VS.
set "NVCC_FLAGS=-allow-unsupported-compiler"
rem RTX 3070 is compute capability 8.6. Pinning it keeps the build short -
rem otherwise nvcc emits code for every architecture torch knows about.
set "TORCH_CUDA_ARCH_LIST=8.6"
rem torch's cpp_extension refuses to run once vcvars64 has been called unless
rem this is set - it guards against activating the VC env twice.
set "DISTUTILS_USE_SDK=1"

echo   [1/2] diff-gaussian-rasterization
"%GOF_PYTHON%" -m pip install --no-build-isolation "%GOF_DIR%\submodules\diff-gaussian-rasterization"
echo.
echo   [2/2] simple-knn
"%GOF_PYTHON%" -m pip install --no-build-isolation "%GOF_DIR%\submodules\simple-knn"
echo.

echo   check:
"%GOF_PYTHON%" -c "import diff_gaussian_rasterization, simple_knn; print('    both extensions import OK')"

:done
echo.
pause
exit /b 0
