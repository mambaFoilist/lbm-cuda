@echo off
rem build_gpu.bat - builds CUDA code from any terminal (PowerShell, VS Code, cmd).
rem nvcc needs Microsoft's C++ compiler set up first; this does that, then compiles.
rem
rem   .\build_gpu.bat                      builds tests\check_gpu.cu  ->  check_gpu.exe
rem   .\build_gpu.bat tests\something.cu   builds that file           ->  something.exe
rem
rem Paths are relative to this folder (the repo root).

setlocal
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
rem (no "if (...)" blocks with this path: the ")" in "(x86)" would end the block early)
if not exist "%VCVARS%" goto :novs
call "%VCVARS%" >nul 2>nul

cd /d "%~dp0"
set "SRC=%~1"
if "%SRC%"=="" set "SRC=tests\check_gpu.cu"
for %%F in ("%SRC%") do set "OUT=%%~nF.exe"

echo Building %SRC% -^> %OUT%
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler "/openmp /O2" "%SRC%" -o "%OUT%"
if errorlevel 1 goto :failed
echo Done. Run it with:  .\%OUT%
exit /b 0

:failed
echo BUILD FAILED
exit /b 1

:novs
echo Could not find the Visual Studio Build Tools at:
echo   %VCVARS%
exit /b 1
