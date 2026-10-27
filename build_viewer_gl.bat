@echo off
rem build_viewer_gl.bat - builds the OpenGL viewer for the GPU solver from any terminal.
rem   .\build_viewer_gl.bat     ->  viewer_gl.exe
rem Uses GLFW and GLAD from external\ (nothing to install). See build_gpu.bat for why vcvars.

setlocal
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
rem (no "if (...)" blocks with this path: the ")" in "(x86)" would end the block early)
if not exist "%VCVARS%" goto :novs
call "%VCVARS%" >nul 2>nul

cd /d "%~dp0"
echo Building src\viewer_gl.cu -^> viewer_gl.exe
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler "/openmp /O2" ^
    -Iexternal\glad\include -Iexternal\glfw\include ^
    src\viewer_gl.cu external\glad\src\gl.c -o viewer_gl.exe ^
    -Lexternal\glfw\lib-vc2022 -lglfw3_mt -lopengl32 -luser32 -lgdi32 -lshell32
if errorlevel 1 goto :failed
echo Done. Run it with:  .\viewer_gl.exe
exit /b 0

:failed
echo BUILD FAILED
exit /b 1

:novs
echo Could not find the Visual Studio Build Tools at:
echo   %VCVARS%
exit /b 1
