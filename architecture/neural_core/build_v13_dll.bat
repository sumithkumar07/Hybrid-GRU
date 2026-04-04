@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
echo [SOVEREIGN] Compiling BitNet Engine...
cl /O2 /LD /EHsc /openmp /openmp:experimental /D_AMD64_ sovereign_v13_bitnet.cpp /Fe:sovereign.dll
echo [SOVEREIGN] DLL Generation Finished.
dir sovereign.dll
