@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
echo [hybrid_gru] Compiling BitNet Engine...
cl /O2 /LD /EHsc /openmp /openmp:experimental /D_AMD64_ hybrid_gru_v13_bitnet.cpp /Fe:hybrid_gru.dll
echo [hybrid_gru] DLL Generation Finished.
dir hybrid_gru.dll
