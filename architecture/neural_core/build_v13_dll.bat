@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
echo [SOVEREIGN] Compiling BitNet Engine...
cl /O2 /LD /EHsc /openmp /openmp:experimental /D_AMD64_ sovereign_v13_bitnet.cpp /Fe:sovereign.dll /link /EXPORT:sovereign_init_master /EXPORT:sovereign_free_master /EXPORT:sovereign_init_agent /EXPORT:sovereign_agent_observe /EXPORT:sovereign_agent_act /EXPORT:sovereign_init_fragment /EXPORT:sovereign_get_fragment_bias /EXPORT:sovereign_agent_set_fragment /EXPORT:sovereign_load_compact /EXPORT:sovereign_save_compact /EXPORT:sovereign_train_distill_bulk /EXPORT:sovereign_agent_get_h /EXPORT:sovereign_free_agent
echo [SOVEREIGN] DLL Generation Finished.
dir sovereign.dll
