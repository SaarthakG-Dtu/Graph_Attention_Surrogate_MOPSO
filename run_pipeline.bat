@echo off
REM ===========================================================================
REM  run_pipeline.bat  -  SUTD-PRCM surrogate + VAE-MOIPSO pipeline
REM
REM  From cmd:          run_pipeline.bat [step]
REM  From PowerShell:   cmd /c run_pipeline.bat [step]
REM
REM  steps: phase1 | vae | mopso | cst_calibrate | cst_verify | figures | all
REM  (no argument = all; CST steps need CST Studio 2024 on this PC)
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM call conda activate <your_env>

set "STEP=%~1"
if "%STEP%"=="" set "STEP=all"

if /I "%STEP%"=="phase1"        goto :phase1
if /I "%STEP%"=="vae"           goto :vae
if /I "%STEP%"=="mopso"         goto :mopso
if /I "%STEP%"=="cst_calibrate" goto :cst_calibrate
if /I "%STEP%"=="cst_verify"    goto :cst_verify
if /I "%STEP%"=="figures"       goto :figures
if /I "%STEP%"=="all"           goto :phase1
echo Unknown step "%STEP%"
exit /b 1

:phase1
echo [Phase 1] MSDCNN protocol: PLG27000, PLR27000, PTN27000, RDN27000, RDN108000 (300 epochs, last epoch kept), PGA ablation, benchmark
for %%C in (PLG PLR PTN RDN) do (
    python train_phase1.py --class_name %%C --epochs 300 --patience 1000 --augment --tta --msdcnn_train 27000 --tag n27000 || goto :fail
)
python train_phase1.py --class_name RDN --epochs 300 --patience 1000 --augment --tta --msdcnn_train 108000 --tag n108000 || goto :fail
rem PGA ablation: RDN108000 without PGA blocks, same protocol
python train_phase1.py --class_name RDN --model_type resnet3conv --epochs 300 --patience 1000 --augment --tta --msdcnn_train 108000 --tag n108000_nopga || goto :fail
python benchmark_phase1.py || goto :fail
if /I not "%STEP%"=="all" goto :done

:vae
echo [Phase 2] RDN layout VAE (d = 150) and the collapsing d = 64 control
python phase2_vae.py --mode both --tag rdn_d150 || goto :fail
python phase2_vae.py --mode both --latent_dim 64 --channels 64 128 256 --epochs 80 --tag rdn || goto :fail
if /I not "%STEP%"=="all" goto :done

:mopso
echo [Phase 2] MOIPSO for the X-band in-phase reflector: 3 seeds, long run, pixel-space baseline
for %%S in (0 1 2) do (
    python phase2_mopso.py --seed %%S --tag s%%S || goto :fail
)
python phase2_mopso.py --swarm 1000 --gens 300 --tag long || goto :fail
python phase2_mopso.py --space binary --tag binary || goto :fail
if /I not "%STEP%"=="all" goto :done

:cst_calibrate
echo [CST] substrate-thickness search and phase reference
python cst_thickness_grid.py --wide 0.6 0.8 1.2 1.5 1.8 2.2 --fine 0.95 0.975 1.025 1.05 || goto :fail
python cst_reference.py --t 1.0 || goto :fail
if /I not "%STEP%"=="all" goto :done

:cst_verify
echo [CST] verification of the best Pareto designs: 6 of the long VAE run, 3 of the pixel-space run
python cst_verify.py verify --run_dir outputs_new_pipeline\PHASE2\mopso\amc_x_long --top 6 || goto :fail
python cst_verify.py verify --run_dir outputs_new_pipeline\PHASE2\mopso\amc_x_binary --top 3 || goto :fail
if /I not "%STEP%"=="all" goto :done

:figures
echo [paper] figures and results
python paper\make_figures.py || goto :fail
python paper\schematics.py data:architecture data:pipeline architecture pipeline unitcell || goto :fail
python paper\collect_results.py || goto :fail
goto :done

:done
echo Done.
exit /b 0

:fail
echo ERROR - pipeline stopped (see messages above)
exit /b 1
