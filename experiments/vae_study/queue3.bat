@echo off
cd /d E:\SUTD_BENCHMARK_NOVEL_GNN
:wait
if not exist experiments\vae_study\queue2.done (timeout /t 60 /nobreak >nul & goto wait)
python -u experiments\vae_study\scaling.py > experiments\vae_study\scaling.log 2>&1
echo ALL_DONE > experiments\vae_study\queue3.done
