@echo off
cd /d E:\SUTD_BENCHMARK_NOVEL_GNN
:wait
if not exist experiments\vae_study\sweep_rdn_d150.json (timeout /t 30 /nobreak >nul & goto wait)
python -u experiments\vae_study\search_sweep.py --vae_tag rdn_d150 --bounds 1.0 1.5 --maxvels 0.05 0.1 0.2 --name rdn_small > experiments\vae_study\sweep_rdn_small.log 2>&1
python -u experiments\vae_study\search_sweep.py --vae_tag rdn_d150 --space binary --bounds 3.0 --maxvels 0.1 0.25 0.5 1.0 2.0 --name binary > experiments\vae_study\sweep_binary.log 2>&1
echo ALL_DONE > experiments\vae_study\queue2.done
