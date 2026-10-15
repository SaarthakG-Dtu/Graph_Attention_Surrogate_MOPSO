#!/bin/bash
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
python -u phase2_vae.py --mode both --latent_dim 256 --tag rdn_d256 > experiments/vae_study/train_vae_d256.log 2>&1
python -u experiments/vae_study/search_sweep.py --vae_tag rdn_d200 --bounds 0.5 0.75 --maxvels 0.02 0.1 --name rdn_d200_b > experiments/vae_study/sweep_rdn_d200_b.log 2>&1
VAE_TAG=rdn_d256 python -u experiments/vae_study/latent_diagnostics.py > experiments/vae_study/diag_rdn_d256.log 2>&1
python -u experiments/vae_study/search_sweep.py --vae_tag rdn_d256 --bounds 0.5 0.75 1.0 --maxvels 0.05 --name rdn_d256 > experiments/vae_study/sweep_rdn_d256.log 2>&1
python -u experiments/vae_study/search_sweep.py --vae_tag rdn_d200 --bounds 0.5 --maxvels 0.05 --name rdn_d200_c > experiments/vae_study/sweep_rdn_d200_c.log 2>&1
touch experiments/vae_study/queue_dim2.done
