#!/bin/bash
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
M=outputs_new_pipeline/PHASE2/mopso
python -u cst_verify.py verify --run_dir $M/amc_x_sc_latent_N1000_G300_s0 --top 6 --t_sub 1.0 > experiments/vae_study/cst_latent.log 2>&1
python -u cst_verify.py verify --run_dir $M/amc_x_sc_binary_N1000_G300_s0 --top 3 --t_sub 1.0 > experiments/vae_study/cst_binary.log 2>&1
touch experiments/vae_study/cst.done
