#!/bin/bash
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
M=outputs_new_pipeline/PHASE2/mopso
for t in vs_rdn_d200_latent_B1.5_v0.2_s0 vs_rdn_d200_latent_B1_v0.05_s0; do
  python -u cst_verify.py verify --run_dir $M/amc_x_$t --top 3 --t_sub 1.0 > experiments/vae_study/cst_$t.log 2>&1
  python experiments/vae_study/cst_check.py $M/amc_x_$t 2>&1 | grep sol > experiments/vae_study/cstcheck_$t.txt
done
touch experiments/vae_study/cst_box.done
