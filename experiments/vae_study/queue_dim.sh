#!/bin/bash
cd /e/SUTD_BENCHMARK_NOVEL_GNN
until [ -f outputs_new_pipeline/PHASE2/vae_rdn_d100_b06/vae_diagnostics.json ]; do sleep 30; done
sleep 20
export PYTHONIOENCODING=utf-8
for t in rdn_d200 rdn_d100_b06; do
  VAE_TAG=$t python -u experiments/vae_study/latent_diagnostics.py > experiments/vae_study/diag_$t.log 2>&1
  python -u experiments/vae_study/search_sweep.py --vae_tag $t --bounds 0.75 1.0 --maxvels 0.05 --name $t > experiments/vae_study/sweep_$t.log 2>&1
done
touch experiments/vae_study/queue_dim.done
