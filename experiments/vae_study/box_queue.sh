#!/bin/bash
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
S=experiments/vae_study
python -u $S/search_sweep.py --vae_tag rdn_d200 --bounds 1.5 2.0 --maxvels 0.1 0.2 --name box_d200_a > $S/sweep_box_d200_a.log 2>&1
python -u $S/search_sweep.py --vae_tag rdn_d200 --bounds 3.0 --maxvels 0.1 0.3 --name box_d200_b > $S/sweep_box_d200_b.log 2>&1
touch $S/box.done
