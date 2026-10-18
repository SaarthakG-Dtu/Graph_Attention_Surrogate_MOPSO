#!/bin/bash
# Coding checkerboard searches with the four-output RDN108000 (structures and budgets as in sutd-prcm-solverfree).
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
L=experiments/devices/coding_searches.log
run() { echo "=== $(date +%H:%M:%S) $*" >> $L; python -u phase2_mopso.py "$@" >> $L 2>&1; echo "--- exit $?" >> $L; }
S="--surrogate RDN:n108000 --seed 0"
run --problem coding_diag   --space binary $S --tag pix --gens 300 --swarm 1000
run --problem coding_mirror --space binary $S --tag pix --gens 300 --swarm 1000
run --problem coding_mirror --space latent --vae_tag rdn_d200 $S --tag vae --gens 300 --swarm 1000
run --problem coding_free   --space binary $S --tag pix --gens 250 --swarm 600
run --problem coding_free   --space latent --vae_tag rdn_d200 $S --tag vae --gens 250 --swarm 600
echo "=== CODING DONE $(date +%H:%M:%S)" >> $L
