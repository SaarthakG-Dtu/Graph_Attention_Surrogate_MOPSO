#!/bin/bash
# Converter and rotator searches with the four-output RDN108000: pixel space (3 standard seeds + long),
# then VAE space (d = 200) once the VAE is trained. Results in outputs_new_pipeline/PHASE2/mopso/<problem>_<tag>.
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
L=experiments/devices/searches.log
run() { echo "=== $(date +%H:%M:%S) $*" >> $L; python -u phase2_mopso.py "$@" >> $L 2>&1; echo "--- exit $?" >> $L; }
until grep -q "neighbour agreement\|Traceback" experiments/devices/train_vae_d200.log 2>/dev/null; do sleep 60; done
for p in cp rotator; do
  for s in 0 1 2; do run --problem $p --space binary --surrogate RDN:n108000 --tag pix_s$s --gens 150 --swarm 200 --seed $s; done
  run --problem $p --space binary --surrogate RDN:n108000 --tag pix_long --gens 300 --swarm 1000 --seed 0
done
for p in cp rotator; do
  for s in 0 1 2; do run --problem $p --space latent --vae_tag rdn_d200 --surrogate RDN:n108000 --tag vae_s$s --gens 150 --swarm 200 --seed $s; done
  run --problem $p --space latent --vae_tag rdn_d200 --surrogate RDN:n108000 --tag vae_long --gens 300 --swarm 1000 --seed 0
done
echo "=== SEARCHES DONE $(date +%H:%M:%S)" >> $L
