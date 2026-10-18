#!/bin/bash
# Converter, pixel space, standard budget: our pixel step (0.25/6) vs the solverfree setting (0.5).
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
for v in 0.0416667 0.5; do
  python -u - <<PY >> experiments/devices/step_check.log 2>&1
import argparse, config as C, phase2_mopso as M
C.PIXEL_MAXVEL = $v
C.MOPSO_MAXGEN, C.MOPSO_NP, C.MOPSO_SEED = 150, 200, 0
M.main(argparse.Namespace(problem="cp", space="binary", surrogate="RDN:n108000", vae_tag="", tag="pix_v$v",
                          database_only=False, gens=150, swarm=200, seed=0))
PY
done
echo DONE >> experiments/devices/step_check.log
