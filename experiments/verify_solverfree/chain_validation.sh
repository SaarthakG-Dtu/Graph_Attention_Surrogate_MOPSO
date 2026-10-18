#!/bin/bash
# Re-run the solver validation and dataset baselines of sutd-prcm-solverfree on this machine.
# Each step's outputs overwrite the repo's files; the committed originals stay in git for comparison.
cd /e/SUTD_BENCHMARK_NOVEL_GNN/external/sutd-prcm-solverfree
L=logs/verify_chain.log
run() { echo "=== $(date +%H:%M:%S) $*" >> $L; "$@" >> $L 2>&1; echo "--- exit $?" >> $L; }
run python -I test_fdtd_slab.py
run python -I test_fdtd_dataset.py
run python -I test_fdtd_validation.py
run python -I test_copper_loss.py
run python -I test_ground_loss.py
run python -I test_corner_and_fix.py
run python -I freq_shift.py validation
for a in cp_centered rotator_10g2; do run python -I baseline_app.py --app $a; done
run python -I coding_baseline.py
echo "=== CHAIN DONE $(date +%H:%M:%S)" >> $L
