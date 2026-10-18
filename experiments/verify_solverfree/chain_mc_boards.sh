#!/bin/bash
# Monte Carlo of tolerances and the checkerboard boards of sutd-prcm-solverfree, re-run on this machine.
SF=/e/SUTD_BENCHMARK_NOVEL_GNN/external/sutd-prcm-solverfree
L=$SF/logs/verify_chain.log
cd $SF
run() { echo "=== $(date +%H:%M:%S) $*" >> $L; "$@" >> $L 2>&1; echo "--- exit $?" >> $L; }
cp search/mc_v2/mc.json search/mc_v2/mc_repo.json
run python -I mc_tolerance.py mc 64
for dn in 0:3 0:4 0:6 0:8 4:4 4:6 1:6; do
  run python -I coding_board.py --search search/coding_v2 --design ${dn%%:*} --tiles ${dn##*:} --pol both --copper-rs 0.026
done
run python -I test_ground_loss.py slab
run python -I test_ground_loss.py cells
echo "=== CHAIN3 DONE $(date +%H:%M:%S)" >> $L
