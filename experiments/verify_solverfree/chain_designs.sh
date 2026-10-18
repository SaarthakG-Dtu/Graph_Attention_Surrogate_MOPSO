#!/bin/bash
# After the validation chain: re-simulate the repo's designs, Monte Carlo and checkerboards.
cd /e/SUTD_BENCHMARK_NOVEL_GNN
L=external/sutd-prcm-solverfree/logs/verify_chain.log
until grep -q "CHAIN DONE" $L; do sleep 60; done
run() { echo "=== $(date +%H:%M:%S) $*" >> $L; "$@" >> $L 2>&1; echo "--- exit $?" >> $L; }
export PYTHONIOENCODING=utf-8
run python -u experiments/verify_solverfree/fdtd_designs.py
cd external/sutd-prcm-solverfree
run python -I mc_tolerance.py mc 64
for dn in 0:3 0:4 0:6 0:8 4:4 4:6 1:6; do
  run python -I coding_board.py --search search/coding_v2 --design ${dn%%:*} --tiles ${dn##*:} --pol both --copper-rs 0.026
done
echo "=== CHAIN2 DONE $(date +%H:%M:%S)" >> $L
