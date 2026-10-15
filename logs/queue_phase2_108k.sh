#!/usr/bin/env bash
# Phase 2 with the RDN108000 surrogate: searches, CST check of the 6 best designs, schematics.
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
until grep -q "eval" logs/p1_RDN_n108000.log 2>/dev/null; do sleep 120; done
rm -rf outputs_new_pipeline/PHASE2/mopso/amc_x_*
for s in 0 1 2; do python phase2_mopso.py --seed $s --tag s$s > logs/amc_s$s.log 2>&1; done
python phase2_mopso.py --swarm 1000 --gens 300 --tag long > logs/amc_long.log 2>&1
python phase2_mopso.py --space binary --tag binary > logs/amc_binary.log 2>&1
python cst_verify.py verify --run_dir outputs_new_pipeline/PHASE2/mopso/amc_x_long --top 6 > logs/cst_verify_amc_long.log 2>&1
python paper/schematics.py data:architecture data:pipeline architecture pipeline unitcell > logs/schematics.log 2>&1
echo done > logs/phase2_108k.done
