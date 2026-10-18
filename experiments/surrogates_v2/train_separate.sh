#!/bin/bash
# Separate single-output networks (r_xx alone, r_yx alone) for comparison with the two-output RDN108000:
# same architecture and settings, true convention, corrected KK sign, magnitude/phase mirror averaging.
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
L=experiments/surrogates_v2/train_queue.log
SF=external/sutd-prcm-solverfree
until [ -f $SF/search/mc_v2/etch.json ] || grep -q "Traceback" $SF/search/mc_v2/etch.log; do sleep 120; done
for t in co cross; do
  echo "=== $(date +%H:%M:%S) RDN n108000 single-output $t" >> $L
  python -u train_phase1.py --class_name RDN --target $t --convention true --msdcnn_train 108000 \
      --epochs 300 --patience 1000 --augment --tta --tag n108000_v2 > experiments/surrogates_v2/train_RDN_n108000_$t.log 2>&1
  echo "--- exit $? $(grep '\[eval' experiments/surrogates_v2/train_RDN_n108000_$t.log | tr '\n' ' ')" >> $L
done
