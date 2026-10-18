#!/bin/bash
# PGA ablation on the 27,000-layout sets (PTN, RDN): same four-output network without the PGA blocks.
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
for c in PTN RDN; do
  python -u train_phase1.py --class_name $c --target both --convention true --msdcnn_train 27000 --model_type resnet3conv \
      --epochs 300 --patience 1000 --augment --tta --tag n27000_nopga > experiments/surrogates_v2/train_${c}_n27000_nopga.log 2>&1 &
done
wait
for c in PTN RDN; do grep -E "\[eval" experiments/surrogates_v2/train_${c}_n27000_nopga.log; done
