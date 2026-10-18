#!/bin/bash
# Four-output class networks with r_yx trained under the r_xx loss settings (no dB floor, unweighted phase).
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
t() { python -u train_phase1.py --class_name $1 --target both --convention true --msdcnn_train 27000 --epochs 300 \
        --patience 1000 --augment --tta --cross_loss plain --tag n27000_plainyx > experiments/surrogates_v2/train_$1_n27000_plainyx.log 2>&1; }
t PTN & t RDN & wait
t PLG & t PLR & wait
echo PLAINYX_DONE >> experiments/surrogates_v2/train_queue.log
