#!/usr/bin/env bash
# Final per-class surrogates with MSDCNN's training-set sizes (no validation set, last epoch kept).
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
run() { python train_phase1.py --class_name $1 --epochs 300 --patience 1000 --augment --tta --msdcnn_train $2 --tag n$2 > logs/p1_$1_n$2.log 2>&1; }
run RDN 108000 &
(run PLG 27000; run PTN 27000; run RDN 27000; run PLR 27000) &
wait
echo done > logs/msdcnn.done
