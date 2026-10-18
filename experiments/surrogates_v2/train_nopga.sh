#!/bin/bash
# PGA ablation of the four-output RDN108000: same network without the PGA blocks (model resnet3conv).
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
until grep -q "yield_cp\|Traceback" experiments/devices/mc.log 2>/dev/null; do sleep 120; done
python -u train_phase1.py --class_name RDN --target both --convention true --msdcnn_train 108000 --model_type resnet3conv \
    --epochs 300 --patience 1000 --augment --tta --tag n108000_nopga > experiments/surrogates_v2/train_RDN_n108000_nopga.log 2>&1
