#!/bin/bash
# Two-output (r_xx, r_yx) surrogates, one per class, trained as in MSDCNN (27,000 layouts per
# class, RDN also 108,000; no validation set, last of 300 epochs), true layout convention,
# mirror augmentation, four-mirror averaging of magnitude and phase at test time.
cd /e/SUTD_BENCHMARK_NOVEL_GNN
export PYTHONIOENCODING=utf-8
L=experiments/surrogates_v2/train_queue.log
for spec in RDN:108000 RDN:27000 PTN:27000 PLR:27000 PLG:27000; do
  c=${spec%%:*}; n=${spec##*:}
  echo "=== $(date +%H:%M:%S) $c n$n" >> $L
  python -u train_phase1.py --class_name $c --target both --convention true --msdcnn_train $n \
      --epochs 300 --patience 1000 --augment --tta --tag n$n > experiments/surrogates_v2/train_${c}_n$n.log 2>&1
  echo "--- exit $? $(grep '\[eval' experiments/surrogates_v2/train_${c}_n$n.log | tr '\n' ' ')" >> $L
done
echo "=== QUEUE DONE $(date +%H:%M:%S)" >> $L
