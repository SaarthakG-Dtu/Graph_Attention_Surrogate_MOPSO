"""
benchmark_phase1.py — MSDCNN-style table: every per-class surrogate tested on the
test split of every class (MSE of |r_xx| over 2-12 GHz, mirror-averaged).

    python benchmark_phase1.py                      # runs listed in RUNS that exist
    python benchmark_phase1.py PTN:aug RDN:aug_w32

Output: outputs_new_pipeline/benchmark/cross_class.json (+ printed table).
"""

import json
import os
import sys

import numpy as np

import config as C
from dataset import get_split_arrays
from phase1_forward_model import load_surrogate
from train_phase1 import predict

RUNS = ["PLG:n27000", "PLR:n27000", "PTN:n27000", "RDN:n27000", "RDN:n108000"]   # the networks of the paper


def main(runs):
    # MSDCNN test set: the last 3,000 layouts of each class (the same for every n_train)
    tests = {c: get_split_arrays(c, "test", msdcnn_train=27000) for c in C.CLASSES}
    out_path = os.path.join(C.OUTPUT_ROOT, "benchmark", "cross_class.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for spec in runs:
        cls, _, tag = spec.partition(":")
        run_dir = C.phase1_dir(cls, "co", tag)
        ckpt = os.path.join(run_dir, "phase1_forward_best.pth")
        if not os.path.exists(os.path.join(run_dir, "phase1_metrics.json")):
            print(f"skip {spec} (not finished)")
            continue
        rc = json.load(open(os.path.join(run_dir, "run_config.json")))
        model = load_surrogate(ckpt)
        row = dict(params=rc["n_params"], n_train=rc["n_train"])
        for c, (x, y, _, _) in tests.items():
            p = predict(model, x, tta=True)
            row[c] = float(np.mean((np.linalg.norm(p, axis=-1) - np.linalg.norm(y, axis=-1)) ** 2))
        table[spec] = row
        print(f"{spec:14s} params={row['params']:>9,} n_train={row['n_train']:>6} "
              + "  ".join(f"{c}={row[c]:.2e}" for c in C.CLASSES), flush=True)
    json.dump(table, open(out_path, "w"), indent=1)
    print(f"-> {out_path}")


if __name__ == "__main__":
    C.be_nice()
    main(sys.argv[1:] or RUNS)
