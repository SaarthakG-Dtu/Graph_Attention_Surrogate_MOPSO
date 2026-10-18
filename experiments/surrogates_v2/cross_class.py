"""
cross_class.py — Table III of the four-output networks: each network (trained on one class) tested on the
3,000 test layouts of every class, |r_xx| and |r_yx| MSE over 2-12 GHz and the magnitude-weighted phase
error, with four-mirror averaging of magnitude and phase.

    python experiments/surrogates_v2/cross_class.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config as C                                       # noqa: E402
from dataset import get_split_arrays                     # noqa: E402
from phase1_forward_model import load_surrogate          # noqa: E402
from phase1_loss import complex_metrics                  # noqa: E402
from train_phase1 import predict                         # noqa: E402

NETS = [("PLG", 27000), ("PLR", 27000), ("PTN", 27000), ("RDN", 27000), ("RDN", 108000)]
res = {}
tests = {c: get_split_arrays(c, "test", "both", "true", msdcnn_train=27000)[:2] for c in C.CLASSES}
for cls, n in NETS:
    name = f"{cls}{n}"
    m = load_surrogate(os.path.join(C.phase1_dir(cls, "both", f"n{n}"), "phase1_forward_best.pth"))
    res[name] = {}
    for c, (x, y) in tests.items():
        p = predict(m, x, tta=True, target="both")
        row = {}
        for k, out in enumerate(("rxx", "ryx")):
            row[f"mse_{out}"] = float(np.mean((np.linalg.norm(p[:, k], axis=-1) - np.linalg.norm(y[:, k], axis=-1)) ** 2))
            row[f"phase_{out}_deg"] = complex_metrics(p[:, k], y[:, k])["phase_mae_deg_weighted"]
        res[name][c] = row
    print(name, "  ".join(f"{c}: {1e3 * r['mse_rxx']:.3f}/{1e3 * r['mse_ryx']:.3f}" for c, r in res[name].items()), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_class.json"), "w"), indent=1)
