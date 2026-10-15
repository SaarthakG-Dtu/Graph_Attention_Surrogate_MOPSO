"""Disagreement of independently trained RDN surrogates (RDN108000, its no-PGA
ablation, RDN27000) as a proxy of prediction reliability for searched layouts."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config as C
from phase1_forward_model import load_surrogate
from train_phase1 import predict
from dataset import get_split_arrays
M = [load_surrogate(os.path.join(C.phase1_dir("RDN", "co", t), "phase1_forward_best.pth"))
     for t in ("n108000", "n108000_nopga", "n27000")]
def dis(x):
    P = [predict(m, x.astype(np.float32), tta=False) for m in M]
    P = [p[..., 0] + 1j * p[..., 1] for p in P]
    return np.mean([np.mean(np.abs(P[0] - p) ** 2, 1) for p in P[1:]], 0)   # per layout
if __name__ == "__main__":
    x, _, _, _ = get_split_arrays("RDN", "test", msdcnn_train=27000)
    print("RDN test   median %.2e  p95 %.2e" % tuple(np.percentile(dis(x[:2000]), [50, 95])))
    mop = os.path.join(C.OUTPUT_ROOT, "PHASE2", "mopso")
    for tag in sys.argv[1:]:
        d = np.load(os.path.join(mop, "amc_x_" + tag, "Final", "pareto_designs.npz"))
        o = np.argsort(d["objectives"].sum(1))
        v = dis(d["patterns"][o])
        print(f"{tag:40s} top6 {' '.join('%.1e' % a for a in v[:6])} | archive median {np.median(v):.1e}")
