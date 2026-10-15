"""
phase2_surrogate.py — the frozen Phase 1 co-pol surrogate as the MOPSO evaluator.

Surrogate.predict(patterns) -> R: complex co-pol reflection r_xx on the 101-point
2-12 GHz grid. The head outputs |S| cos(phi), |S| sin(phi), so the complex output
carries both magnitude and phase.

sanity_check() reports the surrogate's error on the held-out test split of its
class; database() returns every layout of the given classes with its CST r_xx.
"""

import json
import os

import numpy as np
import torch

import config as C
from dataset import get_split_arrays, freq_grid, load_class, to_convention
from phase1_forward_model import load_surrogate


def surrogate_ckpt(spec, target="co"):
    """spec "CLASS[:tag]" -> checkpoint of that Phase 1 run."""
    cls, _, tag = spec.partition(":")
    return os.path.join(C.phase1_dir(cls, target, tag), "phase1_forward_best.pth")


def _convention(ckpt):
    rc = os.path.join(os.path.dirname(ckpt), "run_config.json")
    return json.load(open(rc)).get("layout_convention", "file") if os.path.exists(rc) else "file"


class Surrogate:
    def __init__(self, spec=C.MOPSO_SURROGATE, batch=2048):
        self.spec, self.cls, self.batch = spec, spec.partition(":")[0], batch
        self.ckpt = surrogate_ckpt(spec)
        if not os.path.exists(self.ckpt):
            raise FileNotFoundError(f"surrogate not found: {self.ckpt}")
        self.model = load_surrogate(self.ckpt)
        self.convention = _convention(self.ckpt)
        rc = os.path.join(os.path.dirname(self.ckpt), "run_config.json")
        self.msdcnn_train = json.load(open(rc)).get("msdcnn_train") if os.path.exists(rc) else None
        self.freq = freq_grid()
        print(f"  [surrogate] {self.ckpt} (layout convention: {self.convention})")

    @torch.no_grad()
    def predict(self, patterns):
        x = torch.as_tensor(np.asarray(patterns), dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = []
        for i in range(0, len(x), self.batch):
            c = self.model(x[i:i + self.batch].to(C.DEVICE)).float().cpu().numpy()
            out.append(c[..., 0] + 1j * c[..., 1])
        return np.concatenate(out)


def sanity_check(sur: Surrogate) -> dict:
    """Test-split error of the surrogate (cached next to its checkpoint)."""
    cache = os.path.join(os.path.dirname(sur.ckpt), "surrogate_sanity.json")
    key = f"{sur.ckpt}@{os.path.getmtime(sur.ckpt):.0f}"
    if os.path.exists(cache):
        res = json.load(open(cache))
        if res.get("key") == key:
            return res
    x, y, _, _ = get_split_arrays(sur.cls, "test", "co", convention=sur.convention,
                                  msdcnn_train=sur.msdcnn_train)     # the surrogate's own test set
    R = y[..., 0] + 1j * y[..., 1]
    pR = sur.predict(x)
    mt, mp = np.abs(R), np.abs(pR)
    res = dict(key=key, n=int(len(x)), split="test",
               lin_mse=float(np.mean((mp - mt) ** 2)),
               r2=float(1 - np.sum((mp - mt) ** 2) / np.sum((mt - mt.mean()) ** 2)),
               phase_mae_deg=float(np.degrees(np.abs(np.angle(pR * np.conj(R))).mean())))
    json.dump(res, open(cache, "w"), indent=2)
    return res


def database(classes=C.CLASSES, convention=None):
    """Every SUTD-PRCM layout (in the given convention) with its CST r_xx (101 points)."""
    xs, Rs, labels, idx = [], [], [], []
    for c in classes:
        im, cu = load_class(c)
        xs.append(to_convention(im, c, convention)); Rs.append(cu[:, 0])
        labels += [c] * len(im); idx.append(np.arange(len(im)))
    return np.concatenate(xs), np.concatenate(Rs), np.array(labels), np.concatenate(idx)
