"""
rotating_cells.py — RDN cells (all 111,000; most are training cells) that rotate polarization near 10 GHz (PCR > 0.9 over 9.8-10.6 GHz): mean
|r_yx|^2 over 9.5-10.9 GHz from CST (dataset), from the four-output RDN108000 and from FDTD (h -> 0).

    python experiments/devices/rotating_cells.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fdtd_solver"))
import config as C                                       # noqa: E402
from dataset import get_split_arrays                     # noqa: E402
from phase1_forward_model import load_surrogate          # noqa: E402
from train_phase1 import predict                         # noqa: E402
from fdtd import Cell, to_port                           # noqa: E402
from extrapolate import converged                        # noqa: E402

parts = [get_split_arrays("RDN", s, "both", "true", msdcnn_train=108000)[:2] for s in ("train", "test")]
x, y = np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])   # all RDN cells (rare: mostly training cells)
Y = y[..., 0] + 1j * y[..., 1]
F = np.linspace(2, 12, 101)
P = (np.abs(Y) ** 2).sum(1); PCR = np.abs(Y[:, 1]) ** 2 / np.maximum(P, 1e-12)
cand = np.where((PCR[:, (F >= 9.8) & (F <= 10.6)].min(1) > 0.9) & (P.max(1) <= 1.001))[0]
pick = np.random.default_rng(0).choice(cand, min(16, len(cand)), replace=False)
net = load_surrogate(os.path.join(C.phase1_dir("RDN", "both", "n108000"), "phase1_forward_best.pth"))
p = predict(net, x[pick].astype(np.float32), tta=True, target="both"); Sp = p[..., 0] + 1j * p[..., 1]
Ss = {}
for h in (0.25e-3, 0.125e-3):
    rxx, ryx = Cell(h=h, copper_rs=0.026).simulate(torch.tensor(x[pick].transpose(0, 2, 1).astype(bool).copy(), device="cuda"),
                                                   F * 1e9, t_max=30e-9)
    Ss[h] = np.stack([to_port(rxx, F * 1e9, 10.705e-3), to_port(ryx, F * 1e9, 10.705e-3)], 1)
S0 = np.stack([np.where(np.isnan(s := converged(Ss[0.25e-3][j], Ss[0.125e-3][j], F)[0]), Ss[0.125e-3][j], s)
               for j in range(len(pick))])
m = (F >= 9.5) & (F <= 10.9)
E = lambda S: (np.abs(S[:, 1]) ** 2)[:, m].mean(1)
res = dict(n=int(len(pick)), n_candidates=int(len(cand)), E_cst=E(Y[pick]).tolist(), E_net=E(Sp).tolist(), E_fdtd=E(S0).tolist(),
           net_minus_cst=float(np.mean(E(Sp) - E(Y[pick]))), fdtd_minus_cst=float(np.mean(E(S0) - E(Y[pick]))))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotating_cells.json"), "w"), indent=1)
print(f"{res['n']} of {res['n_candidates']} rotating test cells: mean |r_yx|^2 network - CST {res['net_minus_cst']:+.3f}, "
      f"FDTD - CST {res['fdtd_minus_cst']:+.3f}")
