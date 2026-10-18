"""
rdn_block_orientation.py — which reading of a stored RDN image reproduces its stored
spectra, for rows of the two blocks marked in outputs_new_pipeline/notes/rdn_not_inverted_mask.npy
and for ordinary RDN rows (control).

Four readings of the stored image s (rows = x, the incident field):
  s, flipud(s), 1 - s, flipud(1 - s)
|r_xx| decides copper = 1 or 0 (a mirror does not change r_xx); the sign of r_yx decides the
mirror (a mirror reverses it). FDTD solver (fdtd_solver/), h = 0.25 mm,
lossy copper, phase moved to the dataset's port plane (10.705 mm).

    python experiments/dataset_checks/rdn_block_orientation.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "fdtd_solver"))
from fdtd import Cell, to_port                          # noqa: E402

DATA = os.path.join(ROOT, "SUTD_DATASET", "RDNDATASET", "full_data_list")
OUT = os.path.dirname(os.path.abspath(__file__))
FREQ = np.linspace(2, 12, 1001) * 1e9
rng = np.random.default_rng(0)


def rows(folder, lo, hi, n):
    d = os.path.join(DATA, folder)
    img = np.load([os.path.join(d, f) for f in os.listdir(d) if f.startswith("Integrate_image")][0]).reshape(-1, 16, 16)
    crv = np.load([os.path.join(d, f) for f in os.listdir(d) if f.startswith("Integrate_curve")][0])
    cand = [i for i in range(lo, hi) if np.abs(crv[i, 1]).max() > 0.3 and (np.abs(crv[i]) ** 2).sum(0).max() <= 1.001]
    return [(folder, int(i), img[i].astype(bool), crv[i]) for i in rng.choice(cand, n, replace=False)]


cells = [("block 1",) + c for c in rows("Data_082", 0, 1000, 4)] + \
        [("block 2",) + c for c in rows("Data_087", 0, 1000, 4)] + \
        [("normal",) + c for c in rows("Data_001", 0, 1000, 3)]
HYP = {"stored": lambda s: s, "flipud": lambda s: s[::-1], "inverted": lambda s: ~s, "flipud(inverted)": lambda s: ~s[::-1]}
lay = [f(c[3]).T.copy() for f in HYP.values() for c in cells]            # solver convention: transposed
rxx, ryx = Cell(h=0.25e-3, copper_rs=0.026).simulate(torch.tensor(np.stack(lay), device="cuda"), FREQ, t_max=30e-9)
n = len(cells)
res = {}
for k, name in enumerate(HYP):
    for j, (grp, folder, i, _, crv) in enumerate(cells):
        a = to_port(rxx[k * n + j], FREQ, 10.705e-3)
        b = to_port(ryx[k * n + j], FREQ, 10.705e-3)
        res.setdefault(f"{grp} {folder} #{i}", {})[name] = dict(
            mse_abs_rxx=float(np.mean((np.abs(a) - np.abs(crv[0])) ** 2)),
            ryx_err_same_sign=float(np.mean(np.abs(b - crv[1]) ** 2)),
            ryx_err_opposite_sign=float(np.mean(np.abs(b + crv[1]) ** 2)))
json.dump(res, open(os.path.join(OUT, "rdn_block_orientation.json"), "w"), indent=1)
for cell, r in res.items():
    best = min(r, key=lambda h: r[h]["mse_abs_rxx"] + r[h]["ryx_err_same_sign"])
    print(f"{cell:28s} best: {best:17s} " + "  ".join(
        f"{h}: |rxx| {v['mse_abs_rxx']:.1e} ryx {v['ryx_err_same_sign']:.1e}" for h, v in r.items()))
