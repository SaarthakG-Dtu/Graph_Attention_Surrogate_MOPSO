"""
verify_coding.py — select coding pairs from the coding searches and check their unit cells with FDTD.

Selection (as in sutd-prcm-solverfree): the non-dominated pairs of all coding runs; the 3 widest bands with
D <= 10, then the smallest penalty sum, 6 pairs. FDTD of A, B, A^T, B^T at h = 0.25 and 0.125 mm (lossy copper),
extrapolated to h -> 0; the reduction is scored with the -10 dB definition (CodingObjectives, search=False).
Writes verify_coding/shortlist.npz and verify.json (mesh shift) for fdtd_solver/coding_board.py.

    python experiments/devices/verify_coding.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fdtd_solver"))
import config as C                                       # noqa: E402
from phase2_devices import CodingObjectives              # noqa: E402
from phase2_mopso import non_dominated                   # noqa: E402
from fdtd import Cell, to_port                           # noqa: E402
from extrapolate import fit_scale, converged             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_coding")
os.makedirs(OUT, exist_ok=True)
F = np.linspace(2, 12, 101)
conv = json.load(open(os.path.join(ROOT, "fdtd_solver", "convention.json")))

pairs, Fp, V, src, Pred = [], [], [], [], []
for d in sorted(glob.glob(os.path.join(C.OUTPUT_ROOT, "PHASE2", "mopso", "coding_*"))):
    z = np.load(os.path.join(d, "Final", "pareto_designs.npz"))
    pairs.append(z["patterns"]); Fp.append(z["objectives"]); V.append(z["values"]); Pred.append(z["R"])
    src += [f"{os.path.basename(d)}#{i}" for i in range(len(z["patterns"]))]
pairs, Fp, V, Pred = map(np.concatenate, (pairs, Fp, V, Pred))
nd = np.where(non_dominated(Fp))[0]
wide = [i for i in nd[np.argsort(-V[nd, 0])] if V[i, 2] <= 10][:3]
rest = [i for i in nd[np.argsort(Fp[nd].sum(1))] if i not in wide][:6 - len(wide)]
sel = wide + rest
X = pairs[sel].astype(bool)                                         # (6, 2, 16, 16): A, B
cells = np.concatenate([X[:, 0], X[:, 1], X[:, 0].transpose(0, 2, 1), X[:, 1].transpose(0, 2, 1)])
n = len(X)


def fdtd(h):
    lay = np.stack([c.T if conv["x_along_rows"] else c for c in cells])
    rxx, ryx = Cell(h=h, copper_rs=0.026).simulate(torch.tensor(lay, device="cuda"), F * 1e9, t_max=30e-9)
    return np.stack([to_port(r, F * 1e9, conv["delta_mm"] * 1e-3) for r in (rxx, ryx)], 1)


Sc, Sf = fdtd(0.25e-3), fdtd(0.125e-3)
S0, scales = [], []
for k in range(len(cells)):
    s0, s, _ = converged(Sc[k], Sf[k], F)
    S0.append(np.where(np.isnan(s0), Sf[k], s0)); scales.append(s)
S0 = np.stack(S0)
S_cells = np.stack([S0[0:n], S0[n:2 * n], S0[2 * n:3 * n], S0[3 * n:]], 1)      # (6, 4, 2, K): A, B, A^T, B^T
rep = CodingObjectives("free", F, search=False)
Vf, _ = rep.values(S_cells, X.astype(np.uint8))
Vp, _ = rep.values(Pred[sel], X.astype(np.uint8))
has, flo, fhi = rep.band(S_cells)[:3]
delta_coarse_pct = float(100 * 4 * (np.median(scales) - 1) / 3)                 # mesh shift at h = 0.25 mm
np.savez(os.path.join(OUT, "shortlist.npz"), X=X.astype(np.uint8), info=np.array(src)[sel])
np.savez(os.path.join(OUT, "verify_spectra.npz"), freq=F, pred=Pred[sel], fdtd_cells_h0=S_cells, coarse=Sc, fine=Sf)
rows = [dict(design=i, source=src[sel[i]], B_pred=float(Vp[i, 0]), B_fdtd_cells=float(Vf[i, 0]),
             R_pred=float(Vp[i, 1]), R_fdtd_cells=float(Vf[i, 1]), D=int(Vf[i, 2]),
             band_fdtd=[float(flo[i]), float(fhi[i])]) for i in range(n)]
for r in rows:
    print(f"pair {r['design']} {r['source']:24s} B pred {r['B_pred']:5.1f}  FDTD cells {r['B_fdtd_cells']:5.1f} %  "
          f"R {r['R_pred']:.1f}/{r['R_fdtd_cells']:.1f} dB  D {r['D']}  band {r['band_fdtd'][0]:.2f}-{r['band_fdtd'][1]:.2f} GHz")
json.dump(dict(delta_coarse_pct=delta_coarse_pct, designs=rows), open(os.path.join(OUT, "verify.json"), "w"), indent=1)
print(f"mesh shift at h = 0.25 mm: {delta_coarse_pct:.2f} %")
