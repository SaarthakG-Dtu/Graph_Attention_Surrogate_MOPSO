"""
cst_coding.py — CST check of the coding pairs of verify_coding/ (cells only; the board is checked with FDTD).

For B = mirror(A), the spectra of B follow exactly from those of A (r_xx kept, r_yx reversed), so only A and
A^T are simulated (A alone for the diagonal pairs, A = A^T). Reduction from the area-average formula,
-10 dB definition (CodingObjectives, search=False), against the prediction and the FDTD of the cells.

    python experiments/devices/cst_coding.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config as C                                       # noqa: E402
from cst_verify import simulate                          # noqa: E402
from phase2_devices import CodingObjectives              # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_coding")
X = np.load(os.path.join(D, "shortlist.npz"))["X"]
z = np.load(os.path.join(D, "verify_spectra.npz"))
F = z["freq"]


def cst(L, name):
    s = simulate(L.astype(np.uint8), C.CST_SUB_T_MM, os.path.join(D, "cst", f"{name}_t{C.CST_SUB_T_MM:g}.npz"))
    it = lambda a: np.interp(F, s["freq"], a.real) + 1j * np.interp(F, s["freq"], a.imag)
    return np.stack([it(s["s11"]), C.CST_RYX_SIGN * it(s["s21"])])


flip = lambda S: S * np.array([1, -1])[:, None]
S = []
for i, (A, B) in enumerate(X):
    assert np.array_equal(B, A[:, ::-1]), "cst_coding.py assumes mirror pairs"
    SA = cst(A, f"pair{i}_A")
    SAT = SA if np.array_equal(A, A.T) else cst(A.T, f"pair{i}_AT")
    S.append(np.stack([SA, flip(SA), SAT, flip(SAT)]))
    print(f"pair {i} simulated", flush=True)
S = np.stack(S)
rep = CodingObjectives("mirror", F, search=False)
Vc, _ = rep.values(S, X); Vf, _ = rep.values(z["fdtd_cells_h0"], X); Vp, _ = rep.values(z["pred"], X)
_, lo, hi = rep.band(S)[:3]
rows = [dict(pair=i, B_pred=float(Vp[i, 0]), B_fdtd=float(Vf[i, 0]), B_cst=float(Vc[i, 0]), R_cst=float(Vc[i, 1]),
             band_cst=[float(lo[i]), float(hi[i])]) for i in range(len(X))]
for r in rows:
    print(f"pair {r['pair']}  B pred {r['B_pred']:5.1f}  FDTD cells {r['B_fdtd']:5.1f}  CST cells {r['B_cst']:5.1f} %  "
          f"R {r['R_cst']:.1f} dB  band {r['band_cst'][0]:.2f}-{r['band_cst'][1]:.2f} GHz")
np.save(os.path.join(D, "cst_cells.npy"), S)
json.dump(rows, open(os.path.join(D, "cst.json"), "w"), indent=1)
