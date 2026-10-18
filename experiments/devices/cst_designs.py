"""
cst_designs.py — CST check of the converter / rotator designs verified with FDTD (verify_<device>/spectra.npz).

Each layout (true convention, 1 = copper, incident field along the rows) is simulated as is at the substrate
thickness config.CST_SUB_T_MM; r_xx = S(mode 1 -> mode 1), r_yx = CST_RYX_SIGN * S(mode 2 -> mode 1), both at
the dataset's port plane, interpolated to the 101-point grid and scored with the standard definitions.

    python experiments/devices/cst_designs.py cp
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fdtd_solver"))
import config as C                                       # noqa: E402
from cst_verify import simulate                          # noqa: E402
from phase2_devices import DeviceObjectives              # noqa: E402
from extrapolate import mse_abs                          # noqa: E402

dev = sys.argv[1]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"verify_{dev}")
z = np.load(os.path.join(D, "spectra.npz"))
F = z["freq"]
S = []
for i, x in enumerate(z["X"]):
    s = simulate(x.astype(np.uint8), C.CST_SUB_T_MM, os.path.join(D, "cst", f"design_{i + 1:02d}_t{C.CST_SUB_T_MM:g}.npz"))
    it = lambda a: np.interp(F, s["freq"], a.real) + 1j * np.interp(F, s["freq"], a.imag)
    S.append(np.stack([it(s["s11"]), C.CST_RYX_SIGN * it(s["s21"])]))
    print(f"design {i + 1} simulated", flush=True)
S = np.stack(S)
rep = DeviceObjectives(dev, F, search=False)
X = z["X"].astype(np.uint8)
Vc, _ = rep.values(S, X); Vf, _ = rep.values(z["extrap"], X); Vp, _ = rep.values(z["pred"], X)
_, clo, chi = rep.band(S)[:3]
rows = [dict(design=i + 1, cst=dict(zip(rep.names, map(float, Vc[i]))), band_cst=[float(clo[i]), float(chi[i])],
             B_pred=float(Vp[i, 0]), B_fdtd=float(Vf[i, 0]), meets_cst=bool(rep.good(Vc[i:i + 1])[0]),
             mse_cst_vs_pred=mse_abs(S[i], z["pred"][i]), mse_cst_vs_fdtd=mse_abs(S[i], z["extrap"][i]))
        for i in range(len(X))]
for r, v in zip(rows, Vc):
    print(f"design {r['design']:2d}  B pred {r['B_pred']:5.1f}  FDTD {r['B_fdtd']:5.1f}  CST {v[0]:5.1f} %  quality {v[1]:.2f}  "
          f"meets {r['meets_cst']}  MSE vs pred {r['mse_cst_vs_pred']:.1e}  vs FDTD {r['mse_cst_vs_fdtd']:.1e}")
np.save(os.path.join(D, "cst_spectra.npy"), S)
json.dump(dict(device=dev, t_sub_mm=C.CST_SUB_T_MM, designs=rows, n_meet_cst=sum(r["meets_cst"] for r in rows)),
          open(os.path.join(D, "cst.json"), "w"), indent=1)
print(f"{dev}: {sum(r['meets_cst'] for r in rows)}/{len(rows)} meet the target in CST")
