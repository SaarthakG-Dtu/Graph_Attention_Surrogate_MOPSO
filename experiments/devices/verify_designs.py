"""
verify_designs.py — select the converter / rotator designs of the searches and check them with FDTD.

Selection (as in sutd-prcm-solverfree): the 6 designs of the best-HV run with the smallest penalty sum,
then designs of all runs predicted to meet the spec (search definitions), by penalty sum, at most 12.
FDTD at h = 0.25 and 0.125 mm (lossy copper), extrapolated to h -> 0; all reported objectives use the
standard definitions (phase2_devices, search=False), for the prediction and for FDTD.

    python experiments/devices/verify_designs.py cp
    python experiments/devices/verify_designs.py rotator
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
from phase2_devices import DeviceObjectives              # noqa: E402
from fdtd import Cell, to_port                           # noqa: E402
from extrapolate import converged, mse_abs               # noqa: E402

dev = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"verify_{dev}")
os.makedirs(OUT, exist_ok=True)
F = np.linspace(2, 12, 101)
conv = json.load(open(os.path.join(ROOT, "fdtd_solver", "convention.json")))

# ---- selection
runs = {}
for d in sorted(glob.glob(os.path.join(C.OUTPUT_ROOT, "PHASE2", "mopso", f"{dev}_*"))):
    f = os.path.join(d, "Final", "pareto_designs.npz")
    if os.path.exists(f) and "stepcheck" not in d:
        runs[os.path.basename(d)] = (np.load(f), json.load(open(os.path.join(d, "comparison.json")))["hv"])
best = max(runs, key=lambda k: runs[k][1])
search = DeviceObjectives(dev, F, search=True)
picked, seen = [], set()


def add(run, i):
    key = runs[run][0]["patterns"][i].tobytes()
    if key not in seen:
        seen.add(key); picked.append((run, i))


z = runs[best][0]
for i in np.argsort(z["objectives"].sum(1))[:6]:
    add(best, i)
cand = [(runs[r][0]["objectives"][i].sum(), r, i) for r in runs for i in range(len(runs[r][0]["patterns"]))
        if search.good(runs[r][0]["values"][i:i + 1])[0]]
for _, r, i in sorted(cand):
    if len(picked) >= 12:
        break
    add(r, i)
X = np.stack([runs[r][0]["patterns"][i] for r, i in picked]).astype(bool)
P = np.stack([runs[r][0]["R"][i] for r, i in picked])                    # predicted (four-mirror average)


# ---- FDTD
def fdtd(h):
    lay = np.stack([x.T if conv["x_along_rows"] else x for x in X])
    rxx, ryx = Cell(h=h, copper_rs=0.026).simulate(torch.tensor(lay, device="cuda"), F * 1e9, t_max=30e-9)
    return np.stack([to_port(r, F * 1e9, conv["delta_mm"] * 1e-3) for r in (rxx, ryx)], 1)


Sc, Sf = fdtd(0.25e-3), fdtd(0.125e-3)
S0, delta = [], []
for i in range(len(X)):
    s0, _, d = converged(Sc[i], Sf[i], F)
    S0.append(np.where(np.isnan(s0), Sf[i], s0)); delta.append(d)
S0 = np.stack(S0)
np.savez(os.path.join(OUT, "spectra.npz"), X=X, pred=P, coarse=Sc, fine=Sf, extrap=S0, freq=F,
         source=np.array([f"{r}#{i}" for r, i in picked]))

rep = DeviceObjectives(dev, F, search=False)
Vp, _ = rep.values(P, X.astype(np.uint8)); Vf, _ = rep.values(S0, X.astype(np.uint8))
_, plo, phi = rep.band(P)[:3]; _, flo, fhi = rep.band(S0)[:3]
rows = []
for i in range(len(X)):
    rows.append(dict(design=i + 1, source=f"{picked[i][0]}#{picked[i][1]}",
                     pred=dict(zip(rep.names, map(float, Vp[i]))), fdtd=dict(zip(rep.names, map(float, Vf[i]))),
                     band_pred=[float(plo[i]), float(phi[i])], band_fdtd=[float(flo[i]), float(fhi[i])],
                     meets_pred=bool(rep.good(Vp[i:i + 1])[0]), meets_fdtd=bool(rep.good(Vf[i:i + 1])[0]),
                     mse_abs=mse_abs(S0[i], P[i]), mesh_shift_pct=100 * float(delta[i])))
    r = rows[-1]
    print(f"design {i + 1:2d} {r['source']:28s} B pred {Vp[i, 0]:5.1f} FDTD {Vf[i, 0]:5.1f} %  "
          f"quality {Vp[i, 1]:.2f}/{Vf[i, 1]:.2f}  D {int(Vf[i, 2])}  band FDTD {flo[i]:.2f}-{fhi[i]:.2f} GHz  "
          f"meets {r['meets_fdtd']}  MSE {r['mse_abs']:.1e}", flush=True)
json.dump(dict(device=dev, best_run=best, designs=rows, n_meet_fdtd=sum(r["meets_fdtd"] for r in rows)),
          open(os.path.join(OUT, "verify.json"), "w"), indent=1)
print(f"{dev}: {sum(r['meets_fdtd'] for r in rows)}/{len(rows)} meet the target in FDTD (h -> 0); best run {best}")
