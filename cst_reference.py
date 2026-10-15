"""
cst_reference.py — CST check of a substrate thickness and the phase reference.

1) Simulates the given dataset layouts at --t (mm) and prints the |r_xx| MSE and
   mean phase difference against the dataset (1001 points).
2) Simulates the empty cell (substrate and ground only) at --t and fits the
   distance Delta between the Floquet-port plane and the cell surface, by
   matching CST to the analytic reflection of a grounded dielectric slab.

    python cst_reference.py --t 1.0
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

import config as C
from cst_verify import simulate
from cst_thickness_search import _layouts, _score

C0 = 299_792_458.0
ETA0 = 376.730313668


def slab_gamma(f_ghz, t_mm, eps=C.CST_SUB_EPS, tand=C.CST_SUB_TAND):
    """Reflection of a PEC-backed dielectric slab of thickness t at its surface."""
    k0 = 2 * np.pi * f_ghz * 1e9 / C0
    epc = eps * (1 - 1j * tand)
    beta = k0 * np.sqrt(epc)
    zin = 1j * ETA0 / np.sqrt(epc) * np.tan(beta * t_mm * 1e-3)
    return (zin - ETA0) / (zin + ETA0)


def fit_delta(sim, t_mm):
    f = sim["freq"]
    g = slab_gamma(f, t_mm)
    k0 = 2 * np.pi * f * 1e9 / C0

    def err(delta_mm):
        r = sim["s11"] * np.exp(2j * k0 * delta_mm * 1e-3)
        return np.degrees(np.abs(np.angle(r * np.conj(g)))).mean()

    res = minimize_scalar(err, bounds=(0.0, 30.0), method="bounded", options=dict(xatol=1e-4))
    return float(res.x), float(res.fun)


def main(args):
    out = os.path.join(C.CST_DIR, "calibration", f"reference_t{args.t:g}.json")
    rows = []
    for d in _layouts(args.layouts):
        r = _score(d, args.t)
        rows.append(dict(layout=f"{d['cls']}#{d['index']}", **r))
        print(f"t={args.t:g} mm  {rows[-1]['layout']:>11}  |r| MSE {r['mse']:.3e}  phase {r['phase']:.2f} deg", flush=True)
        json.dump(dict(t=args.t, layouts=rows), open(out, "w"), indent=1)
    empty = simulate(np.zeros((C.CST_GRID_SIZE, C.CST_GRID_SIZE), np.uint8), args.t,
                     os.path.join(C.CST_DIR, "reference", f"empty_t{args.t:g}.npz"))
    delta, fit = fit_delta(empty, args.t)
    print(f"empty cell t={args.t:g} mm: phase reference Delta = {delta:.3f} mm (mean residual {fit:.3f} deg)", flush=True)
    json.dump(dict(t=args.t, layouts=rows, delta_mm=delta, delta_residual_deg=fit), open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--layouts", nargs="+",
                    default=["RDN:10530", "RDN:22021", "RDN:101796", "PLR:7606", "PTN:29546", "PLG:5078"])
    main(ap.parse_args())
