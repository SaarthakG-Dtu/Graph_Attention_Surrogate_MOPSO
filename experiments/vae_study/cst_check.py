"""Quick comparison of finished CST runs with the surrogate for a MOIPSO run dir."""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C
from phase2_mopso import inphase_band, surface_phase_deg
run = sys.argv[1]
d = np.load(os.path.join(run, "Final", "pareto_designs.npz")); f = d["freq_ghz"]; f0 = C.OBJECTIVE_PRESETS["amc_x"][0]["f0"]
def BL(R):
    lo, hi, _, _ = inphase_band(R[None], f, f0)
    if lo[0] < 0: return 0.0, float("nan")
    fl, fh = f[lo[0]], f[hi[0]]; B = 200 * (fh - fl) / (fh + fl)
    return B, float(20 * np.log10(np.abs(R[lo[0]:hi[0] + 1])).min())
for p in sorted(glob.glob(os.path.join(run, "CST_verification", "runs", "solution_*_t1.npz"))):
    k = int(os.path.basename(p).split("_")[1]) - 1
    s = np.load(p); Rc = np.interp(f, s["freq"], s["s11"].real) + 1j * np.interp(f, s["freq"], s["s11"].imag)
    Rp = d["R"][k]
    mse = np.mean((np.abs(Rc) - np.abs(Rp)) ** 2)
    ph = np.mean(np.abs((surface_phase_deg(Rc[None], f)[0] - surface_phase_deg(Rp[None], f)[0] + 180) % 360 - 180))
    print(f"sol {k+1}: MSE {1e3*mse:.2f}e-3 phase {ph:.1f} deg | pred B {BL(Rp)[0]:.1f} L {BL(Rp)[1]:.2f} | CST B {BL(Rc)[0]:.1f} L {BL(Rc)[1]:.2f} | D {d['values'][k][2]:.0f} fill {d['patterns'][k].mean():.2f}")
