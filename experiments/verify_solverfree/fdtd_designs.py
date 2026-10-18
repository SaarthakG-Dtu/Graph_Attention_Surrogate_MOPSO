"""
fdtd_designs.py — re-simulate, on this machine, the shortlisted LP-to-CP converters and rotators of
sutd-prcm-solverfree with its FDTD solver (h = 0.25 and 0.125 mm, copper R_s = 0.026 ohm), extrapolate
to h -> 0, score with the standard objectives, and compare with the results stored in that repo.
No network is needed: the predicted spectra stored with the repo's verification are used.

    python experiments/verify_solverfree/fdtd_designs.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF = os.path.join(ROOT, "external", "sutd-prcm-solverfree")
sys.path.insert(0, SF)
from fdtd import Cell, to_port                                   # noqa: E402
from extrapolate import converged, mse_abs                       # noqa: E402
from objectives import cp_objectives, rotator_objectives         # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
F = np.linspace(2, 12, 101) * 1e9
conv = json.load(open(os.path.join(SF, "fdtd_checks", "convention.json")))
RUNS = {"cp_centered_v2": ("cp", 10.0), "rotator_10g2_v2": ("rotator", 10.2)}


def fdtd(X, h):
    lay = np.stack([x.T if conv["x_along_rows"] else x for x in X])
    rxx, ryx = Cell(h=h, copper_rs=0.026).simulate(torch.tensor(lay, device="cuda"), F, t_max=30e-9)
    rxx, ryx = (to_port(r, F, conv["delta_mm"] * 1e-3) for r in (rxx, ryx))
    if conv["conjugate"]:
        rxx, ryx = np.conj(rxx), np.conj(ryx)
    return np.stack([rxx, ryx], 1)


def score(fam, f0, S, lay):
    f0_idx = int(round((f0 - 2) / 0.1))
    o = (cp_objectives(torch.tensor(S), lay, f0_idx=f0_idx) if fam == "cp"
         else rotator_objectives(torch.tensor(S), lay, f0_idx=f0_idx, pcr_min=0.9))
    q = "AR" if fam == "cp" else "E"
    lo, hi = o["f_lo"].numpy(), o["f_hi"].numpy()
    inb = np.isfinite(lo) & (o["B"].numpy() > 0)
    return [dict(design=i, B=float(o["B"][i]), quality=float(o[q][i]), D=int(o["D"][i]), meets=bool(o["meets_spec"][i]),
                 margin_pct=float(100 * min(f0 - lo[i], hi[i] - f0) / f0) if inb[i] else 0.0)
            for i in range(len(lay))]


res = {}
for tag, (fam, f0) in RUNS.items():
    S = os.path.join(SF, "search", tag)
    X = np.load(os.path.join(S, "shortlist.npz"))["X"].astype(bool)
    ref = np.load(os.path.join(S, "verify_spectra_cu.npz"))
    Sc, Sf = fdtd(X, 0.25e-3), fdtd(X, 0.125e-3)
    S0 = []
    for i in range(len(X)):
        s0, _, _ = converged(Sc[i], Sf[i], F / 1e9)
        S0.append(np.where(np.isnan(s0), Sf[i], s0))
    S0 = np.stack(S0)
    np.savez(os.path.join(OUT, f"designs_{tag}.npz"), coarse=Sc, fine=Sf, extrap=S0, freq=F)
    lay = torch.tensor(X)
    ours = score(fam, f0, S0, lay)
    theirs = json.load(open(os.path.join(S, "rescore_cu.json")))["fdtd_h0_extrap"]["designs"]
    res[tag] = dict(
        max_abs_diff_vs_stored=dict(coarse=float(np.abs(Sc - ref["fdtd_h250um"]).max()),
                                    fine=float(np.abs(Sf - ref["fdtd_h125um"]).max())),
        mse_extrap_vs_surrogate=[mse_abs(S0[i], ref["pred"][i]) for i in range(len(X))],
        n_meet=sum(r["meets"] for r in ours), n=len(X),
        B_range=[min(r["B"] for r in ours), max(r["B"] for r in ours)],
        quality_range=[min(r["quality"] for r in ours), max(r["quality"] for r in ours)],
        D_range=[min(r["D"] for r in ours), max(r["D"] for r in ours)],
        margin_range=[min(r["margin_pct"] for r in ours), max(r["margin_pct"] for r in ours)],
        repo_B=[t["B"] for t in theirs], ours=ours)
    r = res[tag]
    print(f"{tag}: meet {r['n_meet']}/{r['n']}  B {r['B_range'][0]:.1f}-{r['B_range'][1]:.1f} %  "
          f"quality {r['quality_range'][0]:.2f}-{r['quality_range'][1]:.2f}  D {r['D_range']}  "
          f"margin {r['margin_range'][0]:.1f}-{r['margin_range'][1]:.1f} %  | max diff vs stored FDTD "
          f"{r['max_abs_diff_vs_stored']}", flush=True)
json.dump(res, open(os.path.join(OUT, "fdtd_designs.json"), "w"), indent=1)
