"""
cst_thickness_search.py — find the substrate thickness of the SUTD-PRCM cell
that best reproduces the dataset (whose spectra are CST results themselves).

Stage 1: coarse sweep over --coarse thicknesses on the first layout.
Stage 2: golden-section search around the best coarse value, scored on all
         layouts (mean MSE of |r_xx| over the dataset's 1001 points).
Runs are cached in CST/calibration/runs (same names as cst_verify calibrate),
so thicknesses that were simulated before are not repeated.

    python cst_thickness_search.py
    python cst_thickness_search.py --layouts RDN:10530 RDN:22021 RDN:101796 --coarse 0.6 0.8 1.0 1.4 1.8 2.2
"""

import argparse
import json
import os

import numpy as np

import config as C
from cst_verify import simulate
from dataset import load_class, load_full_curves, to_convention

OUT = os.path.join(C.CST_DIR, "calibration")
F_GT = np.linspace(2, 12, 1001)


def _layouts(specs):
    out = []
    for s in specs:
        cls, k = s.split(":"); k = int(k)
        im, _ = load_class(cls)
        copper = to_convention(im[k:k + 1], cls, "physical")[0]      # 1 = copper, as drawn in CST
        tag = f"{cls}_{k}" + ("_invert" if cls == "RDN" else "")       # cache names of cst_verify calibrate
        out.append(dict(cls=cls, index=k, copper=copper, tag=tag, R=load_full_curves(cls, [k])[0][0]))
    return out


def _score(d, t):
    sim = simulate(d["copper"], t, os.path.join(OUT, "runs", f"{d['cls']}_{d['index']}_t{t:g}"
                                                 + ("_invert" if d["cls"] == "RDN" else "") + ".npz"))
    R = np.interp(F_GT, sim["freq"], sim["s11"].real) + 1j * np.interp(F_GT, sim["freq"], sim["s11"].imag)
    return dict(mse=float(np.mean((np.abs(R) - np.abs(d["R"])) ** 2)),
                phase=float(np.degrees(np.abs(np.angle(R * np.conj(d["R"])))).mean()))


def main(args):
    designs = _layouts(args.layouts)
    log = []

    def evaluate(t, which):
        t = round(float(t), 3)
        rows = []
        for d in which:
            r = _score(d, t)
            rows.append(dict(layout=f"{d['cls']}#{d['index']}", t=t, **r))
            log.append(rows[-1])
            print(f"  t={t:.3f} mm  {rows[-1]['layout']:>11}  |r| MSE {r['mse']:.3e}  phase {r['phase']:.1f} deg",
                  flush=True)
        json.dump(log, open(os.path.join(OUT, "thickness_search.json"), "w"), indent=1)
        return float(np.mean([r["mse"] for r in rows]))

    print(f"[stage 1] coarse sweep on {designs[0]['tag']}")
    coarse = {t: evaluate(t, designs[:1]) for t in args.coarse}
    t0 = min(coarse, key=coarse.get)
    lo, hi = max(min(args.coarse), t0 - args.width), min(max(args.coarse), t0 + args.width)
    print(f"[stage 2] golden-section search on [{lo:.2f}, {hi:.2f}] mm, all {len(designs)} layouts")
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = evaluate(c, designs), evaluate(d, designs)
    while b - a > args.tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = evaluate(c, designs)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = evaluate(d, designs)
    best = c if fc < fd else d
    f0 = evaluate(t0, designs)                                     # coarse winner on all layouts, for reference
    summary = dict(coarse=coarse, golden_interval=[a, b], best_t=round(best, 3),
                   best_mean_mse=min(fc, fd), coarse_best_t=t0, coarse_best_mean_mse=f0)
    json.dump(dict(summary=summary, runs=log), open(os.path.join(OUT, "thickness_search.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", default=["RDN:10530", "RDN:22021", "RDN:101796"])
    ap.add_argument("--coarse", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.4, 1.8, 2.2])
    ap.add_argument("--width", type=float, default=0.2, help="half-width of the refinement interval (mm)")
    ap.add_argument("--tol", type=float, default=0.04, help="final interval width (mm)")
    main(ap.parse_args())
