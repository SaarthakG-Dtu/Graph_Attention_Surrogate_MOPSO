"""
cst_thickness_grid.py — CST check of explicit substrate thicknesses (cached runs
are reused). Waits for another CST job to finish first if --wait_for is given.

    python cst_thickness_grid.py --wide 1.2 1.4 1.8 2.2 --fine 0.95 0.975 1.025 1.05
"""

import argparse
import json
import os
import time

import numpy as np

import config as C
from cst_thickness_search import _layouts, _score


def main(args):
    if args.wait_for:
        while not (os.path.exists(args.wait_for) and "delta_mm" in open(args.wait_for).read()):
            time.sleep(60)
    designs = _layouts(args.layouts)
    out = os.path.join(C.CST_DIR, "calibration", "thickness_grid.json")
    rows = []
    plan = [(t, designs[:1]) for t in args.wide] + [(t, designs) for t in sorted(set(args.fine) | {1.0})]
    for t, which in plan:
        for d in which:
            r = _score(d, t)
            rows.append(dict(layout=f"{d['cls']}#{d['index']}", t=t, **r))
            print(f"t={t:g} mm  {rows[-1]['layout']:>11}  |r| MSE {r['mse']:.3e}  phase {r['phase']:.2f} deg", flush=True)
            json.dump(rows, open(out, "w"), indent=1)
    for t in sorted(set(args.fine) | {1.0}):
        m = [r["mse"] for r in rows if r["t"] == t]
        if len(m) == len(designs):
            print(f"mean over {len(designs)} layouts  t={t:g} mm  |r| MSE {np.mean(m):.3e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", default=["RDN:10530", "RDN:22021", "RDN:101796"])
    ap.add_argument("--wide", type=float, nargs="*", default=[1.2, 1.4, 1.8, 2.2])
    ap.add_argument("--fine", type=float, nargs="*", default=[0.95, 0.975, 1.025, 1.05])
    ap.add_argument("--wait_for", default=None, help="JSON file of a running CST job; start when it has delta_mm")
    main(ap.parse_args())
