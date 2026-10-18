"""
scaling.py — VAE space vs pixel space, each with its best setting from the
sweeps, at growing budgets (swarm x generations). Reports the archive
hypervolume, the wall time, and the number of layouts and seconds each search
needs to reach given hypervolume targets (front of all scored layouts).

    python experiments/vae_study/scaling.py
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper"))
import config as C                                  # noqa: E402
import phase2_mopso as M                            # noqa: E402

BUDGETS = [(200, 150), (500, 300), (1000, 300), (2000, 300)]
TARGETS = [0.35, 0.40, 0.45]


def tiny_sweep():
    """Smaller latent boxes and steps (run before the scaling study)."""
    from search_sweep import run as sweep_run
    out = os.path.join(HERE, "sweep_rdn_tiny.json")
    if os.path.exists(out):
        return
    rows = []
    for B in (0.5, 0.75):
        for v in (0.02, 0.05):
            hv = [sweep_run("rdn_d150", B, v, s)["hv"] for s in (0, 1, 2)]
            rows.append(dict(vae="rdn_d150", space="latent", bound=B, maxvel=v, hv=hv, mean=sum(hv) / len(hv)))
            print(f"RESULT latent B={B:g} vmax={v:g}: HV {['%.3f' % h for h in hv]} mean {rows[-1]['mean']:.3f}", flush=True)
            json.dump(rows, open(out, "w"), indent=1)


def best(name):
    rows = json.load(open(os.path.join(HERE, f"sweep_{name}.json")))
    if name == "rdn_d150":
        for extra in ("sweep_rdn_small.json", "sweep_rdn_tiny.json"):
            if os.path.exists(os.path.join(HERE, extra)):
                rows += json.load(open(os.path.join(HERE, extra)))
    r = max(rows, key=lambda r: r["mean"])
    return r["bound"], r["maxvel"], r["mean"]


def run(space, bound, maxvel, Np, gens, seed):
    tag = f"sc_{space}_N{Np}_G{gens}_s{seed}" + "" + (
        f"_B{bound:g}" if space == "latent" and bound != 0.5 else "")
    d = C.mopso_dir(C.MOPSO_PROBLEM, tag)
    if not os.path.exists(os.path.join(d, "comparison.json")):
        C.LATENT_BOUND, C.LATENT_MAXVEL, C.PIXEL_MAXVEL = bound, maxvel, maxvel / (2 * bound)
        C.MOPSO_MAXGEN, C.MOPSO_NP, C.MOPSO_SEED = gens, Np, seed
        M.main(argparse.Namespace(problem=C.MOPSO_PROBLEM, space=space, surrogate=C.MOPSO_SURROGATE,
                                  vae_tag=C.MOPSO_VAE_TAG, tag=tag, database_only=False,
                                  gens=gens, swarm=Np, seed=seed))
    comp = json.load(open(os.path.join(d, "comparison.json")))
    from make_figures import _front_history
    h = np.load(os.path.join(d, "swarm_history.npz"))
    hv, _ = _front_history(h["objectives"], h["generation"], 5)
    evals = (hv[:, 0] + 1) * Np
    reach = {}
    for tg in TARGETS:
        k = np.flatnonzero(hv[:, 1] >= tg)
        reach[str(tg)] = None if not len(k) else dict(
            layouts=int(evals[k[0]]), seconds=float(comp["seconds"] * evals[k[0]] / evals[-1]))
    return dict(tag=tag, space=space, Np=Np, gens=gens, seed=seed, layouts=int(comp["evaluations"]),
                seconds=float(comp["seconds"]), hv_archive=float(comp["hv"]), hv_all=float(hv[-1, 1]), reach=reach)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    sel = os.path.join(HERE, "scaling_settings.json")       # written once the sweeps are done
    if not os.path.exists(sel):
        sys.exit("scaling_settings.json missing")
    s = json.load(open(sel))
    C.MOPSO_VAE_TAG = s["vae"]
    cfg = {"latent": tuple(s["latent"]), "binary": tuple(s["binary"]) if "binary" in s else best("binary")}
    print("settings:", cfg, flush=True)
    rows = []
    for Np, gens in BUDGETS:
        for space in ("latent", "binary"):
            B, v, _ = cfg[space]
            for s in a.seeds:
                r = run(space, B, v, Np, gens, s)
                rows.append(r)
                print(f"RESULT {space:6s} {Np}x{gens} seed {s}: {r['layouts']:7d} layouts {r['seconds']:6.1f} s  "
                      f"HV archive {r['hv_archive']:.3f}  all {r['hv_all']:.3f}  reach {r['reach']}", flush=True)
                json.dump(dict(settings=cfg, runs=rows), open(os.path.join(HERE, "scaling%s.json" % ""), "w"), indent=1)
