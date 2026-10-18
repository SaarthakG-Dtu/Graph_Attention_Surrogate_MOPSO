"""
search_sweep.py — latent MOIPSO at the standard budget (200 particles x 150
generations = 30,200 layouts) for several box sizes and maximum velocities, and
seeds; compared by the hypervolume of the final archive.

    python experiments/vae_study/search_sweep.py --vae_tag rdn_d150
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config as C                                  # noqa: E402
import phase2_mopso as M                            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep")


def run(vae_tag, bound, maxvel, seed, space="latent"):
    tag = f"vs_{vae_tag}_{space}_B{bound:g}_v{maxvel:g}_s{seed}" + ""
    keep = C.mopso_dir(C.MOPSO_PROBLEM, tag)            # results stay in the MOIPSO output folder
    if os.path.exists(os.path.join(keep, "comparison.json")):
        return json.load(open(os.path.join(keep, "comparison.json")))
    C.LATENT_BOUND, C.LATENT_MAXVEL, C.PIXEL_MAXVEL = bound, maxvel, maxvel / (2 * bound)
    a = argparse.Namespace(problem=C.MOPSO_PROBLEM, space=space, surrogate=C.MOPSO_SURROGATE, vae_tag=vae_tag,
                           tag=tag, database_only=False, gens=150, swarm=200, seed=seed)
    C.MOPSO_MAXGEN, C.MOPSO_NP, C.MOPSO_SEED = 150, 200, seed
    M.main(a)
    return json.load(open(os.path.join(keep, "comparison.json")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_tag", default="rdn_d150")
    ap.add_argument("--bounds", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--maxvels", type=float, nargs="+", default=[0.1, 0.2, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--space", default="latent", choices=["latent", "binary"])
    ap.add_argument("--name", default=None, help="name of the results file")
    a = ap.parse_args()
    rows = []
    for B in a.bounds:
        for v in a.maxvels:
            hv = [run(a.vae_tag, B, v, s, a.space)["hv"] for s in a.seeds]
            rows.append(dict(vae=a.vae_tag, space=a.space, bound=B, maxvel=v, hv=hv, mean=sum(hv) / len(hv)))
            print(f"RESULT {a.space} vae={a.vae_tag} B={B:g} vmax={v:g}: HV {['%.3f' % h for h in hv]} "
                  f"mean {rows[-1]['mean']:.3f}", flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(OUT), f"sweep_{a.name or a.vae_tag}.json"), "w"), indent=1)
