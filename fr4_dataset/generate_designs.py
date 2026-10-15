"""
generate_designs.py — N_DESIGNS distinct random 16x16 layouts (SUTD RDN class).

Every pixel is copper with probability FILL_PROB = 0.5, independently, as in
SUTD-PRCM RDN (mean fill 0.4995, per-design std 0.031 = sqrt(0.25/256)).
All-empty and all-copper layouts are excluded, and every layout is distinct.

The design file is fixed once the CST sweep has started: sample_XXXXXX.npz
refers to row XXXXXX of it. This script therefore refuses to overwrite an
existing design file (use --force only before any sample exists).

    python generate_designs.py
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

import fr4_config as C


def sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def generate(n, p, seed, grid=C.GRID_SIZE):
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n:
        batch = (rng.random((n, grid, grid)) < p).astype(np.uint8)
        for m in batch:
            s = int(m.sum())
            if s == 0 or s == grid * grid:
                continue
            key = m.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
            if len(out) == n:
                break
    return np.stack(out)


def main(args):
    if os.path.exists(C.DESIGNS_FILE) and not args.force:
        sys.exit(f"{C.DESIGNS_FILE} already exists; the CST samples refer to its rows. "
                 f"Use --force only if no sample has been simulated yet.")
    if args.force and os.path.isdir(C.OUT_DIR) and any(f.startswith("sample_") for f in os.listdir(C.OUT_DIR)):
        sys.exit(f"{C.OUT_DIR} already holds samples of the current design file; refusing to replace it.")

    d = generate(C.N_DESIGNS, C.FILL_PROB, C.DESIGN_SEED)
    fill = d.reshape(len(d), -1).mean(1)
    assert len(np.unique(d.reshape(len(d), -1), axis=0)) == len(d)
    np.save(C.DESIGNS_FILE, d)
    meta = dict(n=len(d), grid=C.GRID_SIZE, fill_prob=C.FILL_PROB, seed=C.DESIGN_SEED,
                convention="1 = copper, 0 = no copper", distinct=True,
                excluded="all-empty and all-copper layouts",
                fill_mean=float(fill.mean()), fill_std=float(fill.std()),
                fill_min=float(fill.min()), fill_max=float(fill.max()),
                sha256=sha256(d))
    json.dump(meta, open(C.DESIGNS_META, "w"), indent=2)
    print(f"saved {len(d)} distinct designs -> {C.DESIGNS_FILE}")
    print(f"fill: mean {fill.mean():.4f}, std {fill.std():.4f} "
          f"(SUTD RDN: 0.4995, 0.031), min {fill.min():.3f}, max {fill.max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
