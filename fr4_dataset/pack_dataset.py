"""
pack_dataset.py — progress report and SUTD-format export of the FR-4 dataset.

    python pack_dataset.py            # status only
    python pack_dataset.py --pack     # also export the finished samples

The export mirrors SUTD-PRCM, so dataset.py of the surrogate can read it:
    packed_<tag>/RDNDATASET/full_data_list/Data_001/Integrate_image_1.npy  (n, 256) uint8, 1 = copper
                                                   /Integrate_curve_1.npy  (n, 2, 1001) complex128
curves[:, 0] = co-pol reflection r_xx (s11_co), curves[:, 1] = cross-pol (s11_cross).
Only complete blocks of 1000 consecutive designs are written as Data_XXX, plus
index.npy with the design index of every packed row. The per-sample files are
never modified.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

import fr4_config as C
from run_dataset import sample_path, sample_ok


def status():
    files = sorted(glob.glob(os.path.join(C.OUT_DIR, "sample_*.npz")))
    done = sorted(int(os.path.basename(f)[7:13]) for f in files)
    secs = []
    for f in files[-200:]:
        try:
            with np.load(f) as d:
                secs.append(float(d["sim_seconds"]))
        except Exception:
            pass
    locks = glob.glob(os.path.join(C.LOCK_DIR, "*.lock"))
    failed = {}
    fpath = os.path.join(C.LOG_DIR, "failed.csv")
    if os.path.exists(fpath):
        for row in csv.reader(open(fpath)):
            if row and row[0].isdigit():
                failed[int(row[0])] = failed.get(int(row[0]), 0) + 1
    target = C.N_SIMULATE
    n_target = sum(1 for i in done if i < target)
    print(f"dataset {C.TAG}: {len(done)} samples saved ({n_target}/{target} of the current target, "
          f"{100 * n_target / target:.1f} %)")
    if secs:
        m = np.mean(secs)
        print(f"  mean CST time per design (last {len(secs)}): {m:.0f} s "
              f"-> remaining {target - n_target} designs = {m * (target - n_target) / 3600:.0f} h on one worker")
    print(f"  running now (locks): {len(locks)};  failed designs: {len(failed)} "
          f"({sum(1 for v in failed.values() if v >= 3)} skipped after 3 tries)")
    return done


def pack(done):
    designs = np.load(C.DESIGNS_FILE)
    root = os.path.join(C.PACKED_DIR, "RDNDATASET", "full_data_list")
    os.makedirs(root, exist_ok=True)
    have = set(done)
    rows, blocks = [], 0
    for b in range(len(designs) // 1000):
        idx = list(range(b * 1000, (b + 1) * 1000))
        if not all(i in have for i in idx):
            continue
        img, cur = [], []
        for i in idx:
            if not sample_ok(sample_path(i), designs[i]):
                raise RuntimeError(f"sample {i} is damaged; run run_dataset.py --verify first")
            with np.load(sample_path(i)) as d:
                img.append(d["matrix"].reshape(-1))
                cur.append(np.stack([d["s11_co"], d["s11_cross"]]))
        out = os.path.join(root, f"Data_{b + 1:03d}")
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "Integrate_image_1.npy"), np.asarray(img, np.uint8))
        np.save(os.path.join(out, "Integrate_curve_1.npy"), np.asarray(cur, np.complex128))
        rows += idx
        blocks += 1
    np.save(os.path.join(C.PACKED_DIR, "index.npy"), np.asarray(rows, np.int64))
    stamp = json.load(open(os.path.join(C.OUT_DIR, "_dataset.json")))
    json.dump(dict(stamp, packed_samples=len(rows), blocks=blocks), open(os.path.join(C.PACKED_DIR, "dataset.json"), "w"),
              indent=2)
    print(f"packed {len(rows)} samples in {blocks} blocks of 1000 -> {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    d = status()
    if a.pack:
        pack(d)
