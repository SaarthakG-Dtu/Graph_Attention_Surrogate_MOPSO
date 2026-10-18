"""
coding_baseline.py — library baseline for the coding checkerboard: the best pair a designer could pick from the
111,000 CST-simulated RDN cells and their mirror images (whose response follows exactly: r_xx kept, r_yx reversed).
Method of sutd-prcm-solverfree/coding_baseline.py, on our data (true convention, passive rows).

The library has x-incidence only, so pairs are scored on s_x alone, which favours the baseline over our designs:
  * mirror pairs (B = mirror(A)): every cell;
  * general pairs: for each A and four frequency windows, the 16 partners nearest to -A (|A + B|^2 = |A|^2 + |B|^2 +
    2 Re<A, B>, one fp16 matrix product per chunk), then every candidate scored exactly.

    python experiments/devices/coding_baseline.py   -> experiments/devices/baseline_coding.json, baseline_coding_pairs.npz
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dataset import load_class, to_convention, freq_grid          # noqa: E402
from phase2_devices import CodingObjectives, mirror               # noqa: E402
from phase2_mopso import non_dominated, hypervolume               # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
F = freq_grid()
im, cu = load_class("RDN")
Xall = to_convention(im, "RDN", "true", np.arange(len(im))).astype(np.uint8)
Yall = cu[:, :2].astype(np.complex64)
idx_ok = np.where((np.abs(Yall) ** 2).sum(1).max(1) <= 1.01)[0]            # passive rows
X, Y = Xall[idx_ok], Yall[idx_ok]
N = len(Y)
ob = CodingObjectives("free", F, search=False)
flip = lambda S: S * np.array([1, -1], np.float32)[None, :, None]


def score(ia, ib, mb):
    """Pairs (A = ia, B = ib, mirrored where mb), x-incidence only (transposes := cells, so s_y = s_x)."""
    SA = Y[ia]; SB = np.where(mb[:, None, None], flip(Y[ib]), Y[ib])
    LB = np.where(mb[:, None, None], mirror(X[ib]), X[ib])
    S = np.stack([SA, SB, SA, SB], 1)
    V, viol = ob.values(S, np.stack([X[ia], LB], 1))
    P = np.clip((V - ob.ideal) / (ob.worst - ob.ideal), 0, 1) + viol[:, None]
    return P, V


t0 = time.time()
ar = np.arange(N)
P_m, V_m = score(ar, ar, np.ones(N, bool))
print(f"mirror pairs: best B {V_m[:, 0].max():.1f} %  ({time.time() - t0:.0f} s)", flush=True)
Yt = torch.tensor(Y, device=dev)
Pall = torch.cat([Yt, torch.stack([Yt[:, 0], -Yt[:, 1]], 1)])             # partners: library + mirrors
cand_a, cand_b = [], []
for lo_f, hi_f in ((4, 12), (6, 12), (8, 12), (5, 10)):
    w = torch.tensor(np.where((F >= lo_f) & (F <= hi_f))[0][::2], device=dev)
    feat = lambda S: torch.cat([S[:, :, w].real, S[:, :, w].imag], 1).flatten(1)
    PA = feat(Yt).half(); PB = feat(Pall).half(); nB = (PB.float() ** 2).sum(1)
    for i in range(0, N, 1024):
        d = nB[None] + 2 * (PA[i:i + 1024] @ PB.T).float()                  # |A + B|^2 - |A|^2
        top = d.topk(16, largest=False).indices
        cand_a.append(np.repeat(np.arange(i, i + len(top)), 16)); cand_b.append(top.flatten().cpu().numpy())
        del d
    print(f"window {lo_f}-{hi_f} GHz done ({time.time() - t0:.0f} s)", flush=True)
ca, cb = np.concatenate(cand_a), np.concatenate(cand_b)
cm = cb >= N; cb = cb % N
keep = ~((cb == ca) & ~cm)                                                   # A with itself is not a pair
ca, cb, cm = ca[keep], cb[keep], cm[keep]
Pg, Vg = [], []
for i in range(0, len(ca), 65536):
    p, v = score(ca[i:i + 65536], cb[i:i + 65536], cm[i:i + 65536]); Pg.append(p); Vg.append(v)
Pg, Vg = np.concatenate(Pg), np.concatenate(Vg)
print(f"general pairs scored: {len(ca)}; best B {Vg[:, 0].max():.1f} %  ({time.time() - t0:.0f} s)", flush=True)
pen = np.concatenate([P_m, Pg]); Vv = np.concatenate([V_m, Vg])
A_all = np.concatenate([ar, ca]); B_all = np.concatenate([ar, cb]); M_all = np.concatenate([np.ones(N, bool), cm])
# front without an all-pairs dominance test: D is an integer, so take the 2-D front of (pB, pR) for each D first
valid = np.where((pen.max(1) < 1) & (Vv[:, 0] > 0))[0]
pre = []
for dval in np.unique(Vv[valid, 2]):
    i = valid[Vv[valid, 2] == dval]
    i = i[np.argsort(pen[i, 0] * 1e3 + pen[i, 1])]
    best = np.minimum.accumulate(pen[i, 1])
    k = np.ones(len(i), bool); k[1:] = pen[i[1:], 1] < best[:-1]
    pre.append(i[k])
pre = np.concatenate(pre)
nd = pre[non_dominated(pen[pre])]
rows = [dict(B=float(Vv[j, 0]), R=float(Vv[j, 1]), D=int(Vv[j, 2]), a=int(idx_ok[A_all[j]]), b=int(idx_ok[B_all[j]]),
             b_mirrored=bool(M_all[j])) for j in nd]
out = dict(note="x-incidence only (the library has no y-incidence); pairs from the passive RDN cells + mirrors; "
                "indices a, b are folder-order RDN indices (dataset.load_class)",
           n_mirror_pairs=int(N), n_general_candidates=int(len(ca)), front_size=len(rows), hv=hypervolume(pen[nd]),
           best_B_mirror=float(V_m[:, 0].max()), best_B_general=float(Vg[:, 0].max()),
           min_D_best_B=int(Vv[np.argmax(Vv[:, 0]), 2]), front=sorted(rows, key=lambda r: -r["B"]))
json.dump(out, open(os.path.join(OUT, "baseline_coding.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "baseline_coding_pairs.npz"), A=X[A_all[nd]],
         B=np.where(M_all[nd][:, None, None], mirror(X[B_all[nd]]), X[B_all[nd]]))
print(json.dumps({k: v for k, v in out.items() if k != "front"}, indent=1))
