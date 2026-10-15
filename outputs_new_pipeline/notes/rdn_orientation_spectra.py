"""Internal note (not for the paper): spectra of RDN layouts as stored in the SUTD
file vs flipped (1 - x).

1) rdn_orientation_cst.png  - 3 layouts simulated in CST in both orientations
   (t = 0.8 mm) against the r_xx stored in the dataset.
2) rdn_orientation_surrogate.png - the 10 layouts of rdn_orientation_10_designs.png:
   dataset r_xx against the RDN108000 prediction for each orientation drawn as copper.
Run from the repo root: python outputs_new_pipeline/notes/rdn_orientation_spectra.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.getcwd())
from dataset import load_class, freq_grid                    # noqa: E402
from phase2_surrogate import Surrogate                       # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join("outputs_new_pipeline", "CST", "calibration", "runs")
CMAP = ListedColormap(["#EDE8D6", "#CC8033"])
C_DATA, C_STORED, C_FLIP = "black", "#d62728", "#2a78d6"
f = freq_grid()
im, cu = load_class("RDN")                                    # as stored in the SUTD files
R_data = cu[:, 0]                                             # co-pol r_xx, 101 points


def show(ax, m, title=None):
    ax.imshow(m, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True); ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.25); ax.tick_params(which="both", length=0)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold")


def mse(a, b):
    return np.mean((np.abs(a) - np.abs(b)) ** 2)


def phase_err(a, b):
    return np.degrees(np.abs(np.angle(a * np.conj(b)))).mean()


# ── 1) CST, both orientations ──
ids = [10530, 22021, 101796]
fig, axes = plt.subplots(len(ids), 4, figsize=(13, 2.6 * len(ids)),
                         gridspec_kw=dict(width_ratios=[0.55, 0.55, 1.6, 1.6], wspace=0.28, hspace=0.45))
for r, k in enumerate(ids):
    cst = {}
    for tag, suf in (("stored", ""), ("flipped", "_invert")):
        d = np.load(os.path.join(CAL, f"RDN_{k}_t0.8{suf}.npz"))
        cst[tag] = np.interp(f, d["freq"], d["s11"].real) + 1j * np.interp(f, d["freq"], d["s11"].imag)
    show(axes[r, 0], im[k], f"#{k} as stored")
    show(axes[r, 1], 1 - im[k], f"#{k} flipped")
    ax = axes[r, 2]
    ax.plot(f, np.abs(R_data[k]), color=C_DATA, lw=1.6, label="dataset (SUTD)")
    ax.plot(f, np.abs(cst["flipped"]), color=C_FLIP, lw=1.2, ls="--",
            label=f"CST, flipped  (MSE {mse(cst['flipped'], R_data[k]):.1e})")
    ax.plot(f, np.abs(cst["stored"]), color=C_STORED, lw=1.2, ls=":",
            label=f"CST, as stored  (MSE {mse(cst['stored'], R_data[k]):.1e})")
    ax.set_ylabel("|r_xx|", fontsize=8); ax.set_xlim(2, 12); ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax = axes[r, 3]
    ax.plot(f, np.degrees(np.angle(R_data[k])), color=C_DATA, lw=1.6, label="dataset")
    ax.plot(f, np.degrees(np.angle(cst["flipped"])), color=C_FLIP, lw=1.2, ls="--",
            label=f"CST, flipped  ({phase_err(cst['flipped'], R_data[k]):.1f}° mean diff.)")
    ax.plot(f, np.degrees(np.angle(cst["stored"])), color=C_STORED, lw=1.2, ls=":",
            label=f"CST, as stored  ({phase_err(cst['stored'], R_data[k]):.1f}° mean diff.)")
    ax.set_ylabel("phase of r_xx (deg)", fontsize=8); ax.set_xlim(2, 12); ax.set_ylim(-180, 180)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    for ax in axes[r, 2:]:
        ax.tick_params(labelsize=7)
        if r == len(ids) - 1:
            ax.set_xlabel("frequency (GHz)", fontsize=8)
fig.suptitle("RDN layouts simulated in CST (t = 0.8 mm) in both orientations vs the r_xx stored in the dataset: "
             "only the flipped layout reproduces the dataset", fontsize=10, fontweight="bold")
fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.07)
fig.savefig(os.path.join(OUT, "rdn_orientation_cst.png"), dpi=180)
plt.close(fig)

# ── 2) surrogate, the 10 layouts of rdn_orientation_10_designs.png ──
idx = np.sort(np.random.default_rng(7).choice(len(im), 10, replace=False))
sur = Surrogate("RDN:n108000")                                 # trained on flipped (1 = copper) layouts
p_flip = sur.predict((1 - im[idx]).astype(np.float32))
p_stored = sur.predict(im[idx].astype(np.float32))
fig, axes = plt.subplots(5, 6, figsize=(15, 11),
                         gridspec_kw=dict(width_ratios=[0.42, 0.42, 1.5] * 2, wspace=0.3, hspace=0.5))
for n, k in enumerate(idx):
    r, c = n % 5, 3 * (n // 5)
    show(axes[r, c], im[k], f"#{k} stored")
    show(axes[r, c + 1], 1 - im[k], "flipped")
    ax = axes[r, c + 2]
    ax.plot(f, np.abs(R_data[k]), color=C_DATA, lw=1.6, label="dataset (SUTD)")
    ax.plot(f, np.abs(p_flip[n]), color=C_FLIP, lw=1.2, ls="--",
            label=f"flipped as copper (MSE {mse(p_flip[n], R_data[k]):.1e})")
    ax.plot(f, np.abs(p_stored[n]), color=C_STORED, lw=1.2, ls=":",
            label=f"stored as copper (MSE {mse(p_stored[n], R_data[k]):.1e})")
    ax.set_xlim(2, 12); ax.tick_params(labelsize=7); ax.set_ylabel("|r_xx|", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    if r == 4:
        ax.set_xlabel("frequency (GHz)", fontsize=8)
fig.suptitle("10 RDN layouts: dataset |r_xx| vs the RDN108000 surrogate's prediction for each orientation drawn as "
             "copper\n(surrogate = learned model; the CST check of 3 layouts is in rdn_orientation_cst.png)",
             fontsize=10, fontweight="bold")
fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.05)
fig.savefig(os.path.join(OUT, "rdn_orientation_surrogate.png"), dpi=170)
plt.close(fig)
print("saved to", OUT)
