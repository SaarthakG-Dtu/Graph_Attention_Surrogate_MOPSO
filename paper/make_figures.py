"""
make_figures.py — regenerate the data figures of the manuscript into paper/figures/
(the schematics are TikZ: paper/schematics.py).

    python paper/make_figures.py            # all figures whose inputs exist
    python paper/make_figures.py forward

Each figure function reads only saved pipeline outputs (no training).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import config as C
import utils
from dataset import freq_grid

plt = utils.plt
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
F = freq_grid()


def _need(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


CLASS_NAMES = {"PLG": "PLG: polygons", "PLR": "PLR: polygon rings", "PTN": "PTN: shape combinations",
               "RDN": "RDN: random pixels"}


def fig_dataset(seed=4):
    """One test layout of each class (as MSDCNN Fig. 8(a)-(c))."""
    from dataset import get_split_arrays
    fig, axes = plt.subplots(1, 4, figsize=(utils.COL1, 1.12))
    rng = np.random.default_rng(seed)
    for ax, cls in zip(axes, C.CLASSES):
        x, _, _, _ = get_split_arrays(cls, "test")
        utils.show_layout(ax, x[rng.integers(len(x))])
        ax.set_title(cls, fontsize=9.5, pad=2, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.08)
    utils.save(fig, os.path.join(FIG, "fig_dataset.png"))

def fig_vae(n=6, seed=1):
    """RDN test layouts, their reconstructions by the d = 200 and d = 64 VAEs
    (wrongly decoded pixels in red) and layouts decoded from random z and from
    the central box of the latent space searched by MOIPSO."""
    import torch
    from matplotlib.colors import ListedColormap
    from dataset import get_split_arrays
    from phase2_vae import load_vae
    x, _, _, _ = get_split_arrays("RDN", "test")
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), n, replace=False)].astype(np.float32)
    xt = torch.as_tensor(x[:, None], device=C.DEVICE)
    good = load_vae(_need(os.path.join(C.vae_dir(C.MOPSO_VAE_TAG), "vae_best.pth")))
    bad = load_vae(_need(os.path.join(C.vae_dir("rdn"), "vae_best.pth")))
    rec_good = good.reconstruct(xt)[:, 0].cpu().numpy()
    rec_bad = bad.reconstruct(xt)[:, 0].cpu().numpy()
    g = torch.Generator(device=C.DEVICE).manual_seed(seed)
    d = good.latent_dim
    prior = good.decode(torch.randn(n, d, device=C.DEVICE, generator=g))[:, 0].cpu().numpy()
    box = good.decode(C.LATENT_BOUND * (2 * torch.rand(n, d, device=C.DEVICE, generator=g) - 1))[:, 0].cpu().numpy()
    rows = [("test layouts", x, None), (f"VAE, $d={d}$", rec_good, x), ("VAE, $d=64$", rec_bad, x),
            (r"$z\sim\mathcal{N}(0,I)$", prior, None), (rf"$|z_k|\leq{C.LATENT_BOUND:g}$", box, None)]
    fig, axes = plt.subplots(len(rows), n, figsize=(utils.COL1, 2.4))
    for r, (lab, imgs, ref) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            utils.show_layout(ax, imgs[c] > 0.5, err=None if ref is None else (imgs[c] > 0.5) != (ref[c] > 0.5))
        axes[r, 0].set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=4,
                              fontweight="bold")
        if ref is not None:
            acc = 100 * np.mean(imgs == ref)
            axes[r, -1].text(1.08, 0.5, f"{acc:.1f}%", transform=axes[r, -1].transAxes, fontsize=8.5,
                             va="center", fontweight="bold")
    fig.subplots_adjust(left=0.27, right=0.89, top=0.98, bottom=0.02, wspace=0.06, hspace=0.06)
    utils.save(fig, os.path.join(FIG, "fig_vae.png"))


def fig_forward_predictions():
    """As MSDCNN Fig. 9: mean and standard deviation of |r_xx| over each class's
    test set, dataset (CST) vs prediction, with the test MSE of the class."""
    from dataset import get_split_arrays
    from phase1_forward_model import load_surrogate
    from train_phase1 import predict
    fig, axes = plt.subplots(1, 4, figsize=(utils.COL2, 1.85), sharey=True)
    for ax, c in zip(axes, C.CLASSES):
        m = load_surrogate(_need(os.path.join(C.phase1_dir(c, "co", C.P1_REPORTED_TAG[c]), "phase1_forward_best.pth")))
        x, y, _, _ = get_split_arrays(c, "test", msdcnn_train=27000)       # MSDCNN test set: last 3,000 layouts
        mp = np.linalg.norm(predict(m, x, tta=True), axis=-1)
        mt = np.linalg.norm(y, axis=-1)
        for v, col, ls, lab in ((mt, utils.SERIES[1], "-", "CST (dataset)"), (mp, utils.SERIES[0], "--", "prediction")):
            mu, sd = v.mean(0), v.std(0)
            ax.fill_between(F, mu - sd, np.minimum(mu + sd, 1.0), color=col, alpha=0.18, lw=0)
            ax.plot(F, mu, color=col, ls=ls, lw=1.2, label=lab)
        mse = np.mean((mp - mt) ** 2)
        ax.text(0.04, 0.06, f"MSE = {1e3 * mse:.2f}" + r"$\times10^{-3}$", transform=ax.transAxes, fontsize=8)
        ax.set_title(c, fontsize=9); ax.set_xlim(2, 12); ax.set_ylim(0.5, 1.02)
        ax.set_xlabel("Frequency (GHz)", fontsize=8.5)
    axes[0].set_ylabel(r"$|r_{xx}|$", fontsize=9)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.14), fontsize=7.5, frameon=False)
    fig.tight_layout(w_pad=0.6)
    utils.save(fig, os.path.join(FIG, "fig_forward_predictions.png"))

def _front_history(F, gen, step):
    """Hypervolume and front of all designs evaluated up to each generation."""
    from phase2_mopso import non_dominated, hypervolume
    gens = np.arange(0, gen.max() + 1)
    front, hv, snaps = np.zeros((0, F.shape[1])), [], {}
    for g in gens:
        new = F[gen == g]
        cand = np.vstack([front, new[non_dominated(new)]]) if len(new) else front
        front = cand[non_dominated(cand)]
        if g % step == 0 or g == gens[-1]:
            hv.append((g, hypervolume(front, n=40_000)))
            snaps[int(g)] = front.copy()
    return np.array(hv), snaps


def _penalty_to_values(P):
    """Inverse of the normalisation of the amc_x objectives (B, L, D)."""
    ideal = np.array([s["ideal"] for s in C.OBJECTIVE_PRESETS["amc_x"]])
    worst = np.array([s["worst"] for s in C.OBJECTIVE_PRESETS["amc_x"]])
    V = ideal + np.minimum(P, 1.0) * (worst - ideal)
    V[P[:, 0] >= 1.0, 0] = 0.0
    return V


RUN_LONG = "sc_latent_N1000_G300_s0"     # final runs: VAE and pixel space, 1000 x 300, seed 0
RUN_PIX = "sc_binary_N1000_G300_s0"


def _cst_values(problem, tag):
    import csv
    f = os.path.join(C.mopso_dir(problem, tag), "CST_verification", "cst_vs_surrogate.csv")
    if not os.path.exists(f):
        return np.zeros((0, 3))
    v = [[float(r["In-phase bandwidth (%) CST"]), float(r["Lowest |r| in band (dB) CST"]),
          float(r["Fragile features CST"])] for r in csv.DictReader(open(f))]
    v = np.array(v)
    return v[v[:, 0] > 0]


def fig_pareto(problem=C.MOPSO_PROBLEM):
    """(a) 3-D Pareto fronts of the VAE-space and pixel-space searches and of the
    dataset, with the CST results of the verified designs, (b) the same fronts as
    in-phase bandwidth vs lowest in-band reflection, (c) front of the VAE search
    at several generations (in-phase bandwidth vs fragile features)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    db = np.load(_need(os.path.join(C.mopso_dir(problem, RUN_LONG), "database_front.npz")))
    fv = np.load(_need(os.path.join(C.mopso_dir(problem, RUN_LONG), "Final", "pareto_designs.npz")))["values"]
    fp = np.load(_need(os.path.join(C.mopso_dir(problem, RUN_PIX), "Final", "pareto_designs.npz")))["values"]
    cv, cp = _cst_values(problem, RUN_LONG), _cst_values(problem, RUN_PIX)
    L_MIN = -20.0                                                       # L axis floor; deeper dips drawn at the floor
    sets = [(db["V"], "s", utils.SERIES[1], "dataset (CST)", 9), (fp, "^", utils.SERIES[2], "pixel search", 9),
            (fv, "o", utils.SERIES[0], "VAE search", 9)]
    stars = [(cp, "P", "pixel designs in CST"), (cv, "*", "VAE designs in CST")]
    fig = plt.figure(figsize=(utils.COL2, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0], projection="3d")
    for V, mk, col, lab, s in sets:
        V = V[V[:, 0] > 0]
        ax.scatter(V[:, 0], np.maximum(V[:, 1], L_MIN), V[:, 2], s=s, marker=mk, color=col, depthshade=False, label=lab)
    for V, mk, lab in stars:
        if len(V):
            ax.scatter(V[:, 0], np.maximum(V[:, 1], L_MIN), V[:, 2], s=60 if mk == "*" else 30, marker=mk,
                       color="#ffd400", edgecolor="black", lw=0.6, depthshade=False, label=lab)
    ax.set_ylim(L_MIN, 0)
    ax.set_xticks([5, 10, 15, 20]); ax.set_yticks([-20, -10, 0]); ax.set_zticks([0, 20, 40, 60])
    ax.set_xlabel("B (%)", fontsize=8, labelpad=-6); ax.set_ylabel("L (dB)", fontsize=8, labelpad=-6)
    ax.set_zlabel("D", fontsize=8, labelpad=-9)
    ax.tick_params(labelsize=7, pad=-3)
    ax.view_init(elev=20, azim=-50)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=2, columnspacing=0.6, handletextpad=0.2,
              bbox_to_anchor=(-0.05, 0.97))
    ax.text2D(-0.08, 1.22, "(a)", transform=ax.transAxes, fontsize=9.5, fontweight="bold")

    # (b) in-phase bandwidth vs lowest in-band reflection
    ax = fig.add_subplot(gs[1])
    for V, mk, col, lab, s in sets:
        V = V[V[:, 0] > 0]
        ax.scatter(V[:, 0], np.maximum(V[:, 1], L_MIN), s=s, marker=mk, color=col, lw=0, label=lab)
    for V, mk, lab in stars:
        if len(V):
            ax.scatter(V[:, 0], V[:, 1], s=60 if mk == "*" else 30, marker=mk, color="#ffd400", edgecolor="black",
                       lw=0.6, zorder=5)
    ax.axvline(10, color="0.55", ls="--", lw=0.6); ax.axhline(-1, color="0.55", ls="--", lw=0.6)
    ax.set_xlim(0, 22); ax.set_ylim(L_MIN, 0.5)
    ax.set_xlabel("B (%)", fontsize=8.5); ax.set_ylabel("L (dB)", fontsize=8.5)
    ax.text(-0.28, 1.02, "(b)", transform=ax.transAxes, fontsize=9.5, fontweight="bold")

    # (c) front of the VAE search after several generations
    ax = fig.add_subplot(gs[2])
    h = np.load(_need(os.path.join(C.mopso_dir(problem, RUN_LONG), "swarm_history.npz")))
    _, snaps = _front_history(h["objectives"], h["generation"], 10)
    cmap = plt.get_cmap("viridis")
    gsel = [g for g in (0, 10, 50, 300) if g in snaps]
    for k, g in enumerate(gsel):
        V = _penalty_to_values(snaps[g]); V = V[V[:, 0] > 0]
        ax.scatter(V[:, 0], V[:, 2], s=7, color=cmap(k / max(len(gsel) - 1, 1)), label=f"gen. {g}")
    Vd = db["V"][db["V"][:, 0] > 0]
    ax.scatter(Vd[:, 0], Vd[:, 2], s=9, marker="s", facecolor="none", edgecolor=utils.SERIES[1], lw=0.6, label="dataset")
    ax.axvline(10, color="0.55", ls="--", lw=0.6); ax.axhline(5, color="0.55", ls="--", lw=0.6)
    ax.text(21.6, 2.5, "target", fontsize=7, color="0.4", ha="right", va="center")
    ax.set_xlim(0, 22)
    ax.set_xlabel("B (%)", fontsize=8.5); ax.set_ylabel("D (fragile features)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.text(-0.28, 1.02, "(c)", transform=ax.transAxes, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.18, top=0.88)
    utils.save(fig, os.path.join(FIG, "fig_pareto.png"))


def fig_cst_verification(problem=C.MOPSO_PROBLEM,
                         groups=((RUN_LONG, "", 0, 3), (RUN_LONG, "", 3, 6), (RUN_PIX, "P", 0, 3))):
    """All CST-verified designs, one column per group: designs 1-3 and 4-6 of the
    long VAE search and P1-P3 of the pixel-space search. Layout, |r_xx| and
    surface phase, surrogate (dashed) vs CST (solid)."""
    from phase2_mopso import surface_phase_deg, inphase_band
    f0 = C.OBJECTIVE_PRESETS[problem][0]["f0"]
    cols = []
    for tag, prefix, a, b in groups:
        run = C.mopso_dir(problem, tag)
        d = np.load(_need(os.path.join(run, "Final", "pareto_designs.npz")))
        order = list(np.argsort(d["objectives"].sum(1)))               # verification order (cst_verify --top)
        items = []
        for j, k in list(enumerate(order))[a:b]:
            p = os.path.join(run, "CST_verification", "runs", f"solution_{k + 1:03d}_t{C.CST_SUB_T_MM:g}.npz")
            if os.path.exists(p):
                items.append((f"{prefix}{j + 1}", d["patterns"][k], d["R"][k], p, d["freq_ghz"]))
        cols.append(items)
    rows = max(len(c) for c in cols)
    if rows == 0:
        raise FileNotFoundError("no CST verification runs")
    fig = plt.figure(figsize=(utils.COL2, 1.4 * rows + 0.35))
    outer = fig.add_gridspec(rows, len(cols), wspace=0.16, hspace=0.5)
    first = True
    for c, items in enumerate(cols):
        for r, (name, pat, Rp, p, f) in enumerate(items):
            s = np.load(p)
            Rc = np.interp(f, s["freq"], s["s11"].real) + 1j * np.interp(f, s["freq"], s["s11"].imag)
            inner = outer[r, c].subgridspec(2, 2, width_ratios=[0.62, 2.0], wspace=0.62, hspace=0.14)
            ax = fig.add_subplot(inner[:, 0])
            utils.show_layout(ax, pat)
            ax.set_title(f"design {name}", fontsize=8.5, fontweight="bold", pad=3)
            a_mag, a_ph = fig.add_subplot(inner[0, 1]), fig.add_subplot(inner[1, 1])
            for ax, yc, yp in ((a_mag, 20 * np.log10(np.abs(Rc)), 20 * np.log10(np.abs(Rp))),
                               (a_ph, surface_phase_deg(Rc[None], f)[0], surface_phase_deg(Rp[None], f)[0])):
                for R_, col in ((Rc, utils.SERIES[1]), (Rp, utils.SERIES[0])):
                    lo, hi, _, _ = inphase_band(R_[None], f, f0)
                    if lo[0] >= 0:
                        ax.axvspan(f[lo[0]], f[hi[0]], color=col, alpha=0.13, lw=0)
                ax.plot(f, yc, color=utils.SERIES[1], lw=1.0, label="CST")
                ax.plot(f, yp, color=utils.SERIES[0], lw=0.9, ls="--", label="surrogate")
                ax.set_xlim(2, 12); ax.set_xticks([2, 4, 6, 8, 10, 12]); ax.tick_params(labelsize=7, pad=1.5)
            a_mag.set_ylim(-9, 0.8); a_mag.set_yticks([-8, -4, 0]); a_mag.set_xticklabels([])
            a_mag.set_ylabel("dB", fontsize=8, labelpad=1)
            a_ph.set_ylim(-185, 185); a_ph.set_yticks([-180, 0, 180])
            a_ph.set_ylabel("deg", fontsize=8, labelpad=1)
            for y in (-90, 90):
                a_ph.axhline(y, color=utils.INK2, lw=0.45, ls=":")
            if r == len(items) - 1:
                a_ph.set_xlabel("Frequency (GHz)", fontsize=8.5, labelpad=1)
            if first:
                a_mag.legend(fontsize=7, loc="lower left", handlelength=1.6, borderaxespad=0.2)
                first = False
    fig.subplots_adjust(left=0.01, right=0.995, top=0.965, bottom=0.07)
    utils.save(fig, os.path.join(FIG, "fig_cst_verification.png"))


# Schematics (unit cell, architecture, pipeline) are TikZ: paper/schematics.py.
FIGS = dict(dataset=fig_dataset, vae=fig_vae, forward=fig_forward_predictions, pareto=fig_pareto, cst=fig_cst_verification)

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        try:
            FIGS[n]()
            print("ok  ", n)
        except FileNotFoundError as err:
            print("skip", n, "-", err)
