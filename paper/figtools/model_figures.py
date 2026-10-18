"""Figures of the trained models at their printed size, 8 pt text:
  fig_classes.pdf   one layout of each class (column width)
  fig_fwd.pdf       |r_xx| and |r_yx| over the 3,000 test layouts of each class, CST and the class network
  fig_vae_recon.pdf RDN test layouts, VAE reconstructions (d = 200, 64) and layouts decoded from random z

  python model_figures.py
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib.pyplot as plt
from paper_style import COL1, COL2, save, show_layout, layout_image, tidy, INK, INK2, BLUE, ORANGE
import config as C
from dataset import get_split_arrays
from phase1_forward_model import load_surrogate
from phase2_vae import load_vae
from train_phase1 import predict
import importlib, paper_style
importlib.reload(paper_style)          # our utils.py sets its own rcParams on import

FIG = HERE.parent / "figures_v2"
F = np.linspace(2, 12, 101)
CLASSES = ["PLG", "PLR", "PTN", "RDN"]
NET = {"PLG": "n27000", "PLR": "n27000", "PTN": "n27000", "RDN": "n108000"}
test = {c: get_split_arrays(c, "test", "both", "true", msdcnn_train=27000)[:2] for c in CLASSES}

# ------------------------------------------------------------------ one layout of each class
fig, axs = plt.subplots(1, 4, figsize=(COL1, 1.02))
fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.03, wspace=0.12)
fig.text(0.0, 0.80, "(b)", fontsize=8, va="bottom")
for ax, c, i in zip(axs, CLASSES, (5, 3, 7, 1)):
    show_layout(ax, test[c][0][i]); ax.set_title(c, fontsize=8, pad=2)
save(fig, str(FIG / "fig_classes.pdf"))

# ------------------------------------------------------------------ forward predictions of the class networks
fig, axs = plt.subplots(1, 4, figsize=(COL2, 1.7), sharey=True)
fig.subplots_adjust(left=0.065, right=0.99, top=0.87, bottom=0.245, wspace=0.08)
for ax, c, lab in zip(axs, CLASSES, "abcd"):
    X, y = test[c]
    Y = y[..., 0] + 1j * y[..., 1]
    net = load_surrogate(os.path.join(C.phase1_dir(c, "both", NET[c]), "phase1_forward_best.pth"))
    p = predict(net, X.astype(np.float32), tta=True, target="both"); P = p[..., 0] + 1j * p[..., 1]
    for ch, col in ((0, ORANGE), (1, BLUE)):
        ya, pa = np.abs(Y[:, ch]), np.abs(P[:, ch])
        ax.fill_between(F, ya.mean(0) - ya.std(0), ya.mean(0) + ya.std(0), color=col, alpha=0.18, lw=0)
        ax.plot(F, ya.mean(0), color=col, lw=1.3)
        ax.plot(F, pa.mean(0), color=INK, lw=0.9, ls="--")
    m = [np.mean((np.abs(P[:, k]) - np.abs(Y[:, k])) ** 2) * 1e3 for k in (0, 1)]
    ax.text(0.03, 0.47, f"MSE ($\\times10^{{-3}}$)\n$|r_{{xx}}|$: {m[0]:.2f}\n$|r_{{yx}}|$: {m[1]:.2f}", transform=ax.transAxes,
            fontsize=8, va="center", ha="left", linespacing=1.15)
    ax.set_title(f"({lab}) {c}{NET[c][1:]}", fontsize=8, pad=2, loc="left")
    ax.set(xlim=(2, 12), ylim=(0, 1.05), xticks=[4, 8, 12], yticks=[0, 0.5, 1])
    ax.set_xlabel("frequency (GHz)", labelpad=1)
    tidy(ax)
axs[0].set_ylabel("magnitude", labelpad=1)
axs[0].text(4.0, 0.93, "$|r_{xx}|$", color=ORANGE, fontsize=8, ha="left", va="top")
axs[0].text(4.0, 0.05, "$|r_{yx}|$", color=BLUE, fontsize=8, ha="left", va="bottom")
save(fig, str(FIG / "fig_fwd.pdf"))

# ------------------------------------------------------------------ VAE
Xr = torch.tensor(test["RDN"][0][:, None], dtype=torch.float32, device=C.DEVICE)
rows = [("test layouts", None, None)]
for d in (200, 64):
    vdir = C.vae_dir(f"rdn_d{d}")
    v = load_vae(os.path.join(vdir, "vae_best.pth"))
    acc = json.load(open(os.path.join(vdir, "vae_diagnostics.json")))["pixel_accuracy_mean"]
    with torch.no_grad():
        R = v.reconstruct(Xr[:6])[:, 0].float().cpu().numpy()
        if d == 200:
            g = torch.Generator(device=C.DEVICE).manual_seed(1)
            Z = v.decode(torch.randn(6, d, device=C.DEVICE, generator=g))[:, 0].float().cpu().numpy()
    rows.append((f"VAE, $d={d}$", R, acc))
rows.append(("random $\\mathbf{z}$", Z, None))
fig = plt.figure(figsize=(COL1, 1.44))
x0, w, gap = 0.29, 0.085, 0.01
for r, (lab, L, acc) in enumerate(rows):
    y0 = 1 - (r + 1) * 0.241 + 0.012
    fig.text(x0 - 0.02, y0 + 0.103, lab, ha="right", va="center", fontsize=8)
    for k in range(6):
        ax = fig.add_axes([x0 + k * (w + gap), y0, w, 0.207])
        T = test["RDN"][0][k]
        if L is None:
            ax.imshow(layout_image(T), interpolation="lanczos")
        elif lab.startswith("random"):
            ax.imshow(layout_image(L[k] > 0.5), interpolation="lanczos")
        else:
            ax.imshow(layout_image(L[k] > 0.5, err=(L[k] > 0.5) != (T > 0.5)), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_edgecolor(INK2)
    if acc is not None:
        fig.text(x0 + 6 * (w + gap) + 0.005, y0 + 0.103, f"{100 * acc:.1f} %", ha="left", va="center", fontsize=8)
save(fig, str(FIG / "fig_vae_recon.pdf"))
print("model figures written; VAE accuracies:", [round(r[2], 4) for r in rows if r[2] is not None])
