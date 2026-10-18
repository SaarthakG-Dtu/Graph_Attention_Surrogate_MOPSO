"""
utils.py — shared plotting (journal style) for Phase 1, Phase 2 and CST.

Series colours follow one fixed categorical order everywhere:
    reference / CST / ground truth -> SERIES[1] (orange)
    surrogate prediction           -> SERIES[0] (blue)
    third series                   -> SERIES[2] (aqua)
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import kurtosis

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
PATTERN_CMAP = ListedColormap(["#EDE8D6", "#CC8033"])   # substrate, copper (as in the schematics)
COL1, COL2 = 3.5, 7.16          # IEEE single / double column width (in)


def paper_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 9.5,
        "axes.titlesize": 9.5, "axes.labelweight": "bold", "axes.titleweight": "bold", "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.4, "axes.grid": False,
        "lines.linewidth": 1.2, "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


paper_style()


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)


def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def layout_image(pattern, k=16, err=None):
    """Rendered copper layout (RGB, 16k x 16k): a shaded copper sheet with etched
    edges and a soft shadow on the substrate. err: pixels drawn in red."""
    x = np.asarray(pattern) > 0.5
    n = x.shape[0]
    up = np.kron(x, np.ones((k, k), bool))
    N = n * k
    yy, xx = np.mgrid[0:N, 0:N] / (N - 1)
    rng = np.random.default_rng(0)
    sub = _hex("#EEE9D8") * (1 - 0.025 * rng.random((N, N, 1)))
    t = (0.65 * xx + 0.35 * yy)[..., None]                       # light from the top left
    cu = (1 - t) * _hex("#E9A15C") + t * _hex("#B4652A")
    cu = cu + 0.10 * np.exp(-((xx - yy + 0.15) / 0.18) ** 2)[..., None]   # soft specular band
    img = np.where(up[..., None], cu, sub)
    # soft shadow of the copper on the substrate (offset to the bottom right)
    s = max(1, k // 6)
    sh = np.zeros((N, N), bool); sh[s:, s:] = up[:-s, :-s]
    img = np.where((sh & ~up)[..., None], img * 0.82, img)
    # etched edges: darken copper pixels that border the substrate
    pad = np.pad(up, 1)
    edge = up & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    img = np.where(edge[..., None], img * 0.72, img)
    if err is not None:
        e = np.kron(np.asarray(err, bool), np.ones((k, k), bool))
        img = np.where(e[..., None], 0.25 * img + 0.75 * _hex("#d62728"), img)
    return np.clip(img, 0, 1)


def show_layout(ax, pattern, err=None, frame=True):
    ax.imshow(layout_image(pattern, err=err), interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6 if frame else 0); s.set_edgecolor(INK2)


def inset_pattern(ax, pattern, bounds=(0.02, 0.04, 0.26, 0.42)):
    axin = ax.inset_axes(bounds)
    axin.imshow(layout_image(pattern), interpolation="lanczos")
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_linewidth(0.5); s.set_edgecolor(INK2)
    return axin


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(tr_h, va_h, path):
    keys = [("total", "Total loss"), ("lin_mse", "lin-MSE |S|"), ("db_mse", "dB-MSE"),
            ("reim", "Re/Im term"), ("phase", "Phase term"), ("kk", "KK term")]
    ep = np.arange(1, len(tr_h) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 3.6))
    for ax, (k, t) in zip(axes.flat, keys):
        k = k if k in tr_h[0] else f"{k}_xx"                          # two-output runs: r_xx terms
        ax.plot(ep, [h[k] for h in tr_h], color=SERIES[0], label="train")
        ax.plot(ep, [h[k] for h in va_h], color=SERIES[1], label="validation")
        ax.set_yscale("log"); ax.set_title(t); ax.set_xlabel("Epoch")
        ax.grid(True, which="major")
    axes.flat[0].legend()
    fig.tight_layout()
    save(fig, path)


def spectrum_grid(freq, panels, path, ncols=4, ylabel=r"$|S_{11}|$", legend=("target", "predicted")):
    """panels: list of dict(pattern, curves=[c_ref, c_pred, ...], text)."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL2, 1.25 * nrows + 0.3),
                             sharex=True, sharey=True, squeeze=False)
    colors = [SERIES[1], SERIES[0], SERIES[2]]
    styles = ["-", "--", ":"]
    for ax, pn in zip(axes.flat, panels):
        for c, col, ls, lab in zip(pn["curves"], colors, styles, legend):
            ax.plot(freq, c, color=col, ls=ls, lw=1.1, label=lab)
        ax.set_ylim(-0.02, 1.05); ax.set_xlim(freq[0], freq[-1])
        ax.set_yticks([0, 0.5, 1.0])
        inset_pattern(ax, pn["pattern"])
        if pn.get("text"):
            ax.text(0.97, 0.05, pn["text"], transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6, color=INK2)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(legend))
    fig.supxlabel("Frequency (GHz)", fontsize=8); fig.supylabel(ylabel, fontsize=8)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save(fig, path)


def plot_eval_figures(freq, x, y, p, lin, out_dir, n_show=12, title=""):
    mp, mt = np.linalg.norm(p, axis=-1), np.linalg.norm(y, axis=-1)
    order = np.argsort(lin)
    pick = {"best": order[:n_show],
            "median": order[len(order) // 2 - n_show // 2: len(order) // 2 + (n_show + 1) // 2],
            "worst": order[::-1][:n_show],
            "uniform": order[np.linspace(0, len(order) - 1, n_show).astype(int)]}
    for name, ids in pick.items():
        panels = [dict(pattern=x[i], curves=[mt[i], mp[i]], text=f"MSE {lin[i]:.1e}") for i in ids]
        spectrum_grid(freq, panels, os.path.join(out_dir, f"eval_samples_{name}.png"))

    # phase panels for the uniform pick
    ids = pick["uniform"][:8]
    ph_t = np.degrees(np.angle(y[..., 0] + 1j * y[..., 1]))
    ph_p = np.degrees(np.angle(p[..., 0] + 1j * p[..., 1]))
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 2.8), sharex=True, sharey=True)
    for ax, i in zip(axes.flat, ids):
        ax.plot(freq, ph_t[i], color=SERIES[1], lw=1.0, label="target")
        ax.plot(freq, ph_p[i], color=SERIES[0], lw=1.0, ls="--", label="predicted")
        ax.set_ylim(-185, 185); ax.set_yticks([-180, 0, 180])
    axes.flat[0].legend(loc="lower left")
    fig.supxlabel("Frequency (GHz)", fontsize=8); fig.supylabel(r"$\angle S_{11}$ (deg)", fontsize=8)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "eval_phase.png"))

    # mean +- std and kurtosis of 1-|S| (MSDCNN Figs. 9/12), error vs frequency, scatter
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.9))
    a_t, a_p = 1 - np.clip(mt, 0, 1), 1 - np.clip(mp, 0, 1)
    ax = axes[0]
    for arr, col, lab in [(a_t, SERIES[1], "target"), (a_p, SERIES[0], "predicted")]:
        m, s = arr.mean(0), arr.std(0)
        ax.plot(freq, m, color=col, label=lab); ax.fill_between(freq, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.set_title(r"mean $\pm$ std of $1-|S|$"); ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(freq, kurtosis(a_t, axis=0), color=SERIES[1]); ax.plot(freq, kurtosis(a_p, axis=0), color=SERIES[0], ls="--")
    ax.set_title("kurtosis of $1-|S|$")
    ax = axes[2]
    ax.plot(freq, ((mp - mt) ** 2).mean(0), color=SERIES[0]); ax.set_yscale("log")
    ax.set_title("lin-MSE vs frequency")
    ax = axes[3]
    sub = np.random.default_rng(0).choice(mt.size, min(20000, mt.size), replace=False)
    ax.scatter(mt.ravel()[sub], mp.ravel()[sub], s=1, color=SERIES[0], alpha=0.25, lw=0, rasterized=True)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_aspect("equal"); ax.set_title("predicted vs target $|S|$")
    for ax in axes[:3]:
        ax.set_xlabel("Frequency (GHz)")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "eval_statistics.png"))


def plot_heatmap(mat, rows, cols, title, path, fmt="{:.2e}", cmap="Blues"):
    fig, ax = plt.subplots(figsize=(COL1, 0.5 * len(rows) + 1.0))
    ax.imshow(np.log10(mat) if fmt.endswith("e}") else mat, cmap=cmap)
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("test class"); ax.set_ylabel("trained on")
    vals = np.log10(mat) if fmt.endswith("e}") else mat
    mid = np.nanmean(vals)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if vals[i, j] > mid else INK)
    ax.set_title(title)
    save(fig, path)
