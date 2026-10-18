"""Figure style for the manuscript, following the draft's figures (utils.py of the earlier pipeline):
Arial (sans serif, as in the IEEE graphics guide), IEEE column widths, the shaded-copper layout rendering. Every text element is 8 pt, one size in all figures (IEEE graphics guide:
type about 9-10 pt at printed size), and figures are saved at their printed size so that LaTeX does not rescale them.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COL1, COL2 = 3.5, 7.16                        # IEEE single / double column width (in)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, GREEN, GOLD, PURPLE, RED = "#3D74B8", "#E0704A", "#1E9E77", "#D99A0B", "#7A5BA8", "#e34948"
GREY = "#9a9a9a"
PATTERN_CMAP = ListedColormap(["#EDE8D6", "#CC8033"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "axes.titleweight": "normal", "axes.labelweight": "normal", "font.weight": "normal", "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.4, "axes.grid": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, path):
    """Save at the figure's own size (no tight bounding box, so the printed size is the figsize)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    if path.endswith(".pdf"):
        fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)


def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def layout_image(pattern, k=16, err=None):
    """Shaded copper sheet with etched edges and a soft shadow on the substrate (RGB, 16k x 16k)."""
    x = np.asarray(pattern) > 0.5
    n = x.shape[0]
    up = np.kron(x, np.ones((k, k), bool))
    N = n * k
    yy, xx = np.mgrid[0:N, 0:N] / (N - 1)
    rng = np.random.default_rng(0)
    sub = _hex("#EEE9D8") * (1 - 0.025 * rng.random((N, N, 1)))
    t = (0.65 * xx + 0.35 * yy)[..., None]
    cu = (1 - t) * _hex("#E9A15C") + t * _hex("#B4652A")
    cu = cu + 0.10 * np.exp(-((xx - yy + 0.15) / 0.18) ** 2)[..., None]
    img = np.where(up[..., None], cu, sub)
    s = max(1, k // 6)
    sh = np.zeros((N, N), bool); sh[s:, s:] = up[:-s, :-s]
    img = np.where((sh & ~up)[..., None], img * 0.82, img)
    pad = np.pad(up, 1)
    edge = up & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    img = np.where(edge[..., None], img * 0.72, img)
    if err is not None:
        e = np.kron(np.asarray(err, bool), np.ones((k, k), bool))
        img = np.where(e[..., None], 0.25 * img + 0.75 * _hex("#d62728"), img)
    return np.clip(img, 0, 1)


def show_layout(ax, pattern, frame=True):
    ax.imshow(layout_image(pattern), interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6 if frame else 0); s.set_edgecolor(INK2)


def tidy(ax):
    """Full frame: axis lines on all four sides, ticks inside on all four."""
    for s in ax.spines.values():
        s.set_visible(True)
    ax.tick_params(direction="in", pad=2, top=True, right=True)
