"""Result figures at their printed size (IEEE columns), 8 pt sans-serif text.

  python data_figures.py      -> ../figures_v2/fig_fronts.pdf, fig_designs.pdf
Designs of the four-output RDN108000 (searches outputs_new_pipeline/PHASE2/mopso/{cp,rotator,coding}_*), checked with
FDTD (experiments/devices/verify_*: all copper lossy, extrapolated to zero mesh size) and CST; tolerances from
experiments/devices/mc; checkerboards from experiments/devices/verify_coding/board_d0_N*_both_cu.npz.
"""
import glob
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from paper_style import COL2, save, layout_image, show_layout, tidy, INK, INK2, GREY, BLUE, ORANGE, GREEN, RED
import config as C
from dataset import load_class, to_convention, freq_grid
from phase2_devices import DeviceObjectives, CodingObjectives, axial_ratio_db
from phase2_mopso import non_dominated
import importlib, paper_style
importlib.reload(paper_style)          # our utils.py sets its own rcParams on import

FIG = HERE.parent / "figures_v2"
DEV = ROOT / "experiments" / "devices"
F = freq_grid()
APPS = {"cp": dict(dev="cp", f0=10.0), "rot": dict(dev="rotator", f0=10.2)}

# ------------------------------------------------------------------ data for the fronts
im, cu = load_class("RDN")
Xd = to_convention(im, "RDN", "true", np.arange(len(im))).astype(np.uint8)
Yd = cu[:, :2]
passive = (np.abs(Yd) ** 2).sum(1).max(1) <= 1.01
res = {}
for app, a in APPS.items():
    ob = DeviceObjectives(a["dev"], F, search=False)
    Pd, Vd = ob(Yd[passive], Xd[passive])
    ok = np.where(Vd[:, 0] > 0)[0]
    dfront = Vd[ok][non_dominated(Pd[ok])][:, :3]
    Vs, Ps = [], []
    for d in sorted(glob.glob(os.path.join(C.OUTPUT_ROOT, "PHASE2", "mopso", f"{a['dev']}_*"))):
        f = os.path.join(d, "Final", "pareto_designs.npz")
        if os.path.exists(f) and "stepcheck" not in d:
            z = np.load(f); P_, V_ = ob(z["R"], z["patterns"]); Vs.append(V_); Ps.append(P_)
    Vs, Ps = np.concatenate(Vs), np.concatenate(Ps)
    i = np.where(Vs[:, 0] > 0)[0]
    sfront = Vs[i][non_dominated(Ps[i])][:, :3]
    z = np.load(DEV / f"verify_{a['dev']}" / "spectra.npz")
    X = z["X"].astype(np.uint8); S0 = z["extrap"]; pred = z["pred"]
    Scst = np.load(DEV / f"verify_{a['dev']}" / "cst_spectra.npy")
    Vf, _ = ob.values(S0, X); Vp, _ = ob.values(pred, X)
    meets = ob.good(Vf)
    best = int(np.argmax(np.where(meets, Vf[:, 0], -1)))           # widest FDTD band among designs meeting the target
    if app == "rot":                                                 # the rotator without corner contacts or isolated pixels
        best = int(np.argmax(np.where(meets & (Vf[:, 2] == 0), Vf[:, 0], -1)))
    _, f_lo, f_hi = ob.band(S0)[:3]
    res[app] = dict(dfront=dfront, sfront=sfront, f=Vf[:, :3], p=Vp[:, :3], pred=pred, S0=S0, Scst=Scst, X=X, best=best,
                    f_lo=f_lo, f_hi=f_hi)
    print(app, "best design", best + 1, "B", round(float(Vf[best, 0]), 2), "meets", int(meets.sum()), "of", len(meets))

# ------------------------------------------------------------------ Fig: Pareto fronts (text width, one row)
fig, axs = plt.subplots(1, 4, figsize=(COL2, 2.1), layout="constrained")
spec = [("cp", 2, "(a) LP-to-CP: $B$ vs. $D$", "3-dB AR band $B$ (%)", "$D$", 10, None),
        ("cp", 1, "(b) LP-to-CP: $B$ vs. AR", "3-dB AR band $B$ (%)", "mean AR in band (dB)", 10, None),
        ("rot", 2, "(c) rotator: $B$ vs. $D$", "PCR > 0.9 band $B$ (%)", "$D$", 5, None),
        ("rot", 1, "(d) rotator: $B$ vs. $|r_{yx}|^2$", "PCR > 0.9 band $B$ (%)", "mean $|r_{yx}|^2$ in band", 5, 0.8)]
for ax, (app, q, title, xl, yl, btgt, qtgt) in zip(axs, spec):
    r = res[app]
    if q == 2:
        ax.fill_between([btgt, 30], -2, 5.5, color=GREEN, alpha=0.13, lw=0)
    else:
        ax.axvline(btgt, color=GREEN, ls="--", lw=0.8)
        if qtgt:
            ax.axhline(qtgt, color=GREEN, ls="--", lw=0.8)
    ax.scatter(r["dfront"][:, 0], r["dfront"][:, q], s=10, marker="s", c=GREY, edgecolors="none", label="dataset (CST)")
    ax.scatter(r["sfront"][:, 0], r["sfront"][:, q], s=9, c=BLUE, edgecolors="none", label="search (surrogate)")
    for p_, f_ in zip(r["p"], r["f"]):
        ax.plot([p_[0], f_[0]], [p_[q], f_[q]], color=RED, lw=0.5, alpha=0.6)
    ax.scatter(r["f"][:, 0], r["f"][:, q], s=28, marker="*", c=RED, edgecolors="none", label="selected (FDTD)")
    xmax = max(r["dfront"][:, 0].max(), r["sfront"][:, 0].max(), r["f"][:, 0].max()) * 1.06
    ax.set_xlim(0, xmax)
    if q == 2:
        ax.set_ylim(-2, 50)
    ax.set_xlabel(xl); ax.set_ylabel(yl, labelpad=1); ax.set_title(title, loc="left", fontsize=8)
    tidy(ax)
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=8, handletextpad=0.3, columnspacing=1.5)
save(fig, str(FIG / "fig_fronts.pdf"))

# ------------------------------------------------------------------ Fig: best designs (text width, 3 columns)
CS = DEV / "verify_coding"
MI = int(sys.argv[1]) if len(sys.argv) > 1 else 0
zc = np.load(CS / "verify_spectra.npz")
mcf = DEV / "mc" / "mc.npz"
mc = np.load(mcf) if mcf.exists() else None
mc_app = {"cp": mc, "rot": np.load(DEV / "mc_rot2" / "mc.npz") if (DEV / "mc_rot2" / "mc.npz").exists() else None}
MC_K = {"cp": 0, "rot": 1}
pair = np.load(CS / "shortlist.npz")["X"][MI]
red = lambda s: -10 * np.log10(np.clip(s, 1e-3, None))
coding = CodingObjectives("free", F, search=False)
Yall = cu[:, :2]                                                         # library-pair baseline (folder-order indices)
base = json.loads((DEV / "baseline_coding.json").read_text())["front"][0]
A_, B_ = Yall[base["a"]].copy(), Yall[base["b"]].copy()
if base["b_mirrored"]:
    B_[1] *= -1
s_lib = (np.abs(A_ + B_) ** 2).sum(0) / 4
boards = {N: np.load(CS / f"board_d{MI}_N{N}_both_cu.npz")["s"].max(0) for N in (3, 4, 6, 8)
          if (CS / f"board_d{MI}_N{N}_both_cu.npz").exists()}
print("coding pair", MI, "boards", sorted(boards))

fig = plt.figure(figsize=(COL2, 3.45), layout="constrained")
gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.0, 1.0])
for j, app in enumerate(("cp", "rot")):
    r, a = res[app], APPS[app]
    i = r["best"]
    ax = fig.add_subplot(gs[0, j]); ax.axis("off")
    sub = ax.inset_axes([0.0, 0.0, 0.36, 1.0]); show_layout(sub, r["X"][i])
    q = (f"AR = {r['f'][i, 1]:.2f} dB" if app == "cp" else f"$|r_{{yx}}|^2$ = {r['f'][i, 1]:.2f}")
    txt = (("(a) LP-to-CP, 10 GHz" if app == "cp" else "(b) rotator, 10.2 GHz") +
           f"\nband {r['f_lo'][i]:.2f}–{r['f_hi'][i]:.2f} GHz\n$B$ = {r['f'][i, 0]:.1f} %, {q}\n$D$ = {int(r['f'][i, 2])}")
    ax.text(0.40, 0.5, txt, transform=ax.transAxes, va="center", ha="left", fontsize=8, linespacing=1.3)
    ax = fig.add_subplot(gs[1, j])
    for k, col in ((0, ORANGE), (1, BLUE)):
        ax.plot(F, np.abs(r["S0"][i, k]), color=col, lw=1.2)
        ax.plot(F, np.abs(r["Scst"][i, k]), color=col, lw=1.0, ls=":")
        ax.plot(F, np.abs(r["pred"][i, k]), color=col, lw=1.0, ls="--")
    ax.axvline(a["f0"], color=INK2, lw=0.6, ls=":")
    ax.set(xlim=(7, 12), ylim=(0, 1.04), ylabel="magnitude", yticks=[0, 0.5, 1]); ax.set_xticklabels([])
    ax.text(7.15, 0.84, "$|r_{xx}|$", color=ORANGE, fontsize=8, va="top")
    ax.text(7.15, 0.36, "$|r_{yx}|$", color=BLUE, fontsize=8, va="bottom")
    ax.text(11.95, 1.03, "solid: FDTD, dotted: CST, dashed: surrogate", fontsize=8, ha="right", va="bottom", color=INK2)
    tidy(ax)
    ax = fig.add_subplot(gs[2, j])
    if app == "cp":
        m = lambda S: axial_ratio_db(S[:1, 0], S[:1, 1])[0][0]
        ax.axhline(3, color=GREEN, ls="--", lw=0.8); ax.set_ylim(0, 8); ax.set_ylabel("AR (dB)")
    else:
        m = lambda S: (np.abs(S[0, 1]) ** 2 / (np.abs(S[0]) ** 2).sum(0))
        ax.axhline(0.9, color=GREEN, ls="--", lw=0.8); ax.set_ylim(0.5, 1.02); ax.set_ylabel("PCR")
    ax.axvspan(r["f_lo"][i], r["f_hi"][i], color=RED, alpha=0.10, lw=0)
    if mc_app[app] is not None:
        Q = np.stack([m(mc_app[app]["spectra"][k:k + 1, MC_K[app]]) for k in range(len(mc_app[app]["spectra"]))])
        ax.fill_between(F, np.percentile(Q, 5, 0), np.percentile(Q, 95, 0), color=GREY, alpha=0.35, lw=0)
    ax.plot(F, m(r["S0"][i:i + 1]), color=RED, lw=1.2)
    ax.plot(F, m(r["Scst"][i:i + 1]), color=RED, lw=1.0, ls=":")
    ax.plot(F, m(r["pred"][i:i + 1]), color=INK, lw=1.0, ls="--")
    ax.axvline(a["f0"], color=INK2, lw=0.6, ls=":")
    ax.set(xlim=(7, 12), xlabel="frequency (GHz)")
    tidy(ax)
# coding column
ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
for k, img in enumerate(pair):
    sub = ax.inset_axes([0.0 + 0.27 * k, 0.0, 0.24, 0.55]); show_layout(sub, img)
N = max(boards) if boards else 6          # one period of the largest simulated board
board = np.block([[pair[(I // N + J // N) % 2] for J in range(2 * N)] for I in range(2 * N)])
sub = ax.inset_axes([0.60, 0.0, 0.40, 0.70])
sub.imshow(layout_image(board, k=2), interpolation="none"); sub.set_xticks([]); sub.set_yticks([])
for s_ in sub.spines.values():
    s_.set_linewidth(0.6); s_.set_edgecolor(INK2)
sub.axhline(N * 16 * 2 - 0.5, color=BLUE, lw=0.9); sub.axvline(N * 16 * 2 - 0.5, color=BLUE, lw=0.9)
ax.text(0.0, 1.0, "(c) coding: cells A, B and\none period of the board", transform=ax.transAxes, fontsize=8, va="top")
ax = fig.add_subplot(gs[1:, 2])
ax.axhline(10, color=GREEN, ls="--", lw=0.8)
ax.plot(F, red(s_lib), color=GREY, lw=1.0, label="library pair ($x$)")
ax.plot(F, coding.reduction(zc["pred"][MI:MI + 1])[0], color=BLUE, lw=1.0, ls="--", label="surrogate, (9)")
if mc is not None:
    ax.fill_between(F, np.percentile(mc["red"], 5, 0), np.percentile(mc["red"], 95, 0), color=BLUE, alpha=0.18, lw=0,
                    label="tolerances, 5–95 %")
ax.plot(F, coding.reduction(zc["fdtd_cells_h0"][MI:MI + 1])[0], color=BLUE, lw=1.2, label="FDTD cells, (9)")
for N, col, lw in ((3, "#f4a582", 1.0), (4, RED, 1.0), (6, "#7b3294", 1.2), (8, INK, 1.6)):
    if N in boards:
        ax.plot(F, red(boards[N]), color=col, lw=lw, label=f"board, ${N}\\times{N}$ tiles")
ax.set(xlim=(8, 12), ylim=(0, 62), yticks=[0, 10, 20, 30], xlabel="frequency (GHz)", ylabel="specular reduction (dB)")
ax.legend(loc="upper left", fontsize=8, handlelength=1.4, labelspacing=0.2, borderaxespad=0.2)
tidy(ax)
save(fig, str(FIG / "fig_designs.pdf"))
print("figures written")
