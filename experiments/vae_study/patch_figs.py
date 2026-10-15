"""Point paper figures at the final (trusted) MOIPSO runs and replace panel (b)
of fig_pareto by the B-L projection of the fronts."""
import re

P = r"E:/SUTD_BENCHMARK_NOVEL_GNN/paper/make_figures.py"
t = open(P, encoding="utf-8").read()

start = t.index("def fig_pareto(")
end = t.index("def fig_cst_verification(")
new = '''RUN_LONG = "sc_latent_N1000_G300_s0_tr"     # final runs: VAE and pixel space, 1000 x 300, seed 0, trust check
RUN_PIX = "sc_binary_N1000_G300_s0_tr"


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


'''
t = t[:start] + new + t[end:]
t = t.replace('groups=(("long", "", 0, 3), ("long", "", 3, 6), ("binary", "P", 0, 3))):',
              'groups=((RUN_LONG, "", 0, 3), (RUN_LONG, "", 3, 6), (RUN_PIX, "P", 0, 3))):')
open(P, "w", encoding="utf-8").write(t)
print("patched")
