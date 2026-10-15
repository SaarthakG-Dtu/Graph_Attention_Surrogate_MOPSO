"""
report.py — table and figure of the VAE-space vs pixel-space scaling study
(scaling.json): hypervolume, wall time and layouts / seconds to reach given
hypervolume targets for each swarm size; HV-vs-layouts curves and final fronts.

    python experiments/vae_study/report.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper"))
import config as C                                   # noqa: E402
import utils                                         # noqa: E402
from make_figures import _front_history, _penalty_to_values   # noqa: E402

plt = utils.plt
SUF = "_trust" if C.MOPSO_TRUST > 0 else ""
S = json.load(open(os.path.join(HERE, f"scaling{SUF}.json")))
runs = S["runs"]
budgets = sorted({(r["Np"], r["gens"]) for r in runs}, key=lambda b: b[0] * b[1])
SPACES = [("latent", "VAE space", utils.SERIES[0]), ("binary", "pixel space", utils.SERIES[2])]


def sel(space, Np, gens):
    return [r for r in runs if r["space"] == space and r["Np"] == Np and r["gens"] == gens]


def reach(rs, tg, key):
    v = [r["reach"][str(tg)][key] if r["reach"].get(str(tg)) else None for r in rs]
    return None if any(x is None for x in v) else float(np.mean(v))


# ---------- table ----------
lines, summary = [], []
for Np, g in budgets:
    row = {"budget": f"{Np}x{g}"}
    for sp, _, _ in SPACES:
        rs = sel(sp, Np, g)
        if not rs:
            continue
        hv = np.array([r["hv_archive"] for r in rs]); ha = np.array([r["hv_all"] for r in rs])
        row[sp] = dict(n=len(rs), layouts=rs[0]["layouts"], seconds=float(np.mean([r["seconds"] for r in rs])),
                       hv=float(hv.mean()), hv_sd=float(hv.std()), hv_all=float(ha.mean()),
                       reach={str(t): (reach(rs, t, "layouts"), reach(rs, t, "seconds")) for t in (0.35, 0.40, 0.45)})
    summary.append(row)
json.dump(summary, open(os.path.join(HERE, f"report_summary{SUF}.json"), "w"), indent=1)

fmt = lambda v, f="%.0f": "--" if v is None else f % v
print(f"{'budget':10s} {'space':7s} {'layouts':>8s} {'time s':>7s} {'HV arch':>13s} {'HV all':>7s}  "
      f"{'to 0.35 (lay/s)':>18s} {'to 0.40':>18s} {'to 0.45':>18s}")
for row in summary:
    for sp, _, _ in SPACES:
        if sp not in row:
            continue
        d = row[sp]
        rr = [f"{fmt(d['reach'][t][0])}/{fmt(d['reach'][t][1], '%.1f')}" for t in ("0.35", "0.4", "0.45")]
        print(f"{row['budget']:10s} {sp:7s} {d['layouts']:8d} {d['seconds']:7.1f} {d['hv']:.3f}+-{d['hv_sd']:.3f} "
              f"{d['hv_all']:7.3f}  {rr[0]:>18s} {rr[1]:>18s} {rr[2]:>18s}")

# ---------- figure: HV vs layouts (top) and fronts B vs D (bottom), one column per budget ----------
fig, axes = plt.subplots(2, len(budgets), figsize=(utils.COL2, 3.5))
for c, (Np, g) in enumerate(budgets):
    ax, bx = axes[0, c], axes[1, c]
    for sp, lab, col in SPACES:
        curves, fronts = [], []
        for r in sel(sp, Np, g):
            d = C.mopso_dir(C.MOPSO_PROBLEM, r.get("tag", f"sc_{sp}_N{Np}_G{g}_s{r['seed']}" + ("_tr" if SUF else "")))
            h = np.load(os.path.join(d, "swarm_history.npz"))
            hv, snaps = _front_history(h["objectives"], h["generation"], 5)
            curves.append(hv); fronts.append(snaps[max(snaps)])
        x = (curves[0][:, 0] + 1) * Np
        Y = np.array([cv[:, 1] for cv in curves])
        ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=0.2, lw=0)
        ax.plot(x, Y.mean(0), color=col, lw=1.1, label=lab)
        V = _penalty_to_values(fronts[0]); V = V[V[:, 0] > 0]
        bx.scatter(V[:, 0], V[:, 2], s=12 if sp == "latent" else 6, color=col, label=lab, alpha=0.9, lw=0.3 if sp == "latent" else 0,
                   edgecolor="white", zorder=3 if sp == "latent" else 2)
    ax.axhline(0.295, color=utils.SERIES[1], ls=":", lw=0.9)
    ax.set_xscale("log"); ax.set_ylim(0.1, 0.65)
    ax.set_title(f"{Np} particles × {g} gen.", fontsize=8.5)
    ax.set_xlabel("layouts scored", fontsize=8); ax.tick_params(labelsize=7)
    bx.axvline(10, color="0.55", ls="--", lw=0.6); bx.axhline(5, color="0.55", ls="--", lw=0.6)
    bx.set_xlim(0, 22); bx.set_ylim(0, 45)
    bx.set_xlabel("B (%)", fontsize=8); bx.tick_params(labelsize=7)
    if c == 0:
        ax.set_ylabel("hypervolume", fontsize=8.5); bx.set_ylabel("D (fragile features)", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.text(0.03, 0.38, "dataset", transform=ax.transAxes, fontsize=6.5, color=utils.SERIES[1])
fig.tight_layout(h_pad=0.6, w_pad=0.5)
utils.save(fig, os.path.join(ROOT, "paper", "figures", "fig_spaces.png"))
print("figure written")
