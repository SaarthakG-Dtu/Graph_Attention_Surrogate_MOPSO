"""
phase2_mopso.py — VAE-latent MOIPSO for 16x16 layouts scored by the frozen
co-pol forward surrogate (no full-wave solver in the loop).

Port of the 12x12 mopso_vae_latent.py / vae16_mopso.py:
  particles in R^d (VAE latent), decoded to a binary layout only for evaluation;
  inertia 0.9 -> 0.4; c1 = 2.05|cos 2pi r|, c2 = 2.05|sin 2pi r| per generation;
  Gaussian mutation of a random U_MUT fraction with sigma = max(0.02, 0.1(1-t/T))
  times the box width; reflective bounds; grid-based leader selection;
  crowding-distance archive pruning; Pareto personal-best rule; archive
  de-duplicated by decoded layout; whole swarm evaluated in one batch.
Search spaces: "latent" (VAE prior) or "binary" (256 pixels, threshold 0.5).

    python phase2_mopso.py --tag s0                        # config defaults
    python phase2_mopso.py --space binary --tag binary
    python phase2_mopso.py --database_only                 # SUTD baseline front

Outputs (config.mopso_dir(problem, tag)):
  run_config.json, convergence.png, Generation_XXX/,
  Final/ Pareto_Front.png, Pareto_Solution_XXX.png, pareto_summary.csv,
         pareto_designs.npz (-> cst_verify.py)
  Good_Solutions/, swarm_history.npz, database_front.csv/.npz, comparison.json
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import config as C
import utils
from phase2_surrogate import Surrogate, sanity_check, database

_CHECKERS = []


def disagreement(mats, R):
    """Mean squared difference of the complex r_xx between the main surrogate and
    two independently trained RDN surrogates (RDN108000 without PGA, RDN27000).
    Large values mark layouts on which the surrogate prediction is unreliable."""
    if not _CHECKERS:
        _CHECKERS.extend(Surrogate(s) for s in C.MOPSO_TRUST_CHECKERS)
    return np.mean([np.mean(np.abs(R - s.predict(mats)) ** 2, 1) for s in _CHECKERS], 0)
from phase2_vae import load_vae

plt = utils.plt
_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Objectives (co-pol r_xx only)
# ─────────────────────────────────────────────────────────────────────────────

def fragile_count(mats):
    """Fragile features of each layout: 2x2 windows in which metal touches only
    at a corner (diagonal-only contact), plus isolated single metal pixels."""
    m = np.asarray(mats).astype(np.int8)
    a, b, c, d = m[:, :-1, :-1], m[:, :-1, 1:], m[:, 1:, :-1], m[:, 1:, 1:]
    corner = ((a & d & (1 - b) & (1 - c)) | (b & c & (1 - a) & (1 - d))).sum((1, 2))
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
    nb = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]
    isolated = ((m == 1) & (nb == 0)).sum((1, 2))
    return (corner + isolated).astype(float)


def surface_phase_deg(R, freq_ghz):
    """Reflection phase moved from the port reference plane to the cell surface."""
    k = 2 * np.pi * np.asarray(freq_ghz) * 1e9 / 299792458.0
    return np.degrees(np.angle(R * np.exp(2j * k * C.PHASE_REF_SHIFT_MM * 1e-3)))


def inphase_band(R, freq_ghz, f0):
    """Band around f0 in which the surface phase stays within +-90 deg.
    Returns (lo index, hi index, fractional bandwidth %, phase at f0); the band
    is empty (lo = hi = -1, FBW = 0) when the phase at f0 is outside +-90 deg."""
    ph = surface_phase_deg(R, freq_ghz)
    ok = np.abs(ph) <= 90.0
    i0 = int(np.argmin(np.abs(np.asarray(freq_ghz) - f0)))
    n, K = ok.shape
    left = np.argmin(np.c_[ok[:, i0::-1], np.zeros((n, 1), bool)], axis=1)    # consecutive True from i0 down
    right = np.argmin(np.c_[ok[:, i0:], np.zeros((n, 1), bool)], axis=1)      # ... and up
    has = ok[:, i0]
    lo = np.where(has, i0 - left + 1, -1)
    hi = np.where(has, i0 + right - 1, -1)
    f = np.asarray(freq_ghz)
    fbw = np.where(has, 200 * (f[np.maximum(hi, 0)] - f[np.maximum(lo, 0)])
                   / (f[np.maximum(hi, 0)] + f[np.maximum(lo, 0)]), 0.0)
    return lo, hi, fbw, ph[:, i0]


class Objectives:
    KINDS = ("inphase_fbw", "inband_db", "fragile")

    def __init__(self, specs, freq):
        self.specs, self.freq = specs, np.asarray(freq)
        for s in specs:
            if s["kind"] not in self.KINDS:
                raise ValueError(f"objective {s['name']!r}: unknown kind {s['kind']!r}")
        self.names = [s["name"] for s in specs]
        self.ideal = np.array([s["ideal"] for s in specs], float)
        self.worst = np.array([s["worst"] for s in specs], float)
        self.good_value = np.array([s["good"] for s in specs], float)

    def values(self, R, mats):
        """Objective values in physical units, (N, n_obj), plus the in-phase band
        (lo, hi indices) and the surface phase at f0 used for guidance."""
        V = np.zeros((len(R), len(self.specs)))
        band, ph0 = None, None
        for j, s in enumerate(self.specs):
            if s["kind"] in ("inphase_fbw", "inband_db"):
                if band is None:
                    lo, hi, fbw, ph0 = inphase_band(R, self.freq, s["f0"])
                    band = (lo, hi, fbw)
                lo, hi, fbw = band
                if s["kind"] == "inphase_fbw":
                    V[:, j] = fbw
                else:
                    db = 20 * np.log10(np.abs(R) + _EPS)
                    i0 = int(np.argmin(np.abs(self.freq - s["f0"])))
                    idx = np.arange(db.shape[1])
                    inside = (idx[None] >= lo[:, None]) & (idx[None] <= hi[:, None])
                    V[:, j] = np.where(lo >= 0, np.where(inside, db, np.inf).min(1), db[:, i0])
            else:
                V[:, j] = fragile_count(mats)
        return V, band, ph0

    def penalties(self, V, ph0=None):
        """Normalised penalties (0 = ideal, 1 = worst). Designs whose phase at
        f0 is outside +-90 deg get an extra in-phase penalty that grows with the
        phase error, so the swarm is guided towards the in-phase condition."""
        P = (V - self.ideal) / (self.worst - self.ideal)
        if ph0 is not None:
            for j, s in enumerate(self.specs):
                if s["kind"] == "inphase_fbw":
                    P[:, j] += np.maximum(np.abs(ph0) - 90.0, 0) / 90.0
        return np.maximum(P, 0.0)

    def __call__(self, R, mats):
        V, _, ph0 = self.values(R, mats)
        return self.penalties(V, ph0), V

    def good(self, V):
        better_is_higher = self.ideal > self.worst
        return np.all(np.where(better_is_higher, V >= self.good_value, V <= self.good_value), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash(mat):
    return hashlib.md5(np.ascontiguousarray(mat, dtype=np.uint8).tobytes()).hexdigest()


def non_dominated(F):
    """Boolean mask of non-dominated rows (minimisation). Chunked for large N."""
    n = len(F)
    if n > 4000:                     # sort-and-filter for big sets (database front)
        keep = np.ones(n, bool)
        order = np.argsort(F.sum(1))
        front = []
        for i in order:
            f = F[i]
            if front and np.any(np.all(np.array(front) <= f, 1) & np.any(np.array(front) < f, 1)):
                keep[i] = False
            else:
                front.append(f)
        cand = np.where(keep)[0]
        sub = non_dominated(F[cand])
        keep[:] = False
        keep[cand[sub]] = True
        return keep
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    return ~dom.any(axis=0)


def dominates(a, b):
    return np.all(a <= b, axis=1) & np.any(a < b, axis=1)


def hypervolume(F, ref=None, n=200_000, seed=0):
    """Monte-Carlo hypervolume of the region dominated by F inside [0, ref]
    (penalties; ref = 1 = the worst value of every objective)."""
    ref = np.ones(F.shape[1]) if ref is None else np.asarray(ref)
    F = F[np.all(F < ref, 1)]
    if len(F) == 0:
        return 0.0
    pts = np.random.default_rng(seed).uniform(0, 1, (n, len(ref))) * ref
    dom = np.zeros(n, bool)
    for f in F:
        dom |= np.all(pts >= f, 1)
    return float(dom.mean() * np.prod(ref))


# ─────────────────────────────────────────────────────────────────────────────
# MOIPSO
# ─────────────────────────────────────────────────────────────────────────────

class MOIPSO:
    def __init__(self, decode, D, lb, ub, sur, objectives, maxvel):
        self.decode, self.D, self.lb, self.ub = decode, D, lb, ub
        self.sur, self.obj, self.maxvel = sur, objectives, maxvel
        self.M = len(objectives.specs)
        self.Np, self.Nr, self.maxgen = C.MOPSO_NP, C.MOPSO_NR, C.MOPSO_MAXGEN
        self.ngrid = int(np.clip(round(self.Nr ** (1 / self.M) * 2), 5, 30))
        K = len(sur.freq)
        self.rep = dict(z=np.zeros((0, D)), F=np.zeros((0, self.M)), V=np.zeros((0, self.M)),
                        mat=np.zeros((0, 16, 16), np.uint8), R=np.zeros((0, K), complex))
        self.grid_idx, self.quality = np.zeros(0, int), {}
        self.hist = dict(z=[], mat=[], F=[], gen=[])
        self.conv = []

    def evaluate(self, Z):
        mats = self.decode(Z)
        R = self.sur.predict(mats)
        F, V = self.obj(R, mats)
        if C.MOPSO_TRUST > 0:                     # trust check: surrogates must agree
            u = disagreement(mats, R) / C.MOPSO_TRUST
            bad = u > 1
            F[bad] = np.maximum(F[bad], 1.0) + (u[bad, None] - 1)
        return mats, R, F, V

    def _update_grid(self):
        F = self.rep["F"]
        if len(F) == 0:
            return
        lim = np.zeros((self.ngrid + 1, self.M))
        for d in range(self.M):
            mn, mx = F[:, d].min(), F[:, d].max()
            lim[:, d] = np.linspace(mn, mx if mx > mn else mn + 1e-5, self.ngrid + 1)
        sub = np.stack([np.clip(np.searchsorted(lim[:, d], F[:, d]) - 1, 0, self.ngrid - 1)
                        for d in range(self.M)], 1)
        self.grid_idx = np.ravel_multi_index(sub.T, [self.ngrid] * self.M)
        u, c = np.unique(self.grid_idx, return_counts=True)
        self.quality = dict(zip(u, 10.0 / c))

    def _leader(self):
        u, q = list(self.quality), np.cumsum(list(self.quality.values()))
        cell = u[np.searchsorted(q, np.random.rand() * q[-1])]
        cand = np.where(self.grid_idx == cell)[0]
        i = np.random.choice(cand) if len(cand) else np.random.randint(len(self.rep["z"]))
        return self.rep["z"][i]

    def _prune(self):
        n_extra = len(self.rep["F"]) - self.Nr
        if n_extra <= 0:
            return
        F = self.rep["F"]; n = len(F); crowd = np.zeros(n)
        for m in range(self.M):
            o = np.argsort(F[:, m]); f = F[o, m]; d = np.zeros(n)
            d[0] = d[-1] = np.inf
            if f[-1] > f[0]:
                d[1:-1] = (f[2:] - f[:-2]) / (f[-1] - f[0])
            crowd[o] += d
        keep = np.sort(np.argsort(crowd)[n_extra:])
        for k in self.rep:
            self.rep[k] = self.rep[k][keep]
        self._update_grid()

    def _update_repository(self, Z, mats, R, F, V):
        nd = non_dominated(F)
        c = {k: np.concatenate([self.rep[k], v[nd]]) for k, v in
             dict(z=Z, F=F, V=V, mat=mats, R=R).items()}
        keep = non_dominated(c["F"])
        seen, uniq = set(), []
        for i in np.where(keep)[0]:
            h = _hash(c["mat"][i])
            if h not in seen:
                seen.add(h); uniq.append(i)
        uniq = np.array(uniq, int)
        self.rep = {k: v[uniq] for k, v in c.items()}
        self._update_grid()
        self._prune()

    def optimize(self, out_dir):
        rng = np.random.default_rng(C.MOPSO_SEED); np.random.seed(C.MOPSO_SEED)
        width = self.ub - self.lb
        if C.MOPSO_INIT == "prior" and self.lb < 0:
            POS = np.clip(rng.standard_normal((self.Np, self.D)), self.lb, self.ub)
        else:
            POS = rng.uniform(self.lb, self.ub, (self.Np, self.D))
        VEL = np.zeros_like(POS)
        mats, R, F, V = self.evaluate(POS)
        self._log(0, POS, mats, F)
        PB, PBF = POS.copy(), F.copy()
        self._update_repository(POS, mats, R, F, V)
        self._record(0, F)
        prev = []
        for gen in range(1, self.maxgen + 1):
            W = C.MOPSO_W_START - (C.MOPSO_W_START - C.MOPSO_W_END) * gen / self.maxgen
            C1 = 2.05 * abs(np.cos(2 * np.pi * np.random.rand()))
            C2 = 2.05 * abs(np.sin(2 * np.pi * np.random.rand()))
            L = np.stack([self._leader() for _ in range(self.Np)])
            r1, r2 = np.random.rand(self.Np, self.D), np.random.rand(self.Np, self.D)
            VEL = W * VEL + C1 * r1 * (PB - POS) + C2 * r2 * (L - POS)
            POS = POS + VEL
            sigma = max(0.02, 0.1 * (1 - gen / self.maxgen))
            mut = np.random.rand(self.Np) < C.MOPSO_U_MUT
            POS[mut] += np.random.randn(mut.sum(), self.D) * sigma * width
            out = (POS > self.ub) | (POS < self.lb)
            VEL[out] *= -1
            POS = np.clip(POS, self.lb, self.ub)
            VEL = np.clip(VEL, -self.maxvel, self.maxvel)

            mats, R, F, V = self.evaluate(POS)
            self._log(gen, POS, mats, F)
            better = dominates(F, PBF)
            tie = ~better & ~dominates(PBF, F) & (np.random.rand(self.Np) > 0.5)
            upd = better | tie
            PB[upd], PBF[upd] = POS[upd], F[upd]
            prev_front = self.rep["V"].copy()
            self._update_repository(POS, mats, R, F, V)
            self._record(gen, F)
            c = self.conv[-1]
            if gen % 10 == 0 or gen == 1:
                print(f"  gen {gen:4d}/{self.maxgen}  archive={c['archive']:3d}  good={c['n_good']:3d}  best="
                      + "  ".join(f"{v:.3g}" for v in c["best_value"]), flush=True)
            if gen % C.MOPSO_SAVE_EVERY == 0 and gen != self.maxgen:
                gdir = os.path.join(out_dir, f"Generation_{gen:03d}")
                os.makedirs(gdir, exist_ok=True)
                plot_front(self.rep["V"], self.obj.names, os.path.join(gdir, "Pareto_Front.png"),
                           prev=np.array(prev))
                write_summary(self, os.path.join(gdir, "Pareto_Front_Summary.txt"), gen)
                prev = list(prev) + list(prev_front)
        return self.rep

    def _log(self, gen, Z, mats, F):
        self.hist["z"].append(Z.astype(np.float32)); self.hist["mat"].append(mats)
        self.hist["F"].append(F); self.hist["gen"].append(np.full(len(F), gen))

    def _record(self, gen, F):
        V = self.rep["V"]
        best = np.where(self.obj.ideal > self.obj.worst, V.max(0), V.min(0))
        self.conv.append(dict(gen=gen, best_value=best.tolist(), swarm_mean=F.mean(0).tolist(),
                              archive=len(V), n_good=int(self.obj.good(V).sum()),
                              hv=hypervolume(self.rep["F"], n=20_000)))


# ─────────────────────────────────────────────────────────────────────────────
# Plots / outputs
# ─────────────────────────────────────────────────────────────────────────────

def plot_front(V, names, path, prev=None, db=None):
    """Pairwise projections of the archive in physical units."""
    M = V.shape[1]
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(utils.COL2, utils.COL2 / len(pairs) + 0.2), squeeze=False)
    for ax, (i, j) in zip(axes.flat, pairs):
        if prev is not None and len(prev):
            ax.scatter(prev[:, i], prev[:, j], s=3, color=utils.GRID, lw=0, label="earlier archive", rasterized=True)
        if db is not None and len(db):
            ax.scatter(db[:, i], db[:, j], s=10, marker="s", facecolor="none", edgecolor=utils.SERIES[1],
                       lw=0.7, label="database front (CST)")
        ax.scatter(V[:, i], V[:, j], s=12, color=utils.SERIES[0], edgecolor="white", lw=0.3, label="MOIPSO archive")
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j])
    axes.flat[0].legend(loc="best", fontsize=6)
    fig.tight_layout()
    utils.save(fig, path)


def plot_solution(mat, R, freq, values, objectives, path, title=""):
    lo, hi, _, _ = inphase_band(R[None], freq, objectives.specs[0].get("f0", 10.0))
    ph = surface_phase_deg(R[None], freq)[0]
    fig = plt.figure(figsize=(utils.COL2, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.5, 1.5])
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(mat, cmap=utils.PATTERN_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1, ax2 = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ax1.plot(freq, 20 * np.log10(np.abs(R) + _EPS), color=utils.SERIES[0])
    ax1.set_ylabel(r"$|r_{xx}|$ (dB)"); ax1.set_ylim(-30, 1)
    ax2.plot(freq, ph, color=utils.SERIES[0])
    for y in (-90, 90):
        ax2.axhline(y, color=utils.INK2, lw=0.6, ls=":")
    ax2.set_ylabel("surface phase (deg)"); ax2.set_ylim(-185, 185); ax2.set_yticks([-180, -90, 0, 90, 180])
    for ax in (ax1, ax2):
        if lo[0] >= 0:
            ax.axvspan(freq[lo[0]], freq[hi[0]], color=utils.SERIES[2], alpha=0.15, lw=0)
        ax.set_xlabel("Frequency (GHz)"); ax.set_xlim(freq[0], freq[-1])
    fig.suptitle(title + "  [" + ", ".join(f"{n} = {v:.3g}" for n, v in zip(objectives.names, values)) + "]",
                 fontsize=7)
    fig.tight_layout()
    utils.save(fig, path)


def write_summary(opt, path, gen):
    rep, names = opt.rep, opt.obj.names
    with open(path, "w") as f:
        f.write(f"VAE-MOIPSO (surrogate) Pareto archive - generation {gen}\n" + "  ".join(names) + "\n" + "=" * 70 + "\n")
        for i, (mat, V) in enumerate(zip(rep["mat"], rep["V"])):
            f.write(f"\nSolution {i + 1}: " + "  ".join(f"{n} = {v:.4g}" for n, v in zip(names, V)) + "\n")
            for row in mat:
                f.write("  " + " ".join(map(str, row)) + "\n")


def plot_convergence(conv, names, path):
    g = [c["gen"] for c in conv]
    best = np.array([c["best_value"] for c in conv])
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(utils.COL2, 1.8))
    for j, n in enumerate(names):
        ax[j].plot(g, best[:, j], color=utils.SERIES[j]); ax[j].set_title(f"best: {n}", fontsize=6)
    ax[-1].plot(g, [c["hv"] for c in conv], color=utils.SERIES[3]); ax[-1].set_title("hypervolume", fontsize=6)
    for a in ax:
        a.set_xlabel("generation")
    fig.tight_layout()
    utils.save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Database baseline: the best SUTD-PRCM layouts for the same objectives
# ─────────────────────────────────────────────────────────────────────────────

def database_front(objectives, out_dir, sur):
    x, R, labels, idx = database([sur.cls] if sur.cls != C.POOLED_CLASS else C.CLASSES, convention=sur.convention)
    F, V = objectives(R, x)
    nd = np.where(non_dominated(F))[0]
    with open(os.path.join(out_dir, "database_front.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "folder_index"] + objectives.names)
        for i in nd:
            w.writerow([labels[i], int(idx[i])] + [f"{v:.4g}" for v in V[i]])
    Fs, Vs = objectives(sur.predict(x[nd]), x[nd])           # same front scored by the surrogate
    res = dict(n_database=int(len(x)), n_front=int(len(nd)), n_good=int(objectives.good(V).sum()),
               hv_true=hypervolume(F[nd]), hv_surrogate=hypervolume(Fs))
    np.savez(os.path.join(out_dir, "database_front.npz"), patterns=x[nd], R=R[nd], F=F[nd], V=V[nd],
             V_surrogate=Vs, labels=labels[nd], folder_index=idx[nd], layout_convention=sur.convention)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    specs = C.OBJECTIVE_PRESETS[args.problem]
    run_dir = C.mopso_dir(args.problem, args.tag)
    final_dir, good_dir = os.path.join(run_dir, "Final"), os.path.join(run_dir, "Good_Solutions")
    for d in (final_dir, good_dir):
        os.makedirs(d, exist_ok=True)
    print(f"\nVAE-MOIPSO [{args.problem} / {args.space}] -> {run_dir}")
    sur = Surrogate(args.surrogate)
    san = sanity_check(sur)
    print(f"  [sanity] {sur.cls} test split n={san['n']}: lin MSE={san['lin_mse']:.2e} "
          f"R2={san['r2']:.3f} phase MAE={san['phase_mae_deg']:.1f} deg")
    obj = Objectives(specs, sur.freq)
    for s in specs:
        print(f"  objective: {s['name']:<26s} ideal={s['ideal']:g} worst={s['worst']:g} spec={s['good']:g}")

    db = database_front(obj, run_dir, sur)
    print(f"  [database] {db['n_database']} layouts: front={db['n_front']}  meeting the spec={db['n_good']}  "
          f"HV={db['hv_true']:.4f}")
    if args.database_only:
        json.dump(db, open(os.path.join(run_dir, "database_baseline.json"), "w"), indent=1)
        return

    if args.space == "latent":
        vae = load_vae(os.path.join(C.vae_dir(args.vae_tag), "vae_best.pth"))
        D, lb, ub, maxvel = vae.latent_dim, -C.LATENT_BOUND, C.LATENT_BOUND, C.LATENT_MAXVEL

        @torch.no_grad()
        def decode(Z):
            z = torch.as_tensor(Z, dtype=torch.float32, device=C.DEVICE)
            return vae.decode(z).cpu().numpy()[:, 0].astype(np.uint8)
    else:
        D, lb, ub = 256, 0.0, 1.0
        maxvel = C.PIXEL_MAXVEL

        def decode(Z):
            return (Z > 0.5).reshape(-1, 16, 16).astype(np.uint8)

    t0 = time.time()
    opt = MOIPSO(decode, D, lb, ub, sur, obj, maxvel)
    rep = opt.optimize(run_dir)
    dt = time.time() - t0
    n_eval = sum(len(x) for x in opt.hist["F"])
    allF = np.vstack(opt.hist["F"])
    print(f"  done: {n_eval:,} surrogate evaluations in {dt:.1f}s, archive = {len(rep['F'])}")

    order = np.argsort(rep["F"].sum(1))                        # best compromise first
    rep = {k: v[order] for k, v in rep.items()}
    dbz = np.load(os.path.join(run_dir, "database_front.npz"))
    plot_front(rep["V"], obj.names, os.path.join(final_dir, "Pareto_Front.png"), db=dbz["V"])
    plot_convergence(opt.conv, obj.names, os.path.join(run_dir, "convergence.png"))
    write_summary(opt, os.path.join(final_dir, "Pareto_Front_Summary.txt"), C.MOPSO_MAXGEN)
    good = obj.good(rep["V"])
    lo, hi, _, _ = inphase_band(rep["R"], sur.freq, specs[0].get("f0", 10.0))
    with open(os.path.join(final_dir, "pareto_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solution"] + obj.names + ["band_lo_GHz", "band_hi_GHz", "meets_spec", "fill"])
        for i in range(len(rep["F"])):
            w.writerow([i + 1] + [f"{v:.4g}" for v in rep["V"][i]]
                       + [f"{sur.freq[lo[i]]:.2f}" if lo[i] >= 0 else "", f"{sur.freq[hi[i]]:.2f}" if hi[i] >= 0 else "",
                          bool(good[i]), f"{rep['mat'][i].mean():.3f}"])
    for i in range(len(rep["F"])):
        plot_solution(rep["mat"][i], rep["R"][i], sur.freq, rep["V"][i], obj,
                      os.path.join(final_dir, f"Pareto_Solution_{i + 1:03d}.png"), f"Pareto solution #{i + 1}")
        if good[i]:
            plot_solution(rep["mat"][i], rep["R"][i], sur.freq, rep["V"][i], obj,
                          os.path.join(good_dir, f"good_solution_{i + 1:03d}.png"), f"Good solution #{i + 1}")
    np.savez(os.path.join(final_dir, "pareto_designs.npz"), z=rep["z"], patterns=rep["mat"], R=rep["R"],
             freq_ghz=sur.freq, objectives=rep["F"], values=rep["V"], objective_names=np.array(obj.names),
             problem=args.problem, layout_convention=sur.convention)
    np.savez_compressed(os.path.join(run_dir, "swarm_history.npz"), patterns=np.concatenate(opt.hist["mat"]),
                        objectives=allF, generation=np.concatenate(opt.hist["gen"]))
    uniq = len({_hash(m) for m in np.concatenate(opt.hist["mat"])})
    best = np.where(obj.ideal > obj.worst, rep["V"].max(0), rep["V"].min(0))
    comp = dict(problem=args.problem, space=args.space, surrogate=args.surrogate, layout_convention=sur.convention,
                seconds=dt, evaluations=n_eval, unique_designs=uniq, archive=len(rep["F"]),
                n_good=int(good.sum()), best_value=best.tolist(), objective_names=obj.names,
                hv=hypervolume(rep["F"]), database=db)
    json.dump(comp, open(os.path.join(run_dir, "comparison.json"), "w"), indent=1)
    cfg = {k: getattr(C, k) for k in dir(C) if k.startswith(("MOPSO_", "LATENT_", "PHASE_REF"))}
    json.dump(dict(config=cfg, objectives=specs, args=vars(args), sanity=san),
              open(os.path.join(run_dir, "run_config.json"), "w"), indent=2, default=str)
    print(f"  unique designs evaluated: {uniq:,}/{n_eval:,}  meeting the spec: {int(good.sum())}  "
          f"best values: " + ", ".join(f"{n} = {v:.3g}" for n, v in zip(obj.names, best))
          + f"  HV={comp['hv']:.4f} (database {db['hv_true']:.4f})")


if __name__ == "__main__":
    C.be_nice()
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=C.MOPSO_PROBLEM, choices=list(C.OBJECTIVE_PRESETS))
    ap.add_argument("--space", default=C.MOPSO_SEARCH_SPACE, choices=["latent", "binary"])
    ap.add_argument("--surrogate", default=C.MOPSO_SURROGATE, help='co-pol surrogate run, "CLASS[:tag]"')
    ap.add_argument("--vae_tag", default=C.MOPSO_VAE_TAG)
    ap.add_argument("--tag", default="")
    ap.add_argument("--database_only", action="store_true")
    ap.add_argument("--gens", type=int, default=None, help="override MOPSO_MAXGEN")
    ap.add_argument("--swarm", type=int, default=None, help="override MOPSO_NP")
    ap.add_argument("--seed", type=int, default=None, help="override MOPSO_SEED")
    a = ap.parse_args()
    if a.gens:
        C.MOPSO_MAXGEN = a.gens
    if a.swarm:
        C.MOPSO_NP = a.swarm
    if a.seed is not None:
        C.MOPSO_SEED = a.seed
    main(a)
