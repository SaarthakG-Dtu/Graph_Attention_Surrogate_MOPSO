"""
schematics.py — the three schematic figures of the manuscript as vector TikZ:

    fig_architecture   forward surrogate (PGA-ResNet -> cross-attention -> BiGRU -> polar head)
    fig_pipeline       VAE-latent MOIPSO inverse design + CST verification
    fig_unitcell       SUTD-PRCM unit cell and the reflection channels

    python paper/schematics.py [architecture pipeline unitcell]

Real data are drawn into the diagrams: a test layout, the trained surrogate's
feature maps, PGA relation/attention matrices, cross-attention weights and
predicted spectrum (data step, needs a trained pooled surrogate), and VAE
decodes. The data step writes textures and tables to paper/schematics/; the
.tex sources there are compiled with pdflatex into paper/figures/<name>.pdf
(+ a 300-dpi PNG preview).
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import utils
from dataset import get_split_arrays, freq_grid

SRC = os.path.join(ROOT, "paper", "schematics")
TEX = os.path.join(SRC, "tex")
DAT = os.path.join(SRC, "data")
FIG = os.path.join(ROOT, "paper", "figures")
for d in (TEX, DAT, FIG):
    os.makedirs(d, exist_ok=True)

COPPER = np.array([0.80, 0.50, 0.20])
SUBSTRATE = np.array([0.93, 0.91, 0.84])


# ─────────────────────────────────────────────────────────────────────────────
# Data step
# ─────────────────────────────────────────────────────────────────────────────

def _png(name, img, cmap=None, vmin=None, vmax=None):
    """Save an array as a pixel-exact PNG texture (nearest-neighbour upscaled)."""
    path = os.path.join(DAT, name + ".png")
    if img.ndim == 2:
        img = plt.get_cmap(cmap)(plt.Normalize(vmin, vmax)(img))[..., :3]
    k = max(1, 256 // img.shape[0])
    plt.imsave(path, np.clip(np.kron(img, np.ones((k, k, 1))), 0, 1))


def _layout_rgb(x):
    return utils.layout_image(x, k=16)             # shaded copper on substrate, 256 x 256


def _pick_example(cls="RDN", min_dip=0.25):
    """A test layout of `cls` with a deep co-pol resonance that the surrogate predicts well."""
    from phase2_surrogate import Surrogate
    sur = Surrogate()                      # the RDN surrogate of Phase 2
    x, y, _, idx = get_split_arrays(cls, "test", "co", convention=sur.convention,
                                    msdcnn_train=sur.msdcnn_train)   # the surrogate's own test set
    R = y[..., 0] + 1j * y[..., 1]
    pR = sur.predict(x)
    err = np.mean((np.abs(pR) - np.abs(R)) ** 2, 1)
    deep = np.abs(R).min(1) < min_dip
    i = int(np.flatnonzero(deep)[np.argmin(err[deep])])
    return sur, x[i], R[i], pR[i], cls, int(idx[i])


@torch.no_grad()
def data_architecture():
    sur, x, R, pR, cls, idx = _pick_example()
    model = sur.model
    xt = torch.as_tensor(x, dtype=torch.float32, device=C.DEVICE)[None, None]
    bb = model.backbone
    acts, atts = {}, {}

    # PGA attention of the first head of the first PGA block in stage 2 (8x8 grid)
    gat = bb.pga_stacks[1][0].attentions[0]
    feats = bb.layers[1](bb.layers[0](bb.stem(xt)))
    th = gat.theta(feats).flatten(2).transpose(1, 2)
    ph = gat.phi(feats).flatten(2)
    Rm = torch.bmm(th, ph)[0].cpu().numpy()
    from phase1_forward_model import build_adjacency
    A = build_adjacency(8, 8, 8, "cpu").numpy()
    Rmask = np.where(A > 0, Rm, -np.inf)
    att = np.exp(Rmask - Rmask.max(1, keepdims=True)); att /= att.sum(1, keepdims=True)

    s = bb.stem(xt); acts["stem"] = s
    z = s
    for k, (layer, pga) in enumerate(zip(bb.layers, bb.pga_stacks)):
        z = layer(z); acts[f"bb{k + 1}"] = z
        z = pga(z); acts[f"pga{k + 1}"] = z
    tokens = z.flatten(2).transpose(1, 2)
    ca = model.cross_attn
    from phase1_forward_model import sinusoidal_pe
    q = ca.query_embed.weight + sinusoidal_pe(ca.n_queries, ca.embed_dim, C.DEVICE)
    lay = ca.layers[0]
    _, w = lay.attn(lay.norm_q(q[None]), ca.norm_kv(tokens), ca.norm_kv(tokens),
                    need_weights=True, average_attn_weights=True)
    w = w[0].cpu().numpy()                                           # (101, 16)

    # textures
    _png("layout", _layout_rgb(x))
    for name, a in acts.items():
        a = a[0].cpu().numpy()
        order = np.argsort(-a.reshape(len(a), -1).std(1))           # most structured channels first
        for j in range(3):
            m = a[order[j]]
            _png(f"{name}_c{j}", m, "magma", m.min(), m.max() + 1e-9)
    _png("pga_A", A, "Greys", 0, 1.6)
    _png("pga_R", Rm, "Blues", np.percentile(Rm, 2), np.percentile(Rm, 98))
    _png("pga_att", att, "Oranges", 0, att.max())
    _png("xattn", w.T, "Greens", 0, w.max())                         # 16 tokens x 101 freqs
    _png("tokens", acts["pga3"][0].cpu().numpy().reshape(256, 16)[:24].T, "magma")

    # pixel graph: attention of the centre node (3,3) of the 8x8 grid to its neighbours
    c = 3 * 8 + 3
    graph = [dict(dr=int(j // 8 - 3), dc=int(j % 8 - 3), w=float(att[c, j])) for j in np.flatnonzero(A[c])]
    f = freq_grid()
    with open(os.path.join(DAT, "spectrum.dat"), "w") as fh:
        fh.write("f absR absRp phR phRp re im rep imp\n")
        for k in range(len(f)):
            fh.write(f"{f[k]:.2f} {abs(R[k]):.5f} {abs(pR[k]):.5f} {np.degrees(np.angle(R[k])):.3f} "
                     f"{np.degrees(np.angle(pR[k])):.3f} "
                     f"{R[k].real:.5f} {R[k].imag:.5f} {pR[k].real:.5f} {pR[k].imag:.5f}\n")
    with open(os.path.join(DAT, "layout_bits.tex"), "w") as fh:   # copper pixels (row, col) for the unit cell
        fh.write(r"\def\layoutbits{" + ",".join(f"{r}/{c}" for r, c in zip(*np.nonzero(x > 0.5))) + "}\n")
    with open(os.path.join(DAT, "graph.tex"), "w") as fh:
        wmax = max(e["w"] for e in graph)
        fh.write(r"\def\pgagraph{" + ",".join(f"{e['dc']}/{-e['dr']}/{e['w'] / wmax:.3f}" for e in graph) + "}\n")
    meta = dict(cls=cls, index=idx, graph=graph,
                gamma=[[float(b.gama) for b in st] for st in bb.pga_stacks],
                n_params=int(sum(p.numel() for p in model.parameters())))
    json.dump(meta, open(os.path.join(DAT, "architecture.json"), "w"), indent=1)
    print(f"  example {cls} #{idx}; PGA gammas {meta['gamma']}")


@torch.no_grad()
def data_pipeline(vae_tag="rdn_d150", n=6, seed=3):
    """VAE decodes along a latent path (for the latent-space panel)."""
    from phase2_vae import load_vae
    path = os.path.join(C.vae_dir(vae_tag), "vae_best.pth")
    vae = load_vae(path)
    g = torch.Generator().manual_seed(seed)
    z0, z1 = torch.randn(vae.latent_dim, generator=g), torch.randn(vae.latent_dim, generator=g)
    for j, t in enumerate(np.linspace(0, 1, n)):
        z = ((1 - t) * z0 + t * z1).to(C.DEVICE)[None]
        img = vae.decode(z)
        img = (img[0, 0] if img.dim() == 4 else img[0].reshape(16, 16)).cpu().numpy()
        _png(f"vae_{j}", _layout_rgb(img > 0.5))

    # objective space of the RDN database (CST spectra) for the headline problem
    from phase2_mopso import Objectives, non_dominated, surface_phase_deg
    from phase2_surrogate import database
    f = freq_grid()
    obj = Objectives(C.OBJECTIVE_PRESETS[C.MOPSO_PROBLEM], f)
    x, R, _, _ = database(classes=["RDN"], convention="physical")
    Fo, Vo = obj(R, x)
    nd = np.flatnonzero(non_dominated(Fo))
    rng = np.random.default_rng(0)
    inband = np.flatnonzero(Vo[:, 0] > 0)                     # layouts with an in-phase band at f0
    cloud = rng.choice(np.setdiff1d(inband, nd), 1200, replace=False)
    nd = nd[Vo[nd, 0] > 0]
    for name, sel in (("front", nd), ("cloud", cloud)):
        V = Vo[sel].copy(); V[:, 1] = np.maximum(V[:, 1], -20)
        np.savetxt(os.path.join(DAT, name + ".dat"), V, fmt="%.5f", header="B L D", comments="")
    # example spectrum: the best-compromise design of the long search (prediction)
    run = os.path.join(C.mopso_dir(C.MOPSO_PROBLEM, "long"), "Final", "pareto_designs.npz")
    d = np.load(run)
    k = int(np.argmin(d["objectives"].sum(1)))
    Rk = d["R"][k]
    np.savetxt(os.path.join(DAT, "design.dat"),
               np.c_[f, 20 * np.log10(np.abs(Rk)), surface_phase_deg(Rk[None], f)[0]],
               fmt="%.4f", header="f db ph", comments="")
    _png("best_layout", _layout_rgb(d["patterns"][k]))
    A = d["values"][d["values"][:, 0] > 0].copy(); A[:, 1] = np.maximum(A[:, 1], -20)
    np.savetxt(os.path.join(DAT, "archive.dat"), A, fmt="%.5f", header="B L D", comments="")
    print(f"  database front: {len(nd)} designs; example = long-search design #{k + 1}: "
          + ", ".join(f"{n} = {v:.3g}" for n, v in zip(obj.names, d["values"][k])))


# ─────────────────────────────────────────────────────────────────────────────
# Compile
# ─────────────────────────────────────────────────────────────────────────────

def _pdflatex():
    exe = shutil.which("pdflatex")
    if exe:
        return exe
    guess = os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")
    if os.path.exists(guess):
        return guess
    raise FileNotFoundError("pdflatex not found")


def compile_tex(name):
    tex = os.path.join(TEX, name + ".tex")
    r = subprocess.run([_pdflatex(), "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                       cwd=TEX, capture_output=True, text=True, errors="replace")
    pdf = os.path.join(TEX, name + ".pdf")
    if r.returncode or not os.path.exists(pdf):
        log = r.stdout[-3000:]
        raise RuntimeError(f"pdflatex failed for {tex}:\n{log}")
    shutil.copy(pdf, os.path.join(FIG, name + ".pdf"))
    import fitz
    doc = fitz.open(pdf)
    doc[0].get_pixmap(dpi=300).save(os.path.join(FIG, name + ".png"))
    for ext in ("aux", "log"):
        p = os.path.join(TEX, f"{name}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    print(f"  -> paper/figures/{name}.pdf")


DATA = dict(architecture=data_architecture, pipeline=data_pipeline)

if __name__ == "__main__":
    which = sys.argv[1:] or ["architecture", "pipeline", "unitcell"]
    for w in which:
        if w.startswith("data:"):
            DATA[w[5:]]()
        else:
            compile_tex("fig_" + w)
