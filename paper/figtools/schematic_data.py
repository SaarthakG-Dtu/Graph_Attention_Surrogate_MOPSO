"""Data for the TikZ schematics (architecture, method), from our trained networks and results.

  python schematic_data.py      -> ../schematics_v2/data/
Textures are pixel-exact PNGs; curves are whitespace tables read by pgfplots.
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
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from paper_style import layout_image
import config as C
from dataset import get_split_arrays, load_class, to_convention
from phase1_forward_model import load_surrogate, sinusoidal_pe
from phase2_devices import DeviceObjectives, axial_ratio_db
from phase2_mopso import non_dominated
from phase2_vae import load_vae
from train_phase1 import predict

OUT = HERE.parent / "schematics_v2" / "data"; OUT.mkdir(parents=True, exist_ok=True)
F = np.linspace(2, 12, 101)
dev = C.DEVICE


def png(name, img, cmap=None, vmin=None, vmax=None):
    if img.ndim == 2:
        img = plt.get_cmap(cmap)(plt.Normalize(vmin, vmax)(img))[..., :3]
    k = max(1, 256 // img.shape[0])
    plt.imsave(OUT / f"{name}.png", np.clip(np.kron(img, np.ones((k, k, 1))), 0, 1))


# ------------------------------------------------------------------ architecture (four-output RDN108000)
net = load_surrogate(os.path.join(C.phase1_dir("RDN", "both", "n108000"), "phase1_forward_best.pth"))
X, y, _, _ = get_split_arrays("RDN", "test", "both", "true", msdcnn_train=108000)
Y = y[..., 0] + 1j * y[..., 1]
p = predict(net, X.astype(np.float32), tta=True, target="both"); P = p[..., 0] + 1j * p[..., 1]
err = ((np.abs(P) - np.abs(Y)) ** 2).mean((1, 2))
deep = (np.abs(Y[:, 0]).min(1) < 0.3) & (np.abs(Y[:, 1]).max(1) > 0.5)
i = int(np.flatnonzero(deep)[np.argmin(err[deep])])
x_c = X[i].astype(bool)
png("layout", layout_image(x_c))
bb, acts = net.backbone, {}
with torch.no_grad():
    h = bb.stem(torch.tensor(x_c[None, None], dtype=torch.float32, device=dev)); acts["stem"] = h
    for s, (layer, pga) in enumerate(zip(bb.layers, bb.pga_stacks), 1):
        h = layer(h); acts[f"bb{s}"] = h
        if s == 2:
            feats_pga2_in = h
        h = pga(h); acts[f"pga{s}"] = h
    tokens = h.flatten(2).transpose(1, 2)
    ca = net.cross_attn
    q = (ca.query_embed.weight + sinusoidal_pe(ca.n_queries, ca.embed_dim, dev))[None]
    kv = ca.norm_kv(tokens)
    lay0 = ca.layers[0]
    _, w = lay0.attn(lay0.norm_q(q), kv, kv, need_weights=True, average_attn_weights=True)
    w = w[0].cpu().numpy()                                             # 101 x 16
    g = bb.pga_stacks[1][0].attentions[0]                              # first PGA block of stage 2
    th = g.theta(feats_pga2_in)[0].flatten(1).T                        # N x r
    ph = g.phi(feats_pga2_in)[0].flatten(1)                            # r x N
    Rm = (th @ ph).cpu().numpy()                                       # N x N
for name, a in acts.items():
    a = a[0].cpu().numpy()
    order = np.argsort(-a.reshape(len(a), -1).std(1))
    for j in range(3):
        m = a[order[j]]; png(f"{name}_c{j}", m, "magma", m.min(), m.max() + 1e-9)
n8 = 8
A = np.zeros((64, 64))
for r in range(n8):
    for c in range(n8):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < n8 and 0 <= c + dc < n8:
                    A[r * n8 + c, (r + dr) * n8 + c + dc] = 1
Rmask = np.where(A > 0, Rm, -np.inf)
att = np.exp(Rmask - Rmask.max(1, keepdims=True)); att /= att.sum(1, keepdims=True)
png("pga_A", A, "Greys", 0, 1.6)
png("pga_R", Rm, "Blues", np.percentile(Rm, 2), np.percentile(Rm, 98))
png("pga_att", att, "Oranges", 0, att.max())
png("xattn", w.T, "Greens", 0, w.max())
png("tokens", acts["pga3"][0].cpu().numpy().reshape(256, 16)[:24].T, "magma")
c0 = 3 * 8 + 3
graph = [(int(j % 8 - 3), int(-(j // 8 - 3)), float(att[c0, j])) for j in np.flatnonzero(A[c0])]
wmax = max(g_[2] for g_ in graph)
(OUT / "graph.tex").write_text(r"\def\pgagraph{" + ",".join(f"{dc}/{dr}/{wt / wmax:.3f}" for dc, dr, wt in graph) + "}\n")
R, Rp = Y[i], P[i]
with open(OUT / "spectrum.dat", "w") as fh:
    fh.write("f absR absRp absY absYp re im rep imp reY imY repY impY\n")
    for k in range(101):
        fh.write(f"{F[k]:.2f} {abs(R[0,k]):.5f} {abs(Rp[0,k]):.5f} {abs(R[1,k]):.5f} {abs(Rp[1,k]):.5f} "
                 f"{R[0,k].real:.5f} {R[0,k].imag:.5f} {Rp[0,k].real:.5f} {Rp[0,k].imag:.5f} "
                 f"{R[1,k].real:.5f} {R[1,k].imag:.5f} {Rp[1,k].real:.5f} {Rp[1,k].imag:.5f}\n")
(OUT / "layout_bits.tex").write_text(r"\def\layoutbits{" + ",".join(f"{r}/{c}" for r, c in zip(*np.nonzero(x_c))) + "}\n")
json.dump(dict(test_index=i, gamma=[float(m.gama) for st in bb.pga_stacks for m in st]),
          open(OUT / "architecture.json", "w"), indent=1)
print("architecture example: test layout", i)

# ------------------------------------------------------------------ method (VAE decodes, CP design, CP fronts)
vae = load_vae(os.path.join(C.vae_dir("rdn_d200"), "vae_best.pth"))
gen = torch.Generator(device=dev).manual_seed(3)
z0, z1 = torch.randn(200, device=dev, generator=gen), torch.randn(200, device=dev, generator=gen)
with torch.no_grad():
    for j, t in enumerate(np.linspace(0, 1, 6)):
        png(f"vae_{j}", layout_image(vae.decode(((1 - t) * z0 + t * z1)[None])[0, 0].cpu().numpy() > 0.5))
DEVD = ROOT / "experiments" / "devices" / "verify_cp"
zv = np.load(DEVD / "spectra.npz")
ob = DeviceObjectives("cp", F, search=False)
Vf, _ = ob.values(zv["extrap"], zv["X"].astype(np.uint8))
best = int(np.argmax(np.where(ob.good(Vf), Vf[:, 0], -1)))       # widest FDTD band among designs meeting the target
png("best_layout", layout_image(zv["X"][best]))
pred = zv["pred"][best]
ar = axial_ratio_db(pred[None, 0], pred[None, 1])[0][0]
np.savetxt(OUT / "design.dat", np.c_[F, np.abs(pred[0]), np.abs(pred[1]), np.minimum(ar, 12)], fmt="%.4f",
           header="f rxx ryx ar", comments="")
# dataset cloud and front, search archive, CP objectives (B, AR, D)
im, cu = load_class("RDN")
Xd = to_convention(im, "RDN", "true", np.arange(len(im))).astype(np.uint8); Yd = cu[:, :2]
Pd, Vd = ob(Yd, Xd)
ok = np.where((Vd[:, 0] > 0) & ((np.abs(Yd) ** 2).sum(1).max(1) <= 1.001))[0]
nd = ok[non_dominated(Pd[ok])]
rng = np.random.default_rng(0)
cloud = rng.choice(np.setdiff1d(ok, nd), 1200, replace=False)
for name, sel in (("front", nd), ("cloud", cloud)):
    np.savetxt(OUT / f"{name}.dat", Vd[sel][:, :3], fmt="%.4f", header="B AR D", comments="")
za = np.load(os.path.join(C.mopso_dir("cp", "pix_long"), "Final", "pareto_designs.npz"))
_, Va = ob(za["R"], za["patterns"])
m = Va[:, 0] > 0
np.savetxt(OUT / "archive.dat", Va[m][:, :3], fmt="%.4f", header="B AR D", comments="")
print("method data written:", len(nd), "dataset-front designs,", int(m.sum()), "archive designs")
