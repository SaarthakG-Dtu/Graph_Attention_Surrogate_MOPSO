"""
phase2_vae.py — 16x16 layout VAE (port of the 12x12 conv_symmetric VAE).

    python phase2_vae.py --mode both                       # train + diagnose
    python phase2_vae.py --mode both --classes RDN --tag rdn   # control on random layouts

Architecture (conv_symmetric):
    encoder  Conv3x3 s1 (64) -> Conv3x3 s2 (128, 8x8) -> Conv3x3 s2 (256, 4x4) -> mu, logvar
    decoder  Linear -> 256x4x4 -> ConvT s2 (128) -> ConvT s2 (64) -> Conv3x3 -> logit
Loss    : W_BCE * BCE(mean) + beta * KL(mean over dims) + GAMMA * mean p(1-p);
          beta = 0 for KL_FREE_EPOCHS, then a cosine ramp over KL_ANNEAL_EPOCHS;
          checkpoints are only taken after the ramp; early stopping.
Data    : the Phase 1 train / val / test layouts of the chosen classes.

Outputs in config.vae_dir(tag): vae_best.pth, vae_config.json, vae_history.json,
vae_training.png, vae_diagnostics.json, vae_reconstructions.png,
vae_prior_samples.png, vae_interpolation.png, vae_pattern_statistics.png
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from dataset import get_split_arrays
import utils

IMG = C.IMG_SIZE


def _activate(logit, act, st_m):
    if act == "sigmoid":
        return torch.sigmoid(logit)
    if act == "st":
        return 0.5 * (1.0 + torch.tanh((logit - 0.5) * st_m))
    raise ValueError(f"final activation must be 'sigmoid' or 'st', got {act!r}")


class ConvEncoderSym(nn.Module):
    def __init__(self, latent_dim, ch):
        super().__init__()
        c0, c1, c2 = ch
        self.stem = nn.Conv2d(1, c0, 3, 1, 1); self.bn0 = nn.BatchNorm2d(c0)
        self.down1 = nn.Conv2d(c0, c1, 3, 2, 1); self.bn1 = nn.BatchNorm2d(c1)
        self.down2 = nn.Conv2d(c1, c2, 3, 2, 1); self.bn2 = nn.BatchNorm2d(c2)
        self.fc_mu = nn.Linear(c2 * 16, latent_dim)
        self.fc_logvar = nn.Linear(c2 * 16, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.stem(x)))
        x = F.leaky_relu(self.bn1(self.down1(x)))
        x = F.leaky_relu(self.bn2(self.down2(x))).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)


class ConvDecoderSym(nn.Module):
    def __init__(self, latent_dim, ch):
        super().__init__()
        c0, c1, c2 = ch
        self.c2 = c2
        self.fc = nn.Linear(latent_dim, c2 * 16)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, 2, 1); self.bn1 = nn.BatchNorm2d(c1)
        self.up2 = nn.ConvTranspose2d(c1, c0, 4, 2, 1); self.bn2 = nn.BatchNorm2d(c0)
        self.head = nn.Conv2d(c0, 1, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z).view(-1, self.c2, 4, 4)
        x = F.leaky_relu(self.bn1(self.up1(x)))
        x = F.leaky_relu(self.bn2(self.up2(x)))
        return self.head(x)                                          # logit (B,1,16,16)


class VAE16(nn.Module):
    def __init__(self, latent_dim=C.VAE_LATENT_DIM, ch=C.VAE_CONV_CHANNELS, act=C.VAE_FINAL_ACTIVATION):
        super().__init__()
        self.latent_dim, self.act = latent_dim, act
        self.encoder = ConvEncoderSym(latent_dim, ch)
        self.decoder = ConvDecoderSym(latent_dim, ch)

    def forward(self, x, st_m=C.VAE_ST_M[1]):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        logit = self.decoder(z)
        return _activate(logit, self.act, st_m), logit, mu, logvar

    @torch.no_grad()
    def decode(self, z, binarize=True):
        p = _activate(self.decoder(z), self.act, C.VAE_ST_M[1])
        return (p > 0.5).float() if binarize else p

    @torch.no_grad()
    def reconstruct(self, x, binarize=True):
        return self.decode(self.encoder(x)[0], binarize)


def vae_loss(recon, logit, x, mu, logvar, beta):
    """beta is the current (annealed) KL weight."""
    bce = F.binary_cross_entropy_with_logits(logit, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    binr = (recon * (1.0 - recon)).mean()
    return C.VAE_W_BCE * bce + beta * kl + C.VAE_GAMMA * binr, bce, kl


def beta_at(ep, beta_max=C.VAE_BETA):
    if ep <= C.VAE_KL_FREE_EPOCHS:
        return 0.0
    frac = min(1.0, (ep - C.VAE_KL_FREE_EPOCHS) / max(C.VAE_KL_ANNEAL_EPOCHS, 1))
    if C.VAE_KL_ANNEAL_SCHEDULE == "linear":
        return beta_max * frac
    return beta_max * 0.5 * (1.0 - math.cos(math.pi * frac))


def st_m_at(ep):
    m0, m1, n = C.VAE_ST_M
    return m0 + (m1 - m0) * min(1.0, ep / max(n, 1))


def load_vae(path, device=C.DEVICE):
    rc = json.load(open(os.path.join(os.path.dirname(path), "vae_config.json")))
    m = VAE16(rc["latent_dim"], rc["conv_channels"], rc["final_activation"]).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def _augment(x):
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    if torch.rand(1) < 0.5:
        x = x.flip(-2)
    return x


def layouts(classes, split):
    xs, labels = [], []
    for c in classes:
        x, _, lab, _ = get_split_arrays(c, split, convention=C.VAE_CONVENTION)
        xs.append(x); labels.append(lab)
    return np.concatenate(xs).astype(np.float32), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _epoch(model, X, beta, st_m, train, opt=None):
    model.train(train)
    perm = torch.randperm(len(X), device=X.device) if train else torch.arange(len(X), device=X.device)
    tot = np.zeros(4); n = 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), C.VAE_BATCH_SIZE):
            x = X[perm[i:i + C.VAE_BATCH_SIZE]]
            if train and C.VAE_AUGMENT:
                x = _augment(x)
            recon, logit, mu, logvar = model(x, st_m)
            loss, bce, kl = vae_loss(recon, logit, x, mu, logvar, beta)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.VAE_GRAD_CLIP)
                opt.step()
            acc = ((recon > 0.5).float() == x).float().mean()
            tot += np.array([loss.item(), bce.item(), kl.item(), acc.item()]) * len(x)
            n += len(x)
    return dict(zip(("tot", "bce", "kl", "acc"), tot / n))


def train(args):
    out = C.vae_dir(args.tag)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(C.RANDOM_SEED)
    xtr, _ = layouts(args.classes, "train")
    xva, _ = layouts(args.classes, "val")
    Xtr = torch.from_numpy(xtr).unsqueeze(1).to(C.DEVICE)
    Xva = torch.from_numpy(xva).unsqueeze(1).to(C.DEVICE)
    model = VAE16(args.latent_dim, list(args.channels)).to(C.DEVICE)
    cfg = dict(classes=args.classes, latent_dim=args.latent_dim, conv_channels=list(args.channels),
               final_activation=C.VAE_FINAL_ACTIVATION, w_bce=C.VAE_W_BCE, beta=args.beta,
               gamma=C.VAE_GAMMA, kl_free=C.VAE_KL_FREE_EPOCHS, kl_anneal=C.VAE_KL_ANNEAL_EPOCHS,
               augment=C.VAE_AUGMENT, n_train=len(xtr), n_val=len(xva),
               n_params=sum(p.numel() for p in model.parameters()))
    json.dump(cfg, open(os.path.join(out, "vae_config.json"), "w"), indent=2)
    opt = torch.optim.Adam(model.parameters(), lr=C.VAE_LR, weight_decay=C.VAE_WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=C.VAE_LR_MIN)
    print(f"VAE16 [{','.join(args.classes)}] latent={args.latent_dim} params={cfg['n_params']:,} "
          f"train={len(xtr)} val={len(xva)}  loss={C.VAE_W_BCE}*BCE + beta*KL (beta->{args.beta})")
    anneal_end = C.VAE_KL_FREE_EPOCHS + C.VAE_KL_ANNEAL_EPOCHS
    best, bad, hist = float("inf"), 0, []
    for ep in range(1, args.epochs + 1):
        t0, beta, m = time.time(), beta_at(ep, args.beta), st_m_at(ep)
        tr = _epoch(model, Xtr, beta, m, True, opt)
        with torch.no_grad():
            va = _epoch(model, Xva, beta, m, False)
        sch.step()
        hist.append(dict(ep=ep, beta=beta, **{f"tr_{k}": v for k, v in tr.items()},
                         **{f"va_{k}": v for k, v in va.items()}))
        mark = ""
        if ep > anneal_end:
            if va["tot"] < best:
                best, bad, mark = va["tot"], 0, " *"
                torch.save(model.state_dict(), os.path.join(out, "vae_best.pth"))
            else:
                bad += 1
        print(f"  ep {ep:3d} beta={beta:.3f} tr bce={tr['bce']:.4f} kl={tr['kl']:.4f} acc={tr['acc']:.4f} | "
              f"va tot={va['tot']:.4f} bce={va['bce']:.4f} kl={va['kl']:.4f} acc={va['acc']:.4f} "
              f"({time.time() - t0:.0f}s){mark}", flush=True)
        if bad >= C.VAE_EARLY_STOP:
            print(f"  early stop at epoch {ep}")
            break
    if best == float("inf"):
        torch.save(model.state_dict(), os.path.join(out, "vae_best.pth"))
        print("  [warning] training ended before beta finished ramping; saved last weights")
    json.dump(hist, open(os.path.join(out, "vae_history.json"), "w"), indent=1)
    _plot_history(hist, anneal_end, os.path.join(out, "vae_training.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def neighbour_agreement(x):
    """Fraction of 4-adjacent pixel pairs that are equal (0.5 = i.i.d. random)."""
    h = (x[:, :, 1:] == x[:, :, :-1]).mean((1, 2))
    v = (x[:, 1:, :] == x[:, :-1, :]).mean((1, 2))
    return (h + v) / 2


def n_islands(x):
    from scipy.ndimage import label
    return np.array([label(a)[1] for a in x])


def pattern_stats(x):
    sub = x[np.random.default_rng(0).choice(len(x), min(2000, len(x)), replace=False)]
    return dict(fill_mean=float(x.mean()), fill_std=float(x.mean((1, 2)).std()),
                neighbour_agreement=float(neighbour_agreement(x).mean()),
                islands_mean=float(n_islands(sub).mean()))


@torch.no_grad()
def diagnose(args):
    out = C.vae_dir(args.tag)
    model = load_vae(os.path.join(out, "vae_best.pth"))
    cfg = json.load(open(os.path.join(out, "vae_config.json")))
    classes, d = cfg["classes"], cfg["latent_dim"]
    xtr, _ = layouts(classes, "train")
    xte, lte = layouts(classes, "test")
    X = torch.from_numpy(xte).unsqueeze(1).to(C.DEVICE)
    mu, logvar = zip(*[model.encoder(X[i:i + 2048]) for i in range(0, len(X), 2048)])
    mu, logvar = torch.cat(mu), torch.cat(logvar)
    rec = torch.cat([model.decode(mu[i:i + 2048]) for i in range(0, len(mu), 2048)]).squeeze(1).cpu().numpy()
    pix = (rec == xte).mean((1, 2))
    kl_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(0)).cpu().numpy()
    z = torch.randn(5000, d, device=C.DEVICE, generator=torch.Generator(device=C.DEVICE).manual_seed(0))
    samp = model.decode(z).squeeze(1).cpu().numpy()
    Ttr = torch.from_numpy(xtr.reshape(len(xtr), -1)).to(C.DEVICE)
    dmin = torch.cdist(torch.from_numpy(samp[:500].reshape(500, -1)).to(C.DEVICE), Ttr, p=1).min(1).values.cpu().numpy()
    rep = dict(classes=classes, latent_dim=d, n_test=len(xte),
               pixel_accuracy_mean=float(pix.mean()), pixel_accuracy_std=float(pix.std()),
               exact_reconstruction_frac=float((pix == 1).mean()),
               pixel_accuracy_per_class={c: float(pix[lte == c].mean()) for c in classes},
               exact_reconstruction_per_class={c: float((pix[lte == c] == 1).mean()) for c in classes},
               active_latent_dims=int((kl_dim > 0.01).sum()),
               prior_samples_unique_frac=float(len(np.unique(samp.reshape(len(samp), -1), axis=0)) / len(samp)),
               prior_min_hamming_to_train_median=float(np.median(dmin)),
               prior_copies_of_train_frac=float((dmin == 0).mean()),
               stats_data=pattern_stats(xte), stats_reconstruction=pattern_stats(rec),
               stats_prior_samples=pattern_stats(samp))
    json.dump(rep, open(os.path.join(out, "vae_diagnostics.json"), "w"), indent=2)
    sd, sp = rep["stats_data"], rep["stats_prior_samples"]
    print(f"  pixel accuracy {rep['pixel_accuracy_mean']:.4f}  exact {100 * rep['exact_reconstruction_frac']:.1f}%  "
          f"active dims {rep['active_latent_dims']}/{d}  per class {rep['pixel_accuracy_per_class']}")
    print(f"  neighbour agreement data {sd['neighbour_agreement']:.3f} | prior {sp['neighbour_agreement']:.3f};  "
          f"fill data {sd['fill_mean']:.3f} | prior {sp['fill_mean']:.3f};  "
          f"islands data {sd['islands_mean']:.2f} | prior {sp['islands_mean']:.2f}")
    if abs(sp["neighbour_agreement"] - sd["neighbour_agreement"]) > 0.05:
        print("  NOTE: decoded designs are statistically unlike the training layouts -> they fall "
              "partly outside the surrogate's training distribution.")
    _plot_reconstructions(xte, rec, pix, lte, classes, os.path.join(out, "vae_reconstructions.png"))
    _plot_samples(samp[:32], os.path.join(out, "vae_prior_samples.png"))
    _plot_interpolation(model, mu, os.path.join(out, "vae_interpolation.png"))
    _plot_stats(xte, rec, samp, os.path.join(out, "vae_pattern_statistics.png"))
    return rep


def _grid(ax, img):
    ax.imshow(img, cmap=utils.PATTERN_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_edgecolor(utils.INK2)


def _plot_reconstructions(x, rec, pix, labels, classes, path, per_class=4):
    plt = utils.plt
    cols = per_class * len(classes)
    fig, axes = plt.subplots(2, cols, figsize=(utils.COL2, 2 * utils.COL2 / cols + 0.35), squeeze=False)
    k = 0
    for c in classes:
        ids = np.where(labels == c)[0][:per_class]
        for j, i in enumerate(ids):
            _grid(axes[0, k], x[i]); _grid(axes[1, k], rec[i])
            axes[0, k].set_title(c if j == 0 else "", fontsize=7, loc="left")
            axes[1, k].set_xlabel(f"{100 * pix[i]:.0f}%", fontsize=6)
            k += 1
    axes[0, 0].set_ylabel("input", fontsize=7); axes[1, 0].set_ylabel("decoded", fontsize=7)
    fig.tight_layout(w_pad=0.2, h_pad=0.2)
    utils.save(fig, path)


def _plot_samples(samp, path):
    plt = utils.plt
    fig, axes = plt.subplots(4, 8, figsize=(utils.COL1, utils.COL1 * 0.5 + 0.1))
    for ax, s in zip(axes.flat, samp):
        _grid(ax, s)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    utils.save(fig, path)


def _plot_interpolation(model, mu, path, n=8, rows=3):
    plt = utils.plt
    fig, axes = plt.subplots(rows, n, figsize=(utils.COL1, rows * utils.COL1 / n + 0.1))
    pick = np.random.default_rng(1).choice(len(mu), 2 * rows, replace=False)
    for r in range(rows):
        za, zb = mu[pick[2 * r]], mu[pick[2 * r + 1]]
        for k, a in enumerate(np.linspace(0, 1, n)):
            _grid(axes[r, k], model.decode(((1 - a) * za + a * zb)[None]).squeeze().cpu().numpy())
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    utils.save(fig, path)


def _plot_stats(data, rec, samp, path):
    plt = utils.plt
    fig, axes = plt.subplots(1, 2, figsize=(utils.COL2 * 0.7, 1.9))
    for arr, col, lab in ((data, utils.SERIES[1], "test layouts"), (rec, utils.SERIES[0], "reconstructions"),
                          (samp, utils.SERIES[2], r"prior samples $z\sim\mathcal{N}(0,I)$")):
        axes[0].hist(arr.mean((1, 2)), bins=40, range=(0, 1), density=True, histtype="step", color=col, lw=1.2, label=lab)
        axes[1].hist(neighbour_agreement(arr), bins=40, range=(0.3, 1), density=True, histtype="step", color=col, lw=1.2)
    axes[0].set_xlabel("fill fraction"); axes[1].set_xlabel("neighbour agreement")
    axes[0].legend(loc="upper left", fontsize=6)
    for ax in axes:
        ax.set_yticks([])
    fig.tight_layout()
    utils.save(fig, path)


def _plot_history(hist, anneal_end, path):
    plt = utils.plt
    ep = [h["ep"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(utils.COL2, 1.8))
    for ax, k, t in zip(axes, ("bce", "kl", "acc"), ("BCE (mean per pixel)", "KL (mean per dim)", "pixel accuracy")):
        ax.plot(ep, [h[f"tr_{k}"] for h in hist], color=utils.SERIES[0], label="train")
        ax.plot(ep, [h[f"va_{k}"] for h in hist], color=utils.SERIES[1], label="validation")
        ax.axvline(anneal_end, color=utils.INK2, lw=0.6, ls=":")
        ax.set_title(t); ax.set_xlabel("Epoch")
    axes[0].legend()
    fig.tight_layout()
    utils.save(fig, path)


if __name__ == "__main__":
    C.be_nice()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["train", "diagnose", "both"])
    ap.add_argument("--classes", nargs="+", default=C.VAE_TRAIN_CLASSES)
    ap.add_argument("--latent_dim", type=int, default=C.VAE_LATENT_DIM)
    ap.add_argument("--epochs", type=int, default=C.VAE_EPOCHS)
    ap.add_argument("--beta", type=float, default=C.VAE_BETA, help="final KL weight")
    ap.add_argument("--channels", type=int, nargs=3, default=C.VAE_CONV_CHANNELS, help="encoder widths, e.g. 32 64 128")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.mode in ("train", "both"):
        train(args)
    if args.mode in ("diagnose", "both"):
        diagnose(args)
