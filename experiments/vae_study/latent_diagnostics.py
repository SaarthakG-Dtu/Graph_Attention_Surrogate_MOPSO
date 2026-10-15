"""
latent_diagnostics.py — is the VAE latent space close to N(0, I), and what does
the decoder produce inside the MOIPSO box [-B, B]^d?

  * aggregate posterior of the RDN test layouts: per-dimension mean and spread
    of the encoder means, and the mean posterior std;
  * decodes of z ~ N(0, I), of z ~ U[-B, B]^d and of z scaled by s: copper
    fraction, fragile features and how far they are from RDN layouts;
  * latent sensitivity: how many pixels change for a random step of length r.

    python experiments/vae_study/latent_diagnostics.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import config as C                                   # noqa: E402
from dataset import get_split_arrays                 # noqa: E402
from phase2_vae import load_vae                      # noqa: E402
from phase2_mopso import fragile_count               # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
rng = np.random.default_rng(0)
vae = load_vae(os.path.join(C.vae_dir(os.environ.get("VAE_TAG", C.MOPSO_VAE_TAG)), "vae_best.pth"))
d = vae.latent_dim
x, _, _, _ = get_split_arrays("RDN", "test")
xt = torch.as_tensor(x[:, None], dtype=torch.float32, device=C.DEVICE)
with torch.no_grad():
    mu, logvar = vae.encoder(xt)
mu, sig = mu.cpu().numpy(), np.exp(0.5 * logvar.cpu().numpy())

res = {}
agg_std = np.sqrt(mu.var(0) + (sig ** 2).mean(0))
res["aggregate posterior"] = dict(
    mean_of_mu=[float(np.abs(mu.mean(0)).mean()), float(np.abs(mu.mean(0)).max())],
    std_of_mu_per_dim=[float(np.percentile(mu.std(0), q)) for q in (5, 50, 95)],
    posterior_sigma_per_dim=[float(np.percentile(sig.mean(0), q)) for q in (5, 50, 95)],
    aggregate_std_per_dim=[float(np.percentile(agg_std, q)) for q in (5, 50, 95)],
    frac_mu_outside_2=float((np.abs(mu) > 2).mean()), frac_mu_outside_3=float((np.abs(mu) > 3).mean()),
    norm_of_mu=[float(np.percentile(np.linalg.norm(mu, axis=1), q)) for q in (5, 50, 95)],
    norm_expected_for_N01=float(np.sqrt(d)))


def decode(z):
    with torch.no_grad():
        return vae.decode(torch.as_tensor(z, dtype=torch.float32, device=C.DEVICE)).cpu().numpy()[:, 0]


def stats(layouts):
    fill = layouts.reshape(len(layouts), -1).mean(1)
    D = fragile_count(layouts.astype(np.uint8))
    return dict(fill=[float(fill.mean()), float(fill.std())], fragile=[float(D.mean()), float(D.min()), float(np.percentile(D, 5))],
                unique=int(len(np.unique(layouts.reshape(len(layouts), -1), axis=0))))


n = 3000
res["RDN test layouts"] = stats(x)
res["decoded encoder means"] = stats(decode(mu))
res["z ~ N(0,I)"] = stats(decode(rng.standard_normal((n, d))))
for s in (0.5, 1.5, 2.0):
    res[f"z ~ N(0,{s}^2 I)"] = stats(decode(s * rng.standard_normal((n, d))))
for B in (1.0, 2.0, 3.0):
    res[f"z ~ U[-{B},{B}]^d"] = stats(decode(rng.uniform(-B, B, (n, d))))

# sensitivity: pixels changed by a random latent step of length r around encoded layouts
z0 = mu[:500]
x0 = decode(z0)
sens = {}
for r in (0.1, 0.3, 0.5, 1.0, 2.0):
    step = rng.standard_normal(z0.shape)
    step *= r / np.linalg.norm(step, axis=1, keepdims=True)
    changed = (decode(z0 + step) != x0).reshape(len(z0), -1).sum(1)
    sens[f"|dz|={r}"] = dict(mean_pixels_changed=float(changed.mean()), frac_no_change=float((changed == 0).mean()))
# a single PSO coordinate step (max velocity 0.5 in every dimension) has length 0.5*sqrt(d)
res["sensitivity"] = sens
res["note"] = f"max PSO step = 0.5 per dimension -> length up to {0.5 * np.sqrt(d):.1f}"
json.dump(res, open(os.path.join(OUT, "latent_diagnostics_%s.json" % os.environ.get("VAE_TAG", C.MOPSO_VAE_TAG)), "w"), indent=1)
print(json.dumps(res, indent=1))
