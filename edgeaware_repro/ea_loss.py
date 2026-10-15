"""
ea_loss.py — physics-guided objective of the edge-aware paper, eq. (5):

    L = L_reim + 0.5 L_mag,dB + 0.3 L_mag,lin + 0.3 L_phase
        + 0.08 L_pass + 0.06 L_tv + 0.02 L_curv + 0.02 L_KK

with edge samples (first and last 10 % of the K frequencies) weighted 4x in
the per-frequency terms (reim, mag_dB, mag_lin, phase, pass). L_KK is the
discrete Kramers-Kronig loss of eq. (3)-(4); L_phase = |[cos, sin](phi_hat) -
[cos, sin](phi)|^2 (unit-circle phase, no 2 pi wrapping).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W = dict(reim=1.0, mag_db=0.5, mag_lin=0.3, phase=0.3, pass_=0.08, tv=0.06, curv=0.02, kk=0.02)
EDGE_FRAC, EDGE_WEIGHT = 0.10, 4.0
_EPS = 1e-7


def kk_kernel(freq):
    """M[k, j] = dw / (pi (w_j - w_k)), j != k (eq. (3), trapezoidal rule)."""
    f = np.asarray(freq, np.float64)
    dw = 2 * math.pi * (f[1] - f[0])
    diff = 2 * math.pi * (f[None, :] - f[:, None])
    np.fill_diagonal(diff, np.inf)
    return torch.from_numpy((dw / (math.pi * diff)).astype(np.float32))


def edge_weights(K):
    e = max(1, int(round(K * EDGE_FRAC)))
    w = torch.ones(K)
    w[:e] = EDGE_WEIGHT
    w[K - e:] = EDGE_WEIGHT
    return w / w.mean()                      # weighted mean keeps the scale of each term


class EdgeAwareLoss(nn.Module):
    def __init__(self, freq):
        super().__init__()
        self.register_buffer("M", kk_kernel(freq))
        self.register_buffer("w", edge_weights(len(freq)))

    def forward(self, pred, tgt):
        w = self.w
        mp = torch.sqrt((pred ** 2).sum(-1) + _EPS)
        mt = torch.sqrt((tgt ** 2).sum(-1) + _EPS)
        r, x = pred[..., 0], pred[..., 1]
        terms = dict(
            reim=(((pred - tgt) ** 2).sum(-1) / 2 * w).mean(),
            mag_db=(((20 * torch.log10(mp) - 20 * torch.log10(mt)) ** 2) * w).mean(),
            mag_lin=(((mp - mt) ** 2) * w).mean(),
            phase=((((pred / mp.unsqueeze(-1)) - (tgt / mt.unsqueeze(-1))) ** 2).sum(-1) * w).mean(),
            pass_=((F.relu(mp - 1.0) ** 2) * w).mean(),
            tv=(pred[:, 1:] - pred[:, :-1]).abs().mean(),
            curv=((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) ** 2).mean(),
            kk=((r - x @ self.M.T) ** 2 + (x + r @ self.M.T) ** 2).mean(),
        )
        total = sum(W[k] * v for k, v in terms.items())
        return total, {k: v.item() for k, v in terms.items()}
