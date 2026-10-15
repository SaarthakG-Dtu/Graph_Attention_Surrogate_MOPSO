"""
phase1_loss.py — 8-term physics-guided loss and evaluation metrics.

All terms act on complex spectra stored as (B, K, 2) = [Re, Im]:
    reim      Cartesian MSE
    mag_db    MSE of 20 log10 |S|
    mag_lin   MSE of |S|
    phase     |unit(S_hat) - unit(S)|^2  (wrap-free)
    pass      mean max(0, |S_hat| - 1)^2
    tv        mean |S_hat[k+1] - S_hat[k]|
    curv      mean (S_hat[k+1] - 2 S_hat[k] + S_hat[k-1])^2
    kk        discrete Kramers-Kronig (Hilbert-pair) consistency of S_hat
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BAND_EDGE_FRAC,
    P1_W_REIM, P1_W_MAG_DB, P1_W_MAG_LIN, P1_W_PHASE,
    P1_W_PASS, P1_W_TV, P1_W_CURV, P1_W_KK,
)

_EPS = 1e-7
_KK_CACHE: dict = {}


def _kk_kernel(freq, device) -> torch.Tensor:
    """M[k, j] = dw / (pi (w_j - w_k)), j != k, so r_hat = M x and x_hat = -M r."""
    f = np.asarray(freq, dtype=np.float64)
    key = (f.tobytes(), str(device))
    if key not in _KK_CACHE:
        dw = 2 * math.pi * (f[1] - f[0])
        diff = 2 * math.pi * (f[None, :] - f[:, None])
        np.fill_diagonal(diff, np.inf)
        _KK_CACHE[key] = torch.from_numpy((dw / (math.pi * diff)).astype(np.float32)).to(device)
    return _KK_CACHE[key]


def _mag(c):
    return torch.sqrt(c[..., 0] ** 2 + c[..., 1] ** 2 + _EPS)


def physics_loss_terms(pred: torch.Tensor, tgt: torch.Tensor, freq,
                       db_floor=None, phase_mag_ref=None) -> dict:
    """db_floor (dB): clamp both magnitudes before the dB term.
    phase_mag_ref: weight the phase term by min(|S_target| / ref, 1), so the
    phase of near-zero responses (undefined) does not drive training.
    Both are off (None) for the co-pol target."""
    mp, mt = _mag(pred), _mag(tgt)
    M = _kk_kernel(freq, pred.device)
    r, x = pred[..., 0], pred[..., 1]
    dp, dt = 20 * torch.log10(mp), 20 * torch.log10(mt)
    if db_floor is not None:
        dp, dt = dp.clamp(min=db_floor), dt.clamp(min=db_floor)
    ph = ((pred / mp.unsqueeze(-1) - tgt / mt.unsqueeze(-1)) ** 2).sum(-1)
    if phase_mag_ref is not None:
        ph = ph * (mt / phase_mag_ref).clamp(max=1.0)
    return dict(
        reim=((pred - tgt) ** 2).mean(),
        mag_db=((dp - dt) ** 2).mean(),
        mag_lin=((mp - mt) ** 2).mean(),
        phase=ph.mean(),
        pass_=(F.relu(torch.sqrt(r ** 2 + x ** 2) - 1.0) ** 2).mean(),
        tv=(pred[:, 1:] - pred[:, :-1]).abs().mean(),
        curv=((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) ** 2).mean(),
        kk=((r - x @ M.T) ** 2 + (x + r @ M.T) ** 2).mean(),
    )


_WEIGHTS = dict(reim=P1_W_REIM, mag_db=P1_W_MAG_DB, mag_lin=P1_W_MAG_LIN, phase=P1_W_PHASE,
                pass_=P1_W_PASS, tv=P1_W_TV, curv=P1_W_CURV, kk=P1_W_KK)


class PhysicsGuidedLoss(nn.Module):
    """Returns (total, {term: float}); freq is the (K,) grid in GHz."""

    def __init__(self, freq, db_floor=None, phase_mag_ref=None):
        super().__init__()
        self.freq = np.asarray(freq)
        self.db_floor, self.phase_mag_ref = db_floor, phase_mag_ref

    def forward(self, pred, tgt):
        terms = physics_loss_terms(pred, tgt, self.freq, self.db_floor, self.phase_mag_ref)
        total = sum(_WEIGHTS[k] * v for k, v in terms.items())
        return total, {k: v.item() for k, v in terms.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (numpy)
# ─────────────────────────────────────────────────────────────────────────────

def magnitude_metrics(p: np.ndarray, t: np.ndarray, db_floor=None) -> dict:
    """|S| metrics; p, t: (N, K). R^2 uses the global mean (MSDCNN convention).
    db_floor clamps both dB magnitudes (used for the cross-polarised channel)."""
    d = p - t
    dp, dt = 20 * np.log10(p + _EPS), 20 * np.log10(t + _EPS)
    if db_floor is not None:
        dp, dt = np.maximum(dp, db_floor), np.maximum(dt, db_floor)
    db = dp - dt
    r2 = 1.0 - np.sum(d ** 2) / (np.sum((t - t.mean()) ** 2) + _EPS)
    return dict(lin_mse=float(np.mean(d ** 2)), db_mse=float(np.mean(db ** 2)), r2=float(r2),
                rmse=float(np.sqrt(np.mean(d ** 2))), mae=float(np.mean(np.abs(d))))


def complex_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """Phase / complex metrics; pred, tgt: (N, K, 2)."""
    cp = pred[..., 0] + 1j * pred[..., 1]
    ct = tgt[..., 0] + 1j * tgt[..., 1]
    dphi = np.abs(np.angle(cp * np.conj(ct)))                    # wrapped |delta phase|
    w = np.abs(ct)
    K = pred.shape[1]
    e = max(1, int(round(K * BAND_EDGE_FRAC)))
    edge = np.r_[0:e, K - e:K]
    db = 20 * np.log10(np.abs(cp) + _EPS) - 20 * np.log10(np.abs(ct) + _EPS)
    return dict(
        complex_mse=float(np.mean(np.abs(cp - ct) ** 2)),
        phase_mae_deg=float(np.degrees(dphi.mean())),
        phase_mae_deg_weighted=float(np.degrees((dphi * w).sum() / w.sum())),
        acc30=float(np.mean(dphi.mean(1) < np.radians(30))),
        band_edge_abs_db=float(np.abs(db[:, edge]).mean()),
    )
