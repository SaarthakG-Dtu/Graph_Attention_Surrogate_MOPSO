"""
phase2_devices.py — objectives of the polarization devices, from the complex
co- and cross-polarized reflection (r_xx, r_yx) on the 101-point 2-12 GHz grid.

    cp       LP-to-CP converter: B = 3-dB axial-ratio band containing f0, quality =
             mean axial ratio in it (dB), D; constraint: lowest reflected power in
             the band L >= -0.5 dB.
    rotator  90-degree polarization rotator: B = band of PCR = |r_yx|^2 / (|r_xx|^2 +
             |r_yx|^2) > 0.9 containing f0, quality = mean |r_yx|^2 in it, D.

D counts corner contacts and isolated pixels (phase2_mopso.fragile_count).
During the search the band is scored symmetrically about f0,
B_sym = 2 min(f0 - f_lo, f_hi - f0) / f0, with a threshold margin (AR < 2.5 dB,
PCR > 0.93) against the surrogate's error; reported values use the standard
definitions (search=False). Penalties in [0, 1] (0 = ideal); a design whose f0 is
outside the band gets an extra penalty on all three, which guides the swarm.
Adapted from the objectives of sutd-prcm-solverfree (objectives.py).
"""

import numpy as np

_EPS = 1e-12

DEVICES = {
    #          f0 (GHz), threshold (search, report), objective names
    "cp":      dict(f0=10.0, thr=(2.5, 3.0), names=["Axial-ratio bandwidth (%)", "Mean axial ratio (dB)",
                                                    "Corner contacts + isolated pixels", "Lowest reflected power (dB)"],
                    ideal=[30, 0, 0, 0], worst=[0, 3, 60, -20], spec="B >= 10 %, L >= -0.5 dB, D <= 5"),
    "rotator": dict(f0=10.2, thr=(0.93, 0.9), names=["PCR > 0.9 bandwidth (%)", "Mean |r_yx|^2",
                                                    "Corner contacts + isolated pixels"],
                    ideal=[30, 1, 0], worst=[0, 0, 60], spec="B >= 5 %, mean |r_yx|^2 >= 0.8, D <= 5"),
}


def axial_ratio_db(rxx, ryx):
    """Axial ratio (dB) of the reflected wave from the Stokes parameters, and S0."""
    S0 = np.abs(rxx) ** 2 + np.abs(ryx) ** 2
    S3 = 2 * np.imag(np.conj(rxx) * ryx)
    chi = 0.5 * np.arcsin(np.clip(np.abs(S3) / np.maximum(S0, _EPS), 0, 1))
    return np.minimum(20 * np.log10(1 / np.maximum(np.tan(chi), 1e-6)), 40.0), S0


def band_about(q, good, freq, i0, thr, centered):
    """Band of good points containing index i0, edges interpolated where the
    quantity q crosses thr. Returns (in band, f_lo, f_hi, B %, mask of the band)."""
    n, K = q.shape
    idx = np.arange(K)
    bad = ~good
    lb = np.where(bad & (idx <= i0), idx, -1).max(1)                 # last bad point at/below f0
    rb = np.where(bad & (idx >= i0), idx, K).min(1)                  # first bad point at/above f0
    inb = good[:, i0]
    lo, hi = lb + 1, rb - 1

    def cross(ib, ig):
        valid = (ib >= 0) & (ib <= K - 1)
        ib, ig = np.clip(ib, 0, K - 1), np.clip(ig, 0, K - 1)
        qb, qg = q[np.arange(n), ib], q[np.arange(n), ig]
        frac = np.clip((qb - thr) / np.where(np.abs(qb - qg) < 1e-9, 1e-9, qb - qg), 0, 1)
        return np.where(valid, freq[ib] + (freq[ig] - freq[ib]) * frac, freq[ig])

    f_lo, f_hi = cross(lb, lo), cross(rb, hi)
    f0 = freq[i0]
    B = (200 * np.minimum(f0 - f_lo, f_hi - f0) / f0) if centered else (200 * (f_hi - f_lo) / (f_hi + f_lo))
    B = np.where(inb, B, 0.0)
    mask = (idx[None] >= lo[:, None]) & (idx[None] <= hi[:, None]) & inb[:, None]
    return inb, np.where(inb, f_lo, np.nan), np.where(inb, f_hi, np.nan), B, mask


class DeviceObjectives:
    """Drop-in for phase2_mopso.Objectives: __call__(R, mats) -> (penalties, values),
    R complex (N, 2, K) = (r_xx, r_yx)."""

    n_pen = 3

    def __init__(self, device, freq, search=True):
        from phase2_mopso import fragile_count
        self.device, self.search, self.freq = device, search, np.asarray(freq)
        d = DEVICES[device]
        self.f0, self.thr = d["f0"], d["thr"][0 if search else 1]
        self.i0 = int(np.argmin(np.abs(self.freq - self.f0)))
        self.names, self.spec = d["names"], d["spec"]
        self.ideal, self.worst = np.array(d["ideal"], float), np.array(d["worst"], float)
        self.specs = [dict(name=n, f0=self.f0) for n in self.names]
        self._fragile = fragile_count

    def band(self, R):
        """(in band, f_lo, f_hi, B, mask, band quantity) of the device band."""
        rxx, ryx = R[:, 0], R[:, 1]
        if self.device == "cp":
            q, _ = axial_ratio_db(rxx, ryx)
            good = q < self.thr
        else:
            q = np.abs(ryx) ** 2 / np.maximum(np.abs(rxx) ** 2 + np.abs(ryx) ** 2, _EPS)
            good = q > self.thr
        return band_about(q, good, self.freq, self.i0, self.thr, self.search) + (q,)

    def values(self, R, mats):
        inb, f_lo, f_hi, B, m, q = self.band(R)
        D = self._fragile(mats)
        n = np.maximum(m.sum(1), 1)
        if self.device == "cp":
            S0 = np.abs(R[:, 0]) ** 2 + np.abs(R[:, 1]) ** 2
            qual = np.where(inb, (q * m).sum(1) / n, q[:, self.i0])
            L = np.where(inb, np.where(m, 10 * np.log10(np.maximum(S0, _EPS)), np.inf).min(1),
                         10 * np.log10(np.maximum(S0[:, self.i0], _EPS)))
            V = np.stack([B, qual, D, L], 1)
            viol = np.where(inb, 0.0, (q[:, self.i0] - self.thr) / 20) + np.maximum(-0.5 - L, 0) / 3
        else:
            qual = np.where(inb, ((np.abs(R[:, 1]) ** 2) * m).sum(1) / n, q[:, self.i0])
            V = np.stack([B, qual, D], 1)
            viol = np.where(inb, 0.0, self.thr - q[:, self.i0])
        return V, viol

    def penalties(self, V, viol):
        P = (V[:, :3] - self.ideal[:3]) / (self.worst[:3] - self.ideal[:3])
        return np.clip(P, 0, 1) + viol[:, None]

    def __call__(self, R, mats):
        V, viol = self.values(R, mats)
        return self.penalties(V, viol), V

    def good(self, V):
        if self.device == "cp":
            return (V[:, 0] >= 10) & (V[:, 3] >= -0.5) & (V[:, 2] <= 5)
        return (V[:, 0] >= 5) & (V[:, 1] >= 0.8) & (V[:, 2] <= 5)


# ─────────────────────────────────────────────────────────────────────────────
# 1-bit coding checkerboard of two cells A, B (tiles of N x N identical cells)
# ─────────────────────────────────────────────────────────────────────────────
# Specular reflectance relative to a metal plate, the area average of the two cells:
#   s_x = (|A_xx + B_xx|^2 + |A_yx + B_yx|^2) / 4, s_y the same with the transposed
#   layouts (r_yy(L) = r_xx(L^T)); s = max(s_x, s_y), so the reduction holds for both
#   polarizations. Structures: "free" (A, B independent), "mirror" (B = mirror(A):
#   r_yx changes sign, so the cross-polarized parts cancel exactly), "diag" (mirror,
#   with A symmetric about its diagonal, so s_x = s_y).
#   B  widest band with reduction -10 log10 s above 10 dB (search: 11 dB), %
#   R  mean reduction in it (capped at 30 dB per frequency); D of A plus D of B.
# Penalties (80 - B)/80, (25 - R)/25, D/120; extra penalty if there is no band.

CODING = {"coding_diag": "diag", "coding_mirror": "mirror", "coding_free": "free"}


def diag_layout(z):
    """(N, 136) positions in [0, 1] -> (N, 16, 16) layouts symmetric about the diagonal."""
    iu = np.triu_indices(16)
    A = np.zeros((len(z), 16, 16), np.float32)
    A[:, iu[0], iu[1]] = z
    return (np.maximum(A, A.transpose(0, 2, 1)) > 0.5).astype(np.uint8)


def mirror(L):
    """y -> -y (image columns); keeps r_xx, reverses r_yx."""
    return L[..., ::-1]


def specular(SA, SB, SAT, SBT):
    sx = (np.abs(SA + SB) ** 2).sum(1) / 4
    sy = (np.abs(SAT + SBT) ** 2).sum(1) / 4
    return sx, sy


def widest_band(red, T, freq):
    """Widest run of red > T: (lo, hi) indices, edges interpolated (GHz), run length."""
    n, K = red.shape
    good = red > T
    run, best, end = np.zeros(n), np.zeros(n), np.zeros(n, int)
    for k in range(K):
        run = (run + 1) * good[:, k]
        upd = run > best
        best, end = np.where(upd, run, best), np.where(upd, k, end)
    hi, lo = end, end - best.astype(int) + 1

    def edge(io, ii):
        valid = (io >= 0) & (io <= K - 1)
        io, ii = np.clip(io, 0, K - 1), np.clip(ii, 0, K - 1)
        ro, ri = red[np.arange(n), io], red[np.arange(n), ii]
        frac = np.clip((T - ro) / np.maximum(ri - ro, 1e-9), 0, 1)
        return np.where(valid, freq[io] + (freq[ii] - freq[io]) * frac, freq[ii])

    return lo, hi, edge(lo - 1, lo), edge(hi + 1, hi), best


class CodingObjectives:
    """Coding pair; layouts are pairs (N, 2, 16, 16) = (A, B). evaluate(sur, pairs)
    -> (spectra (N, 4, 2, K) of A, B, A^T, B^T, penalties, values)."""

    names = ["-10 dB reduction bandwidth (%)", "Mean reduction (dB)", "Corner contacts + isolated pixels"]
    ideal, worst = np.array([80, 25, 0], float), np.array([0, 0, 120], float)
    spec = "widest -10 dB band >= 40 %, D <= 10 (pair)"
    device = "coding"
    n_pen = 3

    def __init__(self, structure, freq, search=True):
        from phase2_mopso import fragile_count
        self.structure, self.freq, self.search = structure, np.asarray(freq), search
        self.thr = 11.0 if search else 10.0
        self.specs = [dict(name=n, f0=10.0) for n in self.names]
        self._fragile = fragile_count

    def spectra(self, sur, pairs):
        A, B = pairs[:, 0], pairs[:, 1]
        AT = A.transpose(0, 2, 1)
        flip = lambda S: np.stack([S[:, 0], -S[:, 1]], 1)            # spectra of the mirror image
        if self.structure == "free":
            SA, SB, SAT, SBT = np.split(sur.predict(np.concatenate([A, B, AT, B.transpose(0, 2, 1)])), 4)
        elif self.structure == "mirror":
            SA, SAT = np.split(sur.predict(np.concatenate([A, AT])), 2)
            SB, SBT = flip(SA), flip(SAT)
        else:                                                          # diag: A = A^T
            SA = sur.predict(A); SAT = SA
            SB, SBT = flip(SA), flip(SA)
        return np.stack([SA, SB, SAT, SBT], 1)

    def reduction(self, S):
        sx, sy = specular(S[:, 0], S[:, 1], S[:, 2], S[:, 3])
        return -10 * np.log10(np.maximum(np.maximum(sx, sy), 1e-3))

    def band(self, S):
        red = self.reduction(S)
        lo, hi, f_lo, f_hi, n = widest_band(red, self.thr, self.freq)
        has = n > 0
        return has, np.where(has, f_lo, np.nan), np.where(has, f_hi, np.nan), lo, hi, red

    def values(self, S, pairs):
        has, f_lo, f_hi, lo, hi, red = self.band(S)
        Bw = np.where(has, 200 * (f_hi - f_lo) / (f_hi + f_lo), 0.0)
        idx = np.arange(red.shape[1])
        m = (idx[None] >= lo[:, None]) & (idx[None] <= hi[:, None]) & has[:, None]
        R = np.where(has, (red * m).sum(1) / np.maximum(m.sum(1), 1), red.max(1))
        D = self._fragile(pairs[:, 0]) + self._fragile(pairs[:, 1])
        viol = np.where(has, 0.0, np.maximum(self.thr - red.max(1), 0) / self.thr)
        return np.stack([np.nan_to_num(Bw), R, D], 1), viol

    def evaluate(self, sur, pairs):
        S = self.spectra(sur, pairs)
        V, viol = self.values(S, pairs)
        P = np.clip((V - self.ideal) / (self.worst - self.ideal), 0, 1) + viol[:, None]
        return S, P, V

    def good(self, V):
        return (V[:, 0] >= 40) & (V[:, 2] <= 10)
