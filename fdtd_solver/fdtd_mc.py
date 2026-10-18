"""FDTD unit cell for the manufacturing-tolerance (Monte Carlo) study: fdtd.Cell with
  * per-sample substrate permittivity, loss tangent and thickness (arrays of length B, one value per batch element),
  * the substrate thickness resolved by a stretched mesh in z (k_top cells of t_sub/k_top; air and PML keep h),
  * corner contacts opened at random (each contact independently with probability p_open),
  * optionally a ready cell-level copper mask (sub-pixel geometry, e.g. over- and under-etching on a fine mesh).
The production solver fdtd.py is left untouched; with scalar parameters this class reproduces it
(test_fdtd_mc.py). All copper is lossy as in fdtd.py (pixels as resistive sheets, ground as a surface impedance).
"""
import math
import numpy as np
import torch
from fdtd import C0, EPS0, MU0, ETA0, STOP_FILE, Cell as _Cell


class Cell(_Cell):
    def __init__(s, B, h=0.25e-3, period=10e-3, t_sub=1.0e-3, eps_r=2.65, tan_d=0.003, f_loss=10e9,
                 air_src=5e-3, air_mon=2e-3, air_gap=2e-3, n_pml=12, dev="cuda", copper_rs=0.026, ground_rs=None):
        arr = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (B,)).copy()
        t_sub, eps_r, tan_d = arr(t_sub), arr(eps_r), arr(tan_d)
        s.B, s.h, s.dev, s.n_pml, s.pml = B, h, dev, n_pml, "upml"
        s.copper_rs = copper_rs
        s.ground_rs = (copper_rs if ground_rs is None else ground_rs) or None
        s.n = int(round(period / h))
        s.k_top = int(round(float(np.mean(t_sub)) / h))
        assert np.all(np.abs(t_sub / h - s.k_top) < 0.5), "all thicknesses must round to the same number of cells"
        s.k_s = s.k_top + int(round(air_src / h))
        s.k_m = s.k_s + int(round(air_mon / h))
        s.k_pml = s.k_m + int(round(air_gap / h))
        s.nz = s.k_pml + n_pml
        dz_sub = t_sub / s.k_top                                           # (B,) stretched substrate cells
        dzc = np.where(np.arange(s.nz)[None, :] < s.k_top, dz_sub[:, None], h)          # (B, nz) cell k thickness
        dzmin = min(float(dz_sub.min()), h)
        s.dt = 0.99 / (C0 * math.sqrt(2 / h ** 2 + 1 / dzmin ** 2))      # Courant limit of the smallest cell
        s.eps_r, s.tan_d, s.f_loss, s.t_sub = eps_r, tan_d, f_loss, t_sub
        s.sigma_sub = 2 * math.pi * f_loss * EPS0 * eps_r * tan_d          # (B,)
        s.margin = int(round(1e-3 / h)); s.pix = int(round(0.5e-3 / h))
        assert s.margin * 2 + 16 * s.pix == s.n
        dt = s.dt
        t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        # ---- PML (by index above k_pml, all in air), as in fdtd.Cell
        L = n_pml * h
        smax = -(3 + 1) * math.log(1e-8) / (2 * ETA0 * L)
        prof = lambda d: smax * (np.clip(d, 0, None) / L) ** 3
        dE = (np.arange(s.nz + 1) - s.k_pml) * h; dH = (np.arange(s.nz) + 0.5 - s.k_pml) * h
        kz = np.arange(s.nz + 1)[None, :]
        epsE = np.where(kz < s.k_top, eps_r[:, None], 1.0); epsE[:, s.k_top] = (eps_r + 1) / 2
        sigE = np.where(kz < s.k_top, s.sigma_sub[:, None], 0.0); sigE[:, s.k_top] = s.sigma_sub / 2
        sigE = sigE + prof(dE)[None, :]
        kzz = np.arange(s.nz)[None, :]
        epsZ = np.where(kzz < s.k_top, eps_r[:, None], 1.0)
        sigZ = np.where(kzz < s.k_top, s.sigma_sub[:, None], 0.0)
        sigPz_H, sigPz_E = prof(dH), prof(dE)
        aE = sigE * dt / (2 * EPS0 * epsE)
        s.caE = t(((1 - aE) / (1 + aE))[:, None, None, :]); s.cbE = t(((dt / (EPS0 * epsE)) / (1 + aE) / h)[:, None, None, :])
        aZ = sigZ * dt / (2 * EPS0 * epsZ)
        s.caZ = t(((1 - aZ) / (1 + aZ))[:, None, None, :]); s.cbZ = t(((dt / (EPS0 * epsZ)) / (1 + aZ) / h)[:, None, None, :])
        s.psiZ_coef = t((sigPz_H / EPS0 * dt * dt / (2 * EPS0 * epsZ))[:, None, None, :])
        aH = sigPz_H * dt / (2 * EPS0)
        s.daH, s.dbH = t((1 - aH) / (1 + aH)), t((dt / MU0) / (1 + aH) / h)
        s.psiHz_coef = t(sigPz_E / EPS0 * dt * dt / (2 * MU0))
        s.dbHz = dt / MU0 / h
        # ---- stretched z: scale the d/dz differences by h / (local spacing)
        s.rH = t((h / dzc)[:, None, None, :])                                              # between E planes k, k+1
        s.rE = t((h / (0.5 * (dzc[:, :-1] + dzc[:, 1:])))[:, None, None, :])               # between H planes k-1/2, k+1/2
        s.cell_mask = None

    # ------------------------------------------------------------------ geometry
    def copper_edges(s, copper, corner="closed", p_open=0.5, seed=0, groups=None, transposed=None):
        """corner="random": every corner-only contact is cut (as corner="open" in fdtd.Cell) with probability p_open,
        independently per contact. The draw is keyed by (seed, groups[b], contact position in the untransposed cell),
        so a cell and its transpose (the same physical cell under y-polarized incidence, transposed[b] = True) open
        the same contacts. With s.cell_mask set, copper is ignored and the cell-level
        mask (B, n, n) [x, y] is used directly."""
        if s.cell_mask is not None:
            C = s.cell_mask
            return C | torch.roll(C, 1, 2), C | torch.roll(C, 1, 1)
        if corner != "random":
            return _Cell.copper_edges(s, copper, corner)
        ex, ey = _Cell.copper_edges(s, copper, "closed")
        groups = np.arange(len(copper)) if groups is None else np.asarray(groups)
        transposed = np.zeros(len(copper), bool) if transposed is None else np.asarray(transposed, bool)
        cu = copper.cpu().numpy()
        a, b, c, d = cu[:, :-1, :-1], cu[:, 1:, 1:], cu[:, :-1, 1:], cu[:, 1:, :-1]
        m, p = s.margin, s.pix
        for kind, mask in (("diag", a & b & ~c & ~d), ("anti", c & d & ~a & ~b)):
            for bb, r, cc in zip(*np.nonzero(mask)):
                key = (int(r), int(cc)) if not transposed[bb] else (int(cc), int(r))
                if np.random.default_rng([seed, int(groups[bb]), *key, kind == "diag"]).random() >= p_open:
                    continue
                xn, yn = m + (cc + 1) * p, m + (r + 1) * p
                if kind == "diag":
                    ex[bb, xn, yn] = False; ey[bb, xn, yn] = False
                else:
                    ex[bb, xn - 1, yn] = False; ey[bb, xn, yn] = False
        return ex, ey

    # ------------------------------------------------------------------ 3-D run (fdtd.Cell.run with per-sample
    # materials and stretched z; UPML only)
    @torch.no_grad()
    def run(s, copper, t_max=30e-9, f0=7e9, bw=2.33e9, tol=1e-4, check_every=2000, verbose=False, corner="closed",
            p_open=0.5, seed=0, groups=None, transposed=None):
        B, n, nz, dev = s.B, s.n, s.nz, s.dev
        n_steps = int(t_max / s.dt)
        sig_t = 1 / (2 * math.pi * bw); t0 = 6 * sig_t
        tt = np.arange(n_steps) * s.dt
        src_np = np.exp(-((tt - t0) / sig_t) ** 2 / 2) * np.sin(2 * math.pi * f0 * (tt - t0))
        src = torch.tensor(src_np, dtype=torch.float32, device=dev)
        ref = s.reference(torch.tensor(src_np, dtype=torch.float64, device=dev), n_steps).float()
        mex, mey = s.copper_edges(copper, corner, p_open, seed, groups, transposed)
        eps_avg = (s.eps_r + 1) / 2
        sig_c = 1 / (s.copper_rs * s.h) + s.sigma_sub / 2
        ca = np.exp(-sig_c * s.dt / (EPS0 * eps_avg)); cb = (1 - ca) / (sig_c * s.h)
        ca_cu = torch.tensor(ca, dtype=torch.float32, device=dev)[:, None, None]
        cb_cu = torch.tensor(cb, dtype=torch.float32, device=dev)[:, None, None]
        z = lambda *sh: torch.zeros(B, n, n, *sh, device=dev)
        Ex, Ey, Hz = z(nz + 1), z(nz + 1), z(nz + 1)
        Ez, Hx, Hy = z(nz), z(nz), z(nz)
        psiEz, psiHz = z(nz), z(nz + 1)
        monx = torch.zeros(B, n_steps, device=dev); mony = torch.zeros(B, n_steps, device=dev)
        caE, cbE, caZ, cbZ = s.caE, s.cbE, s.caZ, s.cbZ
        daH, dbH, rH, rE = s.daH, s.dbH, s.rH, s.rE
        kt, km, ks = s.k_top, s.k_m, s.k_s
        last = n_steps
        for it in range(n_steps):
            dEz_dy = torch.roll(Ez, -1, 2) - Ez
            dEz_dx = torch.roll(Ez, -1, 1) - Ez
            dEy_dz = (Ey[..., 1:] - Ey[..., :-1]) * rH
            dEx_dz = (Ex[..., 1:] - Ex[..., :-1]) * rH
            Hx = daH * Hx - dbH * (dEz_dy - dEy_dz)
            Hy = daH * Hy - dbH * (dEx_dz - dEz_dx)
            curlEz = (torch.roll(Ey, -1, 1) - Ey) - (torch.roll(Ex, -1, 2) - Ex)
            Hz = Hz - s.dbHz * curlEz - s.psiHz_coef * (2 * psiHz + curlEz / s.h)
            psiHz = psiHz + curlEz / s.h
            dHz_dy = Hz - torch.roll(Hz, 1, 2)
            dHz_dx = Hz - torch.roll(Hz, 1, 1)
            dHy_dz = (Hy[..., 1:] - Hy[..., :-1]) * rE
            dHx_dz = (Hx[..., 1:] - Hx[..., :-1]) * rE
            ex_old, ey_old = Ex[..., kt].clone(), Ey[..., kt].clone()
            Ex[..., 1:-1] = caE[..., 1:-1] * Ex[..., 1:-1] + cbE[..., 1:-1] * (dHz_dy[..., 1:-1] - dHy_dz)
            Ey[..., 1:-1] = caE[..., 1:-1] * Ey[..., 1:-1] + cbE[..., 1:-1] * (dHx_dz - dHz_dx[..., 1:-1])
            cbk = cbE[..., kt]; cak = caE[..., kt]
            cx = (Ex[..., kt] - cak * ex_old) / cbk; cy = (Ey[..., kt] - cak * ey_old) / cbk
            Ex[..., kt] = torch.where(mex, ca_cu * ex_old + cb_cu * cx, Ex[..., kt])
            Ey[..., kt] = torch.where(mey, ca_cu * ey_old + cb_cu * cy, Ey[..., kt])
            curlHz = (Hy - torch.roll(Hy, 1, 1)) - (Hx - torch.roll(Hx, 1, 2))
            Ez = caZ * Ez + cbZ * curlHz + s.psiZ_coef * (2 * psiEz + curlHz / s.h)
            psiEz = psiEz + curlHz / s.h
            if s.ground_rs:
                Ex[..., 0] = -s.ground_rs * Hy[..., 0]; Ey[..., 0] = s.ground_rs * Hx[..., 0]
            Ex[..., ks] += src[it]
            monx[:, it] = Ex[..., km].mean((1, 2)); mony[:, it] = Ey[..., km].mean((1, 2))
            if (it + 1) % check_every == 0 and STOP_FILE.exists():
                print(f"  STOP file found at step {it+1}; returning what has been computed", flush=True)
                last = it + 1
                break
            if (it + 1) % check_every == 0 and it * s.dt > 3e-9:
                rx = (monx[:, :it + 1] - ref[None, :it + 1]).abs(); ry = mony[:, :it + 1].abs()
                peak = rx.max(1).values.clamp_min(1e-30)
                tail = torch.maximum(rx[:, it + 1 - check_every:].max(1).values, ry[:, it + 1 - check_every:].max(1).values)
                if verbose:
                    print(f"  step {it+1}/{n_steps}  t={1e9*(it+1)*s.dt:.1f} ns  worst tail/peak {float((tail/peak).max()):.1e}", flush=True)
                if bool((tail / peak < tol).all()):
                    last = it + 1
                    break
        s.last_steps = last
        return monx[:, :last], mony[:, :last], ref[:last]


def fine_mask(layouts, h, etch_cells=0):
    """Cell-level copper mask (B, n, n) [x, y] of 16 x 16 layouts in the solver convention ([row, col] = [y, x]) on
    a mesh h, with every copper edge moved by etch_cells cells: > 0 removes copper (over-etching), < 0 adds copper
    (under-etching). Erosion and dilation use the 4-neighbourhood, so a corner contact opens under over-etching and
    widens under under-etching."""
    pix, margin = int(round(0.5e-3 / h)), int(round(1e-3 / h))
    n = 16 * pix + 2 * margin
    cells = layouts.repeat_interleave(pix, 1).repeat_interleave(pix, 2)
    G = torch.zeros(layouts.shape[0], n, n, dtype=torch.bool, device=layouts.device)
    G[:, margin:margin + 16 * pix, margin:margin + 16 * pix] = cells
    for _ in range(abs(etch_cells)):
        nb = [torch.roll(G, sh, d) for sh, d in ((1, 1), (-1, 1), (1, 2), (-1, 2))]
        if etch_cells > 0:
            G = G & nb[0] & nb[1] & nb[2] & nb[3]
        else:
            G = G | nb[0] | nb[1] | nb[2] | nb[3]
    return G.transpose(1, 2).contiguous()
