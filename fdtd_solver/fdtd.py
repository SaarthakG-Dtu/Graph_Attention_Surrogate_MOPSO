"""Standalone 3-D FDTD (Yee) unit-cell solver with a z-UPML, for verifying pixelated reflectors.

Geometry (draft, Sec. II-A / IX-A): 10 mm square cell, 16 x 16 pixels of 0.5 mm in the central
8 mm (1 mm margin), copper pixels modelled as zero-thickness PEC on top of a t = 1.0 mm substrate
(eps_r = 2.65, tan d = 0.003), PEC or lossy copper ground. Periodic in x and y, normal incidence, x-polarised.

Method
  * Yee grid, cubic cells of size h (default 0.25 mm = 2 cells per pixel), float32, batched over
    layouts on the GPU. Periodic x/y via torch.roll.
  * Substrate loss as a conductivity sigma = 2 pi f_loss eps0 eps_r tan d (exact at f_loss,
    default 10 GHz). Tangential E on the air/substrate interface uses the averaged permittivity.
  * Copper: tangential E (Ex, Ey) on the interface plane is forced to zero on every edge that lies
    in the CLOSURE of a copper pixel. Two pixels that touch only at a corner therefore share a node
    and conduct; whether such a contact conducts in reality is mesh-/fabrication-dependent.
  * Top boundary: uniaxial PML graded in z only (Gedney): with s = diag(s_z, s_z, 1/s_z) the
    transverse components see a plain conductivity sigma_z (and matched magnetic sigma_z mu0/eps0),
    the z components get an integral (psi) term. Polynomial grading m = 3, R0 = 1e-8. PEC behind it.
  * Source: soft Ex sheet at z_s. Monitor: plane average of Ex, Ey at z_m > z_s (the average over a
    period is exactly the specular Floquet order). The incident wave is taken from a 1-D FDTD with
    the same h, dt and source (identical numerics for the specular order) and subtracted.
  * Reflection referred to the cell surface z_top with the NUMERICAL wavenumber of the grid:
        r(f) = R_m(f) / I_m(f) * exp(2 j k~ (z_s - z_top))      (e^{+j w t} convention)
    Use to_port() to move the reference to the dataset's Floquet port, Delta above the surface.
"""
import math
import numpy as np
import torch

import pathlib
STOP_FILE = pathlib.Path(__file__).parent / "STOP_FDTD"
C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * math.pi
ETA0 = MU0 * C0


class Cell:
    def __init__(s, h=0.25e-3, period=10e-3, t_sub=1.0e-3, eps_r=2.65, tan_d=0.003, f_loss=10e9,
                 air_src=5e-3, air_mon=2e-3, air_gap=2e-3, n_pml=12, dev="cuda", pml="upml", kappa_max=1.0, f_alpha=1e9,
                 copper_rs=None, ground_rs=None):
        s.h, s.dev, s.n_pml, s.pml = h, dev, n_pml, pml
        # ground_rs: surface resistance of the copper ground (ohm); None = same as copper_rs (all copper lossy), 0 = PEC.
        # The lossy ground is a surface-impedance (Leontovich) boundary E_t = R_s n x H, n = +z:
        # Ex(0) = -R_s Hy(h/2), Ey(0) = R_s Hx(h/2). On 96 dataset cells it lowers the median error of the lowest
        # reflected power against CST from 0.006 (pixels only) to 0.0008 (fdtd_checks/ground_loss.json).
        s.ground_rs = (copper_rs if ground_rs is None else ground_rs) or None
        # copper_rs: surface resistance of the copper (ohm). None = PEC. Otherwise the copper edges become a resistive
        # sheet, E_t = R_s J (exponential time-stepping, so the large conductivity stays stable). Copper, 5.8e7 S/m:
        # R_s = sqrt(pi f mu0 / sigma) = 0.026 ohm at 10 GHz (0.018 mm >> skin depth 0.66 um).
        s.copper_rs = copper_rs
        s.n = int(round(period / h))
        s.k_top = int(round(t_sub / h))
        s.k_s = s.k_top + int(round(air_src / h))
        s.k_m = s.k_s + int(round(air_mon / h))
        s.k_pml = s.k_m + int(round(air_gap / h))
        s.nz = s.k_pml + n_pml
        s.dt = 0.99 * h / (C0 * math.sqrt(3))
        s.eps_r, s.tan_d, s.f_loss, s.t_sub = eps_r, tan_d, f_loss, t_sub
        s.sigma_sub = 2 * math.pi * f_loss * EPS0 * eps_r * tan_d
        s.margin = int(round(1e-3 / h)); s.pix = int(round(0.5e-3 / h))
        assert s.margin * 2 + 16 * s.pix == s.n, "grid must resolve the 0.5 mm pixels and 1 mm margin"
        # ---- material and PML profiles along z
        L = n_pml * h
        smax = -(3 + 1) * math.log(1e-8) / (2 * ETA0 * L)
        def prof(zpos):
            d = np.clip(zpos - s.k_pml * h, 0, None)
            return smax * (d / L) ** 3
        kE = np.arange(s.nz + 1) * h                       # Ex, Ey, Hz planes
        kH = (np.arange(s.nz) + 0.5) * h                   # Hx, Hy, Ez planes
        epsE = np.where(np.arange(s.nz + 1) < s.k_top, eps_r, 1.0); epsE[s.k_top] = (eps_r + 1) / 2
        sigE = np.where(np.arange(s.nz + 1) < s.k_top, s.sigma_sub, 0.0); sigE[s.k_top] = s.sigma_sub / 2
        sigE = sigE + prof(kE)
        epsZ = np.where(np.arange(s.nz) < s.k_top, eps_r, 1.0)
        sigZ = np.where(np.arange(s.nz) < s.k_top, s.sigma_sub, 0.0)
        sigPz_H = prof(kH); sigPz_E = prof(kE)              # PML sigma at H(k+1/2) and at Hz/E(k) planes
        dt = s.dt
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        aE = sigE * dt / (2 * EPS0 * epsE)
        s.caE, s.cbE = t((1 - aE) / (1 + aE)), t((dt / (EPS0 * epsE)) / (1 + aE) / h)
        aZ = sigZ * dt / (2 * EPS0 * epsZ)
        s.caZ, s.cbZ = t((1 - aZ) / (1 + aZ)), t((dt / (EPS0 * epsZ)) / (1 + aZ) / h)
        # Gedney D/B form for the z components, time-centred: eps dEz/dt = dDz/dt + (sz/eps0) (Dz^{n+1}+Dz^n)/2,
        # Dz = dt * psi with psi the running sum of curl H. (An uncentred version blows up in ~500 steps.)
        s.psiZ_coef = t(sigPz_H / EPS0 * dt * dt / (2 * EPS0 * epsZ))
        aH = sigPz_H * dt / (2 * EPS0)                          # sigma* / (2 mu0) * dt = sigma dt / (2 eps0)
        s.daH, s.dbH = t((1 - aH) / (1 + aH)), t((dt / MU0) / (1 + aH) / h)
        s.psiHz_coef = t(sigPz_E / EPS0 * dt * dt / (2 * MU0))
        s.dbHz = dt / MU0 / h
        s.inPML_H = bool((sigPz_H > 0).any())
        if pml == "cpml":
            # Convolutional PML (Roden & Gedney 2000) on the d/dz terms only, kappa = 1, CFS alpha graded from
            # alpha_max at the interface to 0: stable for oblique and evanescent waves (the UPML above grows in time
            # on large supercells whose diffracted orders run near grazing). Materials keep their own loss only.
            sig0 = np.where(np.arange(s.nz + 1) < s.k_top, s.sigma_sub, 0.0); sig0[s.k_top] = s.sigma_sub / 2
            aE0 = sig0 * dt / (2 * EPS0 * epsE)
            s.caE, s.cbE = t((1 - aE0) / (1 + aE0)), t((dt / (EPS0 * epsE)) / (1 + aE0) / h)
            s.daH, s.dbH = t(np.ones(s.nz)), t(np.full(s.nz, dt / MU0 / h))
            # kappa > 1 attenuates evanescent fields (near-cut-off diffracted orders of large supercells), and the
            # complex-frequency shift alpha keeps the low-frequency response stable.
            smaxc = 0.8 * (3 + 1) / (ETA0 * h)
            amax = 2 * math.pi * f_alpha * EPS0
            def coef(zpos):
                d = np.clip(zpos - s.k_pml * h, 0, None) / L
                sg = smaxc * d ** 3; al = amax * (1 - d) * (d > 0); ka = 1 + (kappa_max - 1) * d ** 3
                b = np.exp(-(sg / ka + al) * dt / EPS0)
                a = np.where(sg > 0, sg / np.maximum(sg * ka + ka ** 2 * al, 1e-30) * (b - 1), 0.0)
                return t(b), t(a), t(1 / ka)
            s.cp_hs = slice(s.k_pml, s.nz)                           # H planes (k + 1/2) h inside the PML
            s.cp_bH, s.cp_aH, s.cp_kH = coef(kH[s.k_pml:s.nz])
            s.cp_es = slice(s.k_pml, s.nz - 1)                       # E planes k_pml+1 .. nz-1, relative to 1..nz-1
            s.cp_bE, s.cp_aE, s.cp_kE = coef(kE[s.k_pml + 1:s.nz])

    # ------------------------------------------------------------------ geometry
    def copper_edges(s, copper, corner="closed"):
        """copper: (B, 16, 16) bool, [row, col] with x along columns. Returns masks to zero Ex, Ey at k_top.

        corner="closed": two pixels touching only at a corner share the grid node and conduct (staircase PEC).
        corner="open":   the contact is cut: at every corner-only contact the two edge segments of one of
                         the two pixels that end on the shared node are left free, so no PEC path crosses it.
        """
        cells = copper.repeat_interleave(s.pix, 1).repeat_interleave(s.pix, 2)       # B, 16p, 16p  [y, x]
        C = torch.zeros(copper.shape[0], s.n, s.n, dtype=torch.bool, device=s.dev)     # [x, y]
        C[:, s.margin:s.margin + 16 * s.pix, s.margin:s.margin + 16 * s.pix] = cells.transpose(1, 2)
        # Ex at (i+1/2, j): cells (i, j-1) and (i, j) share it; Ey at (i, j+1/2): cells (i-1, j) and (i, j)
        ex = C | torch.roll(C, 1, 2)
        ey = C | torch.roll(C, 1, 1)
        if corner == "open":
            cu = copper.cpu().numpy()
            a, b, c, d = cu[:, :-1, :-1], cu[:, 1:, 1:], cu[:, :-1, 1:], cu[:, 1:, :-1]   # (r,c) (r+1,c+1) (r,c+1) (r+1,c)
            m, p = s.margin, s.pix
            for kind, mask in (("diag", a & b & ~c & ~d), ("anti", c & d & ~a & ~b)):
                for bb, r, cc in zip(*np.nonzero(mask)):
                    xn, yn = m + (cc + 1) * p, m + (r + 1) * p                       # shared node, x = column, y = row
                    if kind == "diag":     # free pixel (r+1, c+1): its bottom edge and left edge starting at the node
                        ex[bb, xn, yn] = False; ey[bb, xn, yn] = False
                    else:                  # free pixel (r+1, c): its bottom edge ending at the node and right edge starting there
                        ex[bb, xn - 1, yn] = False; ey[bb, xn, yn] = False
        return ex, ey

    # ------------------------------------------------------------------ 1-D incident reference
    def reference(s, src, n_steps):
        nz = s.nz + 200                                    # extra free space + bottom PML to swallow the downward wave
        off = 200
        L = s.n_pml * s.h                                  # same PML as the 3-D run (top and bottom)
        smax = -(4) * math.log(1e-8) / (2 * ETA0 * L)
        zE = np.arange(nz + 1) * s.h; zH = (np.arange(nz) + 0.5) * s.h
        top0 = (s.k_pml + off) * s.h
        bot1 = L
        prof = lambda z: smax * (np.clip(z - top0, 0, None) / L) ** 3 + smax * (np.clip(bot1 - z, 0, None) / L) ** 3
        aE = prof(zE) * s.dt / (2 * EPS0); aH = prof(zH) * s.dt / (2 * EPS0)
        caE, cbE = (1 - aE) / (1 + aE), (s.dt / EPS0) / (1 + aE) / s.h
        daH, dbH = (1 - aH) / (1 + aH), (s.dt / MU0) / (1 + aH) / s.h
        # on the CPU: ~10^5 steps of a few hundred numbers are launch-bound on a busy GPU
        src = src.detach().cpu().numpy() if torch.is_tensor(src) else np.asarray(src)
        ex = np.zeros(nz + 1); hy = np.zeros(nz); mon = np.zeros(n_steps)
        ks, km = s.k_s + off, s.k_m + off
        ca, cb = caE[1:-1], cbE[1:-1]
        for n in range(n_steps):
            hy = daH * hy - dbH * (ex[1:] - ex[:-1])
            ex[1:-1] = ca * ex[1:-1] - cb * (hy[1:] - hy[:-1])
            ex[ks] += src[n]
            mon[n] = ex[km]
        return torch.tensor(mon, dtype=torch.float64, device=s.dev)

    # ------------------------------------------------------------------ 3-D run
    @torch.no_grad()
    def run(s, copper, t_max=30e-9, f0=7e9, bw=2.33e9, tol=1e-4, check_every=2000, verbose=False, corner="closed"):
        B, n, nz, dev = copper.shape[0], s.n, s.nz, s.dev
        n_steps = int(t_max / s.dt)
        sig_t = 1 / (2 * math.pi * bw); t0 = 6 * sig_t
        tt = np.arange(n_steps) * s.dt
        src_np = np.exp(-((tt - t0) / sig_t) ** 2 / 2) * np.sin(2 * math.pi * f0 * (tt - t0))
        src = torch.tensor(src_np, dtype=torch.float32, device=dev)
        ref = s.reference(torch.tensor(src_np, dtype=torch.float64, device=dev), n_steps).float()
        mex, mey = s.copper_edges(copper, corner)
        keepx, keepy = (~mex).float(), (~mey).float()
        if s.copper_rs:
            eps_avg = (s.eps_r + 1) / 2
            sig_c = 1 / (s.copper_rs * s.h) + s.sigma_sub / 2
            ca_cu = math.exp(-sig_c * s.dt / (EPS0 * eps_avg))
            cb_cu = (1 - ca_cu) / (sig_c * s.h)
        z = lambda *sh: torch.zeros(B, n, n, *sh, device=dev)
        Ex, Ey, Hz = z(nz + 1), z(nz + 1), z(nz + 1)
        Ez, Hx, Hy = z(nz), z(nz), z(nz)
        psiEz, psiHz = z(nz), z(nz + 1)
        monx = torch.zeros(B, n_steps, device=dev); mony = torch.zeros(B, n_steps, device=dev)
        caE, cbE, caZ, cbZ = s.caE, s.cbE, s.caZ, s.cbZ
        daH, dbH = s.daH, s.dbH
        kt, km, ks = s.k_top, s.k_m, s.k_s
        cpml = s.pml == "cpml"
        if cpml:
            hs, es = s.cp_hs, s.cp_es
            nh, ne = hs.stop - hs.start, es.stop - es.start
            psi_hx, psi_hy = z(nh), z(nh)
            psi_ex, psi_ey = z(ne), z(ne)
            del psiEz, psiHz
        last = n_steps
        for it in range(n_steps):
            # ---- H update
            dEz_dy = torch.roll(Ez, -1, 2) - Ez
            dEz_dx = torch.roll(Ez, -1, 1) - Ez
            dEy_dz = Ey[..., 1:] - Ey[..., :-1]
            dEx_dz = Ex[..., 1:] - Ex[..., :-1]
            if cpml:
                psi_hx = s.cp_bH * psi_hx + s.cp_aH * dEy_dz[..., hs]; dEy_dz[..., hs] = dEy_dz[..., hs] * s.cp_kH + psi_hx
                psi_hy = s.cp_bH * psi_hy + s.cp_aH * dEx_dz[..., hs]; dEx_dz[..., hs] = dEx_dz[..., hs] * s.cp_kH + psi_hy
            Hx = daH * Hx - dbH * (dEz_dy - dEy_dz)
            Hy = daH * Hy - dbH * (dEx_dz - dEz_dx)
            curlEz = (torch.roll(Ey, -1, 1) - Ey) - (torch.roll(Ex, -1, 2) - Ex)
            if cpml:
                Hz = Hz - s.dbHz * curlEz
            else:
                Hz = Hz - s.dbHz * curlEz - s.psiHz_coef * (2 * psiHz + curlEz / s.h)
                psiHz = psiHz + curlEz / s.h
            # ---- E update (interior planes 1..nz-1; k = 0 ground and k = nz backing are PEC)
            dHz_dy = Hz - torch.roll(Hz, 1, 2)
            dHz_dx = Hz - torch.roll(Hz, 1, 1)
            dHy_dz = Hy[..., 1:] - Hy[..., :-1]
            dHx_dz = Hx[..., 1:] - Hx[..., :-1]
            if cpml:
                psi_ex = s.cp_bE * psi_ex + s.cp_aE * dHy_dz[..., es]; dHy_dz[..., es] = dHy_dz[..., es] * s.cp_kE + psi_ex
                psi_ey = s.cp_bE * psi_ey + s.cp_aE * dHx_dz[..., es]; dHx_dz[..., es] = dHx_dz[..., es] * s.cp_kE + psi_ey
            if s.copper_rs:
                ex_old, ey_old = Ex[..., kt].clone(), Ey[..., kt].clone()
            Ex[..., 1:-1] = caE[1:-1] * Ex[..., 1:-1] + cbE[1:-1] * (dHz_dy[..., 1:-1] - dHy_dz)
            Ey[..., 1:-1] = caE[1:-1] * Ey[..., 1:-1] + cbE[1:-1] * (dHx_dz - dHz_dx[..., 1:-1])
            if s.copper_rs:                                  # resistive copper sheet: same curl, sheet coefficients
                cx = (Ex[..., kt] - caE[kt] * ex_old) / cbE[kt]; cy = (Ey[..., kt] - caE[kt] * ey_old) / cbE[kt]
                Ex[..., kt] = torch.where(mex, ca_cu * ex_old + cb_cu * cx, Ex[..., kt])
                Ey[..., kt] = torch.where(mey, ca_cu * ey_old + cb_cu * cy, Ey[..., kt])
            curlHz = (Hy - torch.roll(Hy, 1, 1)) - (Hx - torch.roll(Hx, 1, 2))
            if cpml:
                Ez = caZ * Ez + cbZ * curlHz
            else:
                Ez = caZ * Ez + cbZ * curlHz + s.psiZ_coef * (2 * psiEz + curlHz / s.h)
                psiEz = psiEz + curlHz / s.h
            if not s.copper_rs:
                Ex[..., kt] *= keepx; Ey[..., kt] *= keepy
            if s.ground_rs:
                Ex[..., 0] = -s.ground_rs * Hy[..., 0]; Ey[..., 0] = s.ground_rs * Hx[..., 0]
            Ex[..., ks] += src[it]
            monx[:, it] = Ex[..., km].mean((1, 2)); mony[:, it] = Ey[..., km].mean((1, 2))
            if (it + 1) % check_every == 0 and STOP_FILE.exists():      # graceful stop; never force-kill CUDA jobs here
                print(f"  STOP file found at step {it+1}; returning what has been computed", flush=True)
                last = it + 1
                break
            if (it + 1) % check_every == 0 and it * s.dt > 3e-9:
                rx = (monx[:, :it + 1] - ref[None, :it + 1]).abs(); ry = mony[:, :it + 1].abs()
                peak = rx.max(1).values.clamp_min(1e-30)            # co-pol peak scales both channels
                tail = torch.maximum(rx[:, it + 1 - check_every:].max(1).values, ry[:, it + 1 - check_every:].max(1).values)
                if verbose:
                    print(f"  step {it+1}/{n_steps}  t={1e9*(it+1)*s.dt:.1f} ns  worst tail/peak {float((tail/peak).max()):.1e}", flush=True)
                if bool((tail / peak < tol).all()):
                    last = it + 1
                    break
        s.last_steps = last
        return monx[:, :last], mony[:, :last], ref[:last]

    def spectra(s, monx, mony, ref, freqs):
        """Reflection at the cell surface, e^{+jwt} convention: (r_xx, r_yx), each (B, F) complex."""
        T = monx.shape[1]
        f = torch.tensor(freqs, dtype=torch.float64, device=s.dev)
        tvec = torch.arange(T, dtype=torch.float64, device=s.dev) * s.dt
        out = []
        for sig in (monx - ref[None], mony, ref[None]):
            acc = 0
            for a in range(0, T, 8192):
                ph = torch.exp(-2j * math.pi * f[None, :] * tvec[a:a + 8192, None])
                acc = acc + sig[:, a:a + 8192].double().to(torch.complex128) @ ph
            out.append(acc)
        Rx, Ry, I = out
        w = 2 * math.pi * f
        kt = 2 / s.h * torch.asin((s.h / (C0 * s.dt)) * torch.sin(w * s.dt / 2))     # numerical wavenumber
        corr = torch.exp(2j * kt * (s.k_s - s.k_top) * s.h)
        return (Rx / I * corr).cpu().numpy(), (Ry / I * corr).cpu().numpy()

    def simulate(s, copper, freqs, **kw):
        mx, my, ref = s.run(copper, **kw)
        return s.spectra(mx, my, ref, freqs)


class Supercell(Cell):
    """M x M unit cells (10 mm each, 16 x 16 pixels inside its own 1 mm margin) in one periodic supercell, for
    coding checkerboards: the specular (plane-averaged) reflection then includes the coupling between different
    cells and the finite tile size, which the area-average formula ignores. Diffracted orders leave through the PML."""
    def __init__(s, M, **kw):
        super().__init__(**kw)
        s.M, s.n_cell = M, s.n
        s.n = M * s.n_cell

    def copper_edges(s, copper, corner="closed"):
        """copper: (B, 16M, 16M) bool in this solver's convention ([row, col] = [y, x]); block (I, J) is the cell
        at y-block I, x-block J."""
        assert corner == "closed"
        B, p, m, nc, w = copper.shape[0], s.pix, s.margin, s.n_cell, 16 * s.pix
        cells = copper.repeat_interleave(p, 1).repeat_interleave(p, 2)
        G = torch.zeros(B, s.n, s.n, dtype=torch.bool, device=s.dev)                     # [y, x]
        for I in range(s.M):
            for J in range(s.M):
                G[:, I * nc + m:I * nc + m + w, J * nc + m:J * nc + m + w] = cells[:, I * w:(I + 1) * w, J * w:(J + 1) * w]
        C = G.transpose(1, 2).contiguous()                                                # [x, y]
        return C | torch.roll(C, 1, 2), C | torch.roll(C, 1, 1)


def checkerboard(A, B, N):
    """Cells A, B (16 x 16, corrected convention: rows = x) in a 2 x 2-tile checkerboard of N x N-cell tiles, as a
    (32N, 32N) layout in the solver's convention ([y, x]). Its transpose is the same board for y-polarized incidence."""
    An, Bn = A.T, B.T
    M = 2 * N
    full = torch.zeros(16 * M, 16 * M, dtype=torch.bool, device=A.device)
    for I in range(M):
        for J in range(M):
            full[16 * I:16 * (I + 1), 16 * J:16 * (J + 1)] = An if (I // N + J // N) % 2 == 0 else Bn
    return full


def slab_reflection(freqs, t=1.0e-3, eps_r=2.65, sigma=None, tan_d=0.003, zs=0.0):
    """Analytic grounded-slab reflection at the slab surface, e^{+jwt} convention; zs: ground surface impedance."""
    w = 2 * np.pi * np.asarray(freqs)
    ec = eps_r - 1j * (sigma / (w * EPS0) if sigma is not None else eps_r * tan_d)
    k = w / C0 * np.sqrt(ec)
    zd = ETA0 / np.sqrt(ec); tn = np.tan(k * t)
    zin = zd * (zs + 1j * zd * tn) / (zd + 1j * zs * tn)
    return (zin - ETA0) / (zin + ETA0)


def to_port(r, freqs, delta=10.705e-3):
    """Move the reference from the cell surface to a plane delta above it (the dataset's port)."""
    k = 2 * np.pi * np.asarray(freqs) / C0
    return r * np.exp(-2j * k * delta)
