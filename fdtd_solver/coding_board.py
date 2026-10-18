"""Full-wave FDTD of a coding checkerboard (supercell of 2 x 2 tiles, N x N cells per tile).

  python -I coding_board.py --search search/coding --design 5 --tiles 4 [--pol both] [--copper-rs 0.026] [--t-max 20e-9]
Uses the convolutional PML (kappa 5, alpha at 3 GHz, 20 cells). Large boards still show a slow late-time growth, so the
spectrum is taken from the first --t-cut ns (default 15) with a taper (board_spectrum.py; on the stable 3x3 and 4x4-tile
runs this matches the full 40 ns result to 0.1-0.3 dB). --pol both runs x- and y-polarized incidence together (the y run
is the transposed board). The result is moved to zero mesh size with the shift measured on the unit cells
(verify.json: delta_coarse_pct). Output: <search>/board_d<design>_N<tiles>_<pol>[_cu].npz with the specular reflectance
s(f) = |r_co|^2 + |r_cross|^2 relative to a metal plate, and the monitor time signals.
"""
import sys, json, math, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, torch
from fdtd import Supercell, checkerboard
from extrapolate import warp

p = argparse.ArgumentParser()
p.add_argument("--search", required=True); p.add_argument("--design", type=int, required=True)
p.add_argument("--tiles", type=int, required=True); p.add_argument("--pol", choices=["x", "y", "both"], default="both")
p.add_argument("--t-max", type=float, default=20e-9); p.add_argument("--t-cut", type=float, default=15.0)
p.add_argument("--h", type=float, default=0.25e-3); p.add_argument("--copper-rs", type=float, default=None)
args = p.parse_args()
skip = pathlib.Path(__file__).parent / f"SKIP_BOARD_d{args.design}_N{args.tiles}"     # lets a queued board be dropped
if skip.exists():
    print(f"skipped: {skip.name} exists", flush=True); sys.exit(0)
S = pathlib.Path(__file__).parent / args.search
X = np.load(S / "shortlist.npz")["X"][args.design]
A, B = (torch.tensor(X[k], device="cuda").bool() for k in (0, 1))
board = checkerboard(A, B, args.tiles)
pols = ["x", "y"] if args.pol == "both" else [args.pol]
boards = torch.stack([board if q == "x" else board.T.contiguous() for q in pols])
F = np.linspace(2, 12, 101)
sc = Supercell(2 * args.tiles, h=args.h, pml="cpml", kappa_max=5, f_alpha=3e9, n_pml=20, copper_rs=args.copper_rs)
mx, my, ref = sc.run(boards, t_max=args.t_max, tol=1e-4, verbose=True)
mx, my, ref = mx.cpu().numpy().astype(float), my.cpu().numpy().astype(float), ref.cpu().numpy().astype(float)
dt = sc.dt
n = min(len(ref), int(args.t_cut * 1e-9 / dt))
w = np.ones(n); m = n // 4; w[n - m:] = 0.5 * (1 + np.cos(np.linspace(0, math.pi, m)))
t = np.arange(n) * dt
ph = np.exp(-2j * math.pi * (F * 1e9)[:, None] * t[None])
I = ph @ ref[:n]
s_raw = np.stack([(np.abs(ph @ ((mx[q, :n] - ref[:n]) * w)) ** 2 + np.abs(ph @ (my[q, :n] * w)) ** 2) / np.abs(I) ** 2
                  for q in range(len(pols))])
vj = S / "verify.json" if (S / "verify.json").exists() else S / "verify_cu.json"
delta = json.loads(vj.read_text())["delta_coarse_pct"] / 100 * (args.h / 0.25e-3) ** 2
s0 = np.stack([warp(s.astype(complex), F, 1 / (1 + delta)).real for s in s_raw])
s0 = np.where(np.isnan(s0), s_raw, s0)
sfx = "_cu" if args.copper_rs else ""
np.savez(S / f"board_d{args.design}_N{args.tiles}_{args.pol}{sfx}.npz", freq=F, pols=np.array(pols), s_raw=s_raw, s=s0,
         delta=delta, t_cut_ns=args.t_cut, mx=mx, my=my, ref=ref, dt=dt)
worst = s0.max(0)                                    # the worse of the polarizations at each frequency
red = -10 * np.log10(np.clip(worst, 1e-3, None))
band = F[red >= 10]
print(f"BOARD design {args.design} N={args.tiles} pol={args.pol}{' copper loss' if sfx else ''}: max specular reflectance "
      f"{s0.max():.3f}; -10 dB (worse pol) at {len(band)} of 101 frequencies "
      f"({band.min() if len(band) else 0:.1f}-{band.max() if len(band) else 0:.1f} GHz); steps {sc.last_steps}", flush=True)
