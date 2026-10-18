"""
mc_tolerance.py — manufacturing tolerances of our best designs (method of sutd-prcm-solverfree/mc_tolerance.py).

N boards, h = 0.125 mm, frequency axis corrected by each design's measured mesh shift: eps_r ~ U(2.60, 2.70),
tan d ~ U(0.002, 0.004), substrate t ~ U(0.95, 1.05) mm, every corner contact of every cell open with
probability 0.5 (the same contacts for a cell and its transpose); sample 0 is the nominal board.
Designs: the converter and the rotator with the widest FDTD band meeting the target, and coding pair 0
(cells A, B and their transposes). Objectives with the reported definitions (phase2_devices, search=False).

    python experiments/devices/mc_tolerance.py 64 [rotator=2 cp=5 tag=name]
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fdtd_solver"))
import fdtd_mc                                           # noqa: E402
from fdtd import to_port                                 # noqa: E402
from phase2_devices import DeviceObjectives, CodingObjectives     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mc" + ("_" + dict(a.split("=") for a in sys.argv[2:]).get("tag", "") if len(sys.argv) > 2 else ""))
os.makedirs(OUT, exist_ok=True)
F = np.linspace(2, 12, 101); FS = np.linspace(1.9, 12.0, 102)          # simulated from 1.9 GHz for the shift
DELTA = 10.705e-3
H_MC = 0.125e-3


def best(dev, design=None):
    """Widest FDTD band meeting the target, or the given design (1-based)."""
    v = json.load(open(os.path.join(HERE, f"verify_{dev}", "verify.json")))["designs"]
    ok = [r for r in v if r["meets_fdtd"]]
    r = v[design - 1] if design else max(ok, key=lambda r: list(r["fdtd"].values())[0])
    X = np.load(os.path.join(HERE, f"verify_{dev}", "spectra.npz"))["X"][r["design"] - 1]
    return X.astype(bool), r["mesh_shift_pct"] / 100, r["design"]


ARGS = dict(a.split("=") for a in sys.argv[2:])                  # e.g. rotator=2 tag=rot2
cp_x, cp_d, cp_i = best("cp", int(ARGS["cp"]) if "cp" in ARGS else None)
rot_x, rot_d, rot_i = best("rotator", int(ARGS["rotator"]) if "rotator" in ARGS else None)
pair = np.load(os.path.join(HERE, "verify_coding", "shortlist.npz"))["X"][0].astype(bool)
code_d = json.load(open(os.path.join(HERE, "verify_coding", "verify.json")))["delta_coarse_pct"] / 400   # h = 0.125 mm
LAY = np.stack([cp_x, rot_x, pair[0], pair[1], pair[0].T, pair[1].T])
DMC = [cp_d, rot_d, code_d, code_d, code_d, code_d]
PHYS = [0, 1, 2, 3, 2, 3]
TRANS = [False, False, False, False, True, True]
OB = dict(cp=DeviceObjectives("cp", F, search=False), rot=DeviceObjectives("rotator", F, search=False),
          code=CodingObjectives("free", F, search=False))


def correct(S, d):
    """S on FS (mesh h) -> converged axis on F: S_h(f) = S_0(f (1 + d))."""
    fq = F / (1 + d)
    return np.interp(fq, FS, S.real) + 1j * np.interp(fq, FS, S.imag)


def evaluate(S6):
    """S6: (6, 2, 101) converged spectra of one board -> objectives."""
    Vc, _ = OB["cp"].values(S6[None, 0], LAY[None, 0].astype(np.uint8))
    Vr, _ = OB["rot"].values(S6[None, 1], LAY[None, 1].astype(np.uint8))
    Vk, _ = OB["code"].values(S6[None, 2:], pair[None].astype(np.uint8))
    red = OB["code"].reduction(S6[None, 2:])[0]
    return dict(cp_B=float(Vc[0, 0]), cp_AR=float(Vc[0, 1]), cp_L=float(Vc[0, 3]), cp_meets=bool(OB["cp"].good(Vc)[0]),
                rot_B=float(Vr[0, 0]), rot_E=float(Vr[0, 1]), rot_meets=bool(OB["rot"].good(Vr)[0]),
                code_B=float(Vk[0, 0]), code_R=float(Vk[0, 1]), red=red)


def run_mc(N, per_run=8):
    rng = np.random.default_rng(2026)
    eps = rng.uniform(2.60, 2.70, N); tand = rng.uniform(0.002, 0.004, N); t = rng.uniform(0.95e-3, 1.05e-3, N)
    eps[0], tand[0], t[0] = 2.65, 0.003, 1.0e-3                        # sample 0: nominal materials
    lay_s = torch.tensor(LAY.transpose(0, 2, 1).copy(), device="cuda")   # solver convention
    spectra = np.zeros((N, 6, 2, 101), complex)
    t0 = time.time()
    for a in range(0, N, per_run):
        idx = np.arange(a, min(N, a + per_run)); B = 6 * len(idx)
        rep = lambda v: np.repeat(v[idx], 6)
        cell = fdtd_mc.Cell(B, h=H_MC, eps_r=rep(eps), tan_d=rep(tand), t_sub=rep(t))
        groups = np.concatenate([[1000 * i + p for p in PHYS] for i in idx])
        L = lay_s.repeat(len(idx), 1, 1)
        rxx, ryx = cell.simulate(L, FS * 1e9, t_max=30e-9, corner="random", p_open=0.5, seed=7, groups=groups,
                                 transposed=np.tile(TRANS, len(idx)))
        rxx, ryx = to_port(rxx, FS * 1e9, DELTA), to_port(ryx, FS * 1e9, DELTA)
        for j, i in enumerate(idx):
            for k in range(6):
                spectra[i, k, 0] = correct(rxx[6 * j + k], DMC[k]); spectra[i, k, 1] = correct(ryx[6 * j + k], DMC[k])
        print(f"samples {idx[0]}-{idx[-1]} done ({time.time() - t0:.0f} s)", flush=True)
    cell = fdtd_mc.Cell(6, h=H_MC)                                       # nominal board, contacts closed
    rxx, ryx = cell.simulate(lay_s, FS * 1e9, t_max=30e-9, corner="closed")
    rxx, ryx = to_port(rxx, FS * 1e9, DELTA), to_port(ryx, FS * 1e9, DELTA)
    nominal = np.stack([np.stack([correct(rxx[k], DMC[k]), correct(ryx[k], DMC[k])]) for k in range(6)])
    rows = [evaluate(spectra[i]) for i in range(N)]
    nom = evaluate(nominal)
    q = lambda key: np.percentile([r[key] for r in rows], [5, 50, 95]).round(2).tolist()
    summary = dict(N=N, designs=dict(cp=cp_i, rotator=rot_i, coding_pair=0),
                   ranges=dict(eps_r=[2.60, 2.70], tan_d=[0.002, 0.004], t_mm=[0.95, 1.05], p_corner_open=0.5),
                   nominal={k: v for k, v in nom.items() if k != "red"},
                   yield_cp=float(np.mean([r["cp_meets"] for r in rows])), yield_rot=float(np.mean([r["rot_meets"] for r in rows])),
                   yield_code_10pct=float(np.mean([r["code_B"] >= 10 for r in rows])),
                   cp_B_p5_50_95=q("cp_B"), cp_AR_p5_50_95=q("cp_AR"), rot_B_p5_50_95=q("rot_B"), rot_E_p5_50_95=q("rot_E"),
                   code_B_p5_50_95=q("code_B"))
    np.savez(os.path.join(OUT, "mc.npz"), spectra=spectra, nominal=nominal, eps=eps, tand=tand, t=t, F=F,
             red=np.stack([r["red"] for r in rows]), red_nominal=nom["red"])
    json.dump(dict(summary=summary, samples=[{k: v for k, v in r.items() if k != "red"} for r in rows]),
              open(os.path.join(OUT, "mc.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    run_mc(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
