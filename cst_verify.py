"""
cst_verify.py — CST full-wave runs at the SUTD-PRCM unit cell (cell_common.py).

    python cst_verify.py calibrate                       # find the substrate thickness
    python cst_verify.py calibrate --thicknesses 0.6 0.8 1.0 --n_designs 3
    python cst_verify.py verify --run_dir <mopso run dir> --top 8
    python cst_verify.py status

Geometry (config CST_*): 16x16 copper pixels of 0.5 mm, 1 mm padding (10 mm
cell), 0.018 mm patches, 0.18 mm copper ground, eps_r = 2.65, tan d = 0.003,
unit-cell boundaries in x/y, open in z, normal incidence, 2-12 GHz, 1001 pts.
The SUTD papers do not state the substrate thickness, so `calibrate`
simulates a few dataset layouts at several thicknesses and ranks them by the
error against the stored dataset spectra (|R| and |T|).

Every simulation is stored as an .npz (resume-safe: finished runs are skipped).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

import config as C


def _force_cst_2024():
    import win32com.client
    orig = win32com.client.gencache.EnsureDispatch
    if getattr(orig, "_cst2024", False):        # already patched
        return

    def dispatch(prog_id):
        return orig("CSTStudio.Application.2024" if prog_id == "CSTStudio.Application" else prog_id)
    dispatch._cst2024 = True
    win32com.client.gencache.EnsureDispatch = dispatch


FREQ = [2.0, 12.0]          # CST sweep (GHz); set from --fmin/--fmax
GROW = [0.0]                # patch enlargement (mm) for corner contacts; --grow


def workspace(t_sub):
    tag = (f"f{FREQ[0]:g}-{FREQ[1]:g}_g{C.CST_GRID_SIZE}_p{C.CST_PATCH_SIZE_MM:g}_pad{C.CST_PADDING_MM:g}_sub{t_sub:g}"
           f"_eps{C.CST_SUB_EPS:g}_tand{C.CST_SUB_TAND:g}_gnd{C.CST_GROUND_T_MM:g}"
           f"_pt{C.CST_PATCH_T_MM:g}_{C.CST_CONDUCTOR}")
    return os.path.join(C.CST_DIR, "workspaces", tag)


ORIENT = {"as_is": lambda m: m, "transpose": lambda m: m.T, "fliplr": lambda m: m[:, ::-1],
          "flipud": lambda m: m[::-1, :], "rot90": lambda m: np.rot90(m), "invert": lambda m: 1 - m}


def simulate(matrix, t_sub, out_npz):
    """Run one layout at substrate thickness t_sub; cached in out_npz."""
    if os.path.exists(out_npz):
        return dict(np.load(out_npz))
    _force_cst_2024()
    import cell_common
    from cell_common import build_base_template, simulate_design
    cell_common.FREQ_MIN, cell_common.FREQ_MAX = FREQ
    ws = workspace(t_sub)
    os.makedirs(ws, exist_ok=True)
    build_base_template(ws, C.CST_GRID_SIZE, C.CST_PATCH_SIZE_MM, C.CST_PADDING_MM,
                        C.CST_SUB_EPS, C.CST_SUB_TAND, t_sub,
                        ground_thickness=C.CST_GROUND_T_MM, conductor=C.CST_CONDUCTOR)
    import time
    for attempt in range(4):          # CST may still hold the previous run's result files
        try:
            freq, s11, s21, s12, s22 = simulate_design(
                ws, np.asarray(matrix, dtype=np.uint8), C.CST_GRID_SIZE, C.CST_PATCH_SIZE_MM, C.CST_PADDING_MM,
                t_sub, patch_thickness=C.CST_PATCH_T_MM, conductor=C.CST_CONDUCTOR, return_all_modes=True,
                grow=GROW[0])
            break
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(30)
    res = dict(freq=np.asarray(freq, float), s11=np.asarray(s11, complex), s21=np.asarray(s21, complex),
               s12=np.asarray(s12, complex), s22=np.asarray(s22, complex),
               matrix=np.asarray(matrix, np.uint8), t_sub=float(t_sub))
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **res)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Calibration against the dataset
# ─────────────────────────────────────────────────────────────────────────────

def _calibration_designs(n, seed, n_plg=1, extra=()):
    """Layouts with clear resonances: n RDN + n_plg PLG (+ one of each extra
    class), from the test splits."""
    from dataset import class_split_indices, load_class, load_full_curves
    rng = np.random.default_rng(seed)
    out = []
    for cls, k in (("RDN", n), ("PLG", n_plg)) + tuple((c, 1) for c in extra):
        if k == 0:
            continue
        im, cu = load_class(cls)
        test = class_split_indices(cls)["test"]
        deep = test[np.abs(cu[test, 0]).min(1) < 0.6]           # visibly resonant
        for i in rng.choice(deep, k, replace=False):
            full = load_full_curves(cls, [i])[0]
            out.append(dict(cls=cls, index=int(i), matrix=im[i], R=full[0], T=full[1]))
    return out


def _errors(sim, gt, offsets=(0.0, 1.0)):
    """|R|, |T| errors between a CST run and a dataset sample for both mode
    assignments, with the dataset frequency axis shifted by each offset (GHz)."""
    f_gt = np.linspace(2, 12, 1001)
    res = {}
    for off in offsets:
        fs = f_gt + off
        m = (fs >= sim["freq"][0]) & (fs <= sim["freq"][-1])
        for name, co, cr in (("m1x", sim["s11"], sim["s21"]), ("m2x", sim["s22"], sim["s12"])):
            R = np.interp(fs[m], sim["freq"], np.abs(co))
            T = np.interp(fs[m], sim["freq"], np.abs(cr))
            res[f"{name}_off{off:g}"] = dict(R_mse=float(np.mean((R - np.abs(gt["R"][m])) ** 2)),
                                             T_mse=float(np.mean((T - np.abs(gt["T"][m])) ** 2)))
    return res


def calibrate(args):
    _force_cst_2024()
    out = os.path.join(C.CST_DIR, "calibration")
    os.makedirs(out, exist_ok=True)
    designs = _calibration_designs(args.n_designs, args.seed, args.n_plg, args.extra_classes)
    rows = []
    for t in args.thicknesses:
        for d in designs:
            tag = (f"{d['cls']}_{d['index']}_t{t:g}" + ("" if args.orient == "as_is" else f"_{args.orient}")
                   + ("" if FREQ == [2.0, 12.0] else f"_f{FREQ[0]:g}-{FREQ[1]:g}")
                   + ("" if GROW[0] == 0 else f"_grow{GROW[0]:g}"))
            print(f"[calibrate] {tag}", flush=True)
            try:
                sim = simulate(ORIENT[args.orient](d["matrix"]), t, os.path.join(out, "runs", tag + ".npz"))
            except Exception as err:                       # keep going; log the failure
                print(f"  failed: {err}")
                with open(os.path.join(out, "errors.log"), "a") as fh:
                    fh.write(f"{tag}: {err}\n")
                continue
            e = _errors(sim, d)
            rows.append(dict(cls=d["cls"], index=d["index"], t_sub=t, orient=args.orient, **{f"{m}_{k}": v for m, ev in e.items() for k, v in ev.items()}))
            print("  " + "  ".join(f"{k}={v:.4g}" for k, v in rows[-1].items() if "mse" in k))
    summary = {}
    for t in args.thicknesses:
        r = [x for x in rows if x["t_sub"] == t]
        if r:
            summary[str(t)] = {k: float(np.mean([x[k] for x in r])) for k in r[0] if k.endswith("mse")}
    best = min(summary, key=lambda k: min(v for kk, v in summary[k].items() if kk.endswith("R_mse"))) if summary else None
    json.dump(dict(rows=rows, summary=summary, best_t_sub=best,
                   designs=[dict(cls=d["cls"], index=d["index"]) for d in designs]),
              open(os.path.join(out, "calibration.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1), f"\nbest substrate thickness: {best} mm")
    _plot_calibration(designs, args.thicknesses, out)


def _plot_calibration(designs, thicknesses, out):
    import utils
    plt = utils.plt
    f_gt = np.linspace(2, 12, 1001)
    fig, axes = plt.subplots(len(designs), 2, figsize=(utils.COL2, 1.6 * len(designs)), squeeze=False, sharex=True)
    for r, d in enumerate(designs):
        for c, (key, gt, lab) in enumerate((("s11", d["R"], r"$|r_{xx}|$"), ("s21", d["T"], r"$|r_{yx}|$"))):
            ax = axes[r, c]
            ax.plot(f_gt, np.abs(gt), color=utils.INK, lw=1.4, label="SUTD dataset")
            for k, t in enumerate(thicknesses):
                p = os.path.join(out, "runs", f"{d['cls']}_{d['index']}_t{t:g}.npz")
                if os.path.exists(p):
                    s = np.load(p)
                    ax.plot(s["freq"], np.abs(s[key]), color=utils.SERIES[k % 8], lw=0.9, ls="--", label=f"CST t={t:g} mm")
            ax.set_ylabel(lab); ax.set_ylim(0, 1.05)
            if r == 0 and c == 0:
                ax.legend(fontsize=5, ncol=2)
        utils.inset_pattern(axes[r, 0], d["matrix"], (0.02, 0.05, 0.2, 0.45))
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (GHz)")
    fig.tight_layout()
    utils.save(fig, os.path.join(out, "calibration.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Verification of optimised designs
# ─────────────────────────────────────────────────────────────────────────────

def verify(args):
    """CST simulation of the best designs of a MOPSO run and the surrogate-vs-CST
    error on each: |r_xx| MSE, mean phase error, and the objective values
    (in-phase band, lowest |r_xx| in it) predicted and simulated."""
    _force_cst_2024()
    import csv
    import utils
    from phase2_mopso import Objectives, inphase_band, surface_phase_deg, fragile_count
    t_sub = args.t_sub or C.CST_SUB_T_MM
    d = np.load(os.path.join(args.run_dir, "Final", "pareto_designs.npz"))
    pats, f_s = d["patterns"], d["freq_ghz"]
    if str(d["layout_convention"]) == "file":
        pats = 1 - pats                      # CST cell: 1 = copper
    specs = C.OBJECTIVE_PRESETS[str(d["problem"])]
    obj = Objectives(specs, f_s)
    f0 = specs[0].get("f0", 10.0)
    order = np.argsort(d["objectives"].sum(1))[:args.top]
    out = os.path.join(args.run_dir, "CST_verification")
    os.makedirs(out, exist_ok=True)
    rows, plt = [], utils.plt
    for k in order:
        tag = f"solution_{k + 1:03d}_t{t_sub:g}"
        print(f"[verify] {tag}", flush=True)
        try:
            s = simulate(pats[k], t_sub, os.path.join(out, "runs", tag + ".npz"))
        except Exception as err:
            print(f"  failed: {err}")
            continue
        Rc = np.interp(f_s, s["freq"], s["s11"].real) + 1j * np.interp(f_s, s["freq"], s["s11"].imag)
        Rp = d["R"][k]
        Vp = obj.values(Rp[None], pats[k][None])[0][0]
        Vc = obj.values(Rc[None], pats[k][None])[0][0]
        lo_p, hi_p, _, _ = inphase_band(Rp[None], f_s, f0)
        lo_c, hi_c, _, _ = inphase_band(Rc[None], f_s, f0)
        band = lambda lo, hi: f"{f_s[lo[0]]:.1f}-{f_s[hi[0]]:.1f}" if lo[0] >= 0 else "none"
        row = dict(solution=int(k + 1), mag_mse=float(np.mean((np.abs(Rc) - np.abs(Rp)) ** 2)),
                   phase_mae_deg=float(np.degrees(np.abs(np.angle(Rc * np.conj(Rp)))).mean()),
                   band_pred_GHz=band(lo_p, hi_p), band_cst_GHz=band(lo_c, hi_c))
        for n, vp, vc in zip(obj.names, Vp, Vc):
            row[f"{n} pred"], row[f"{n} CST"] = float(vp), float(vc)
        rows.append(row)

        fig, axes = plt.subplots(1, 3, figsize=(utils.COL2, 1.9), gridspec_kw=dict(width_ratios=[0.7, 1.4, 1.4]))
        axes[0].imshow(pats[k], cmap=utils.PATTERN_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[1].plot(f_s, 20 * np.log10(np.abs(Rc) + 1e-12), color=utils.SERIES[1], label="CST")
        axes[1].plot(f_s, 20 * np.log10(np.abs(Rp) + 1e-12), color=utils.SERIES[0], ls="--", label="surrogate")
        axes[1].set_ylabel(r"$|r_{xx}|$ (dB)"); axes[1].legend(fontsize=6); axes[1].set_ylim(-30, 1)
        axes[2].plot(f_s, surface_phase_deg(Rc[None], f_s)[0], color=utils.SERIES[1])
        axes[2].plot(f_s, surface_phase_deg(Rp[None], f_s)[0], color=utils.SERIES[0], ls="--")
        for y in (-90, 90):
            axes[2].axhline(y, color=utils.INK2, lw=0.6, ls=":")
        axes[2].set_ylabel("surface phase (deg)"); axes[2].set_ylim(-185, 185)
        for ax in axes[1:]:
            ax.set_xlabel("Frequency (GHz)"); ax.set_xlim(f_s[0], f_s[-1])
        fig.tight_layout()
        utils.save(fig, os.path.join(out, tag + ".png"))
    with open(os.path.join(out, "cst_vs_surrogate.csv"), "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} designs verified -> {out}")


def status(_):
    for p in sorted(glob.glob(os.path.join(C.CST_DIR, "**", "*.npz"), recursive=True)):
        print(os.path.relpath(p, C.CST_DIR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--thicknesses", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    c.add_argument("--n_designs", type=int, default=2)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--n_plg", type=int, default=1)
    c.add_argument("--extra_classes", nargs="*", default=[], help="one test layout of each of these classes")
    c.add_argument("--orient", default="as_is", choices=list(ORIENT),
                   help="layout orientation convention between the dataset matrix and the CST model")
    v = sub.add_parser("verify")
    v.add_argument("--run_dir", required=True)
    v.add_argument("--top", type=int, default=8)
    v.add_argument("--t_sub", type=float, default=None)
    v.add_argument("--orient", default="as_is", choices=list(ORIENT))
    sub.add_parser("status")
    for p_ in (c, v):
        p_.add_argument("--fmin", type=float, default=2.0)
        p_.add_argument("--fmax", type=float, default=12.0)
        p_.add_argument("--grow", type=float, default=0.0, help="patch enlargement (mm)")
    a = ap.parse_args()
    if hasattr(a, "fmin"):
        FREQ[:] = [a.fmin, a.fmax]
        GROW[0] = a.grow
    {"calibrate": calibrate, "verify": verify, "status": status}[a.cmd](a)
