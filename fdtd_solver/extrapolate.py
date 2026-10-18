"""Grid-converged FDTD spectra by frequency-warp Richardson extrapolation.

The Yee grid's numerical dispersion slows waves, which mainly shifts every resonance down in frequency by a
fraction delta(h) = c h^2 (second order). Model: S_h(f) = S_0(f (1 + delta_h)). Two grids (h, h/2) give
S_h(f) ~ S_{h/2}(s f) with s = 1 + delta_h - delta_{h/2} = 1 + 3 delta_{h/2}, so the converged spectrum is
S_0(f) = S_{h/2}(f / (1 + delta_{h/2})), delta_{h/2} = (s - 1) / 3.

  python -I extrapolate.py check            the 8 CST cells of fdtd_checks (validation h=0.25, refine h=0.125)
  python -I extrapolate.py search/<tag>     designs of a verified search, against the surrogate
"""
import sys, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).parent
SCALES = np.arange(0.97, 1.08001, 0.0005)


def warp(S, F, s):
    """S(s f) on the grid F (complex, linear interpolation of Re/Im; NaN outside the band)."""
    g = F * s
    out = np.interp(g, F, S.real, left=np.nan, right=np.nan) + 1j * np.interp(g, F, S.imag, left=np.nan, right=np.nan)
    return out


def fit_scale(Sc, Sf, F):
    """s minimising |S_coarse(f) - S_fine(s f)|^2 over both channels (magnitudes, to be insensitive to the
    port-reference phase), f restricted to where both are defined."""
    err = []
    for s in SCALES:
        w = np.stack([np.abs(warp(Sf[k], F, s)) for k in range(2)])
        m = np.isfinite(w).all(0)
        err.append(np.mean((w[:, m] - np.abs(Sc[:, m])) ** 2))
    return float(SCALES[int(np.argmin(err))])


def converged(Sc, Sf, F):
    s = fit_scale(Sc, Sf, F)
    d = (s - 1) / 3
    S0 = np.stack([warp(Sf[k], F, 1 / (1 + d)) for k in range(2)])
    return S0, s, d


def mse_abs(a, b):
    m = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    return float(np.mean((np.abs(a[:, m]) - np.abs(b[:, m])) ** 2))


if __name__ == "__main__":
    tag = sys.argv[1]
    if tag == "check":
        from freq_shift_data import cst_curve                      # noqa: E402  (helper below writes it)
        D = ROOT / "fdtd_checks"; F = np.linspace(2, 12, 1001)
        zc, zf = np.load(D / "validation.npz"), np.load(D / "refine.npz")
        conv = json.loads((D / "convention.json").read_text())
        rows = []
        for j, who in enumerate(zf["who"]):
            one = bool(zf["copper_is_one"][j]); cls = who.split()[0]
            if one != conv["copper_is_one_by_class"][cls]:
                continue
            jc = [i for i, w in enumerate(zc["who"]) if w == who and bool(zc["copper_is_one"][i]) == one][0]
            Sc = np.stack([zc["rxx"][jc], zc["ryx"][jc]]); Sf = np.stack([zf["rxx"][j], zf["ryx"][j]])
            cst = cst_curve(who)
            if np.abs(cst[0]).min() > 0.9:
                continue                                            # no resonance in band: nothing to shift
            S0, s, d = converged(Sc, Sf, F)
            rows.append(dict(who=who, delta_fine_pct=100 * d, mse_coarse=mse_abs(Sc, cst), mse_fine=mse_abs(Sf, cst),
                             mse_extrapolated=mse_abs(S0, cst)))
            r = rows[-1]
            print(f"{who:22s} delta(h=0.125) {r['delta_fine_pct']:+.2f} %   |S| MSE vs CST: coarse {r['mse_coarse']:.2e}  "
                  f"fine {r['mse_fine']:.2e}  extrapolated {r['mse_extrapolated']:.2e}")
        (D / "extrapolate_check.json").write_text(json.dumps(rows, indent=1))
    else:
        sfx = sys.argv[2] if len(sys.argv) > 2 else ""          # "_cu": the copper-loss verification
        S = ROOT / tag; z = np.load(S / f"verify_spectra{sfx}.npz"); F = z["freq"] / 1e9
        S0s, rows = [], []
        for i in range(len(z["pred"])):
            S0, s, d = converged(z["fdtd_h250um"][i], z["fdtd_h125um"][i], F)
            S0s.append(S0)
            rows.append(dict(design=i, delta_fine_pct=100 * d, mse_fine_vs_pred=mse_abs(z["fdtd_h125um"][i], z["pred"][i]),
                             mse_extrap_vs_pred=mse_abs(S0, z["pred"][i])))
            r = rows[-1]
            print(f"design {i:2d}  delta(h=0.125) {r['delta_fine_pct']:+.2f} %   |S| MSE vs surrogate: fine {r['mse_fine_vs_pred']:.2e}  "
                  f"extrapolated {r['mse_extrap_vs_pred']:.2e}")
        np.save(S / f"fdtd_extrapolated{sfx}.npy", np.stack(S0s))
        (S / f"extrapolate{sfx}.json").write_text(json.dumps(rows, indent=1))
