"""
collect_results.py — gather every number reported in the manuscript into
paper/results.json and print them grouped by table.

    python paper/collect_results.py
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C

OUT = {}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def phase1():
    res = {}
    for d in sorted(glob.glob(os.path.join(C.OUTPUT_ROOT, "COPOL", "*")) + glob.glob(os.path.join(C.OUTPUT_ROOT, "XPOL", "*"))):
        m = load(os.path.join(d, "phase1_metrics.json"))
        rc = load(os.path.join(d, "run_config.json"))
        if not m:
            continue
        key = os.path.relpath(d, C.OUTPUT_ROOT).replace("\\", "/")
        t = m["test"]
        res[key] = dict(params=m["n_params"], epochs_run=rc.get("epochs_run") if rc else None,
                        minutes=round(rc.get("minutes", 0), 1) if rc else None,
                        test_lin_mse=t["lin_mse"], test_r2=t["r2"], test_db_mse=t["db_mse"],
                        phase_mae_deg=t.get("phase_mae_deg"), band_edge_db=t.get("band_edge_abs_db"),
                        msdcnn_full=m["msdcnn_protocol"]["full_band"]["mag"]["lin_mse"],
                        msdcnn_plg_band=m["msdcnn_protocol"]["plg_band"]["mag"]["lin_mse"],
                        per_class=m.get("per_class_test"))
    OUT["phase1"] = res
    print("\n== Phase 1 (test split) ==")
    for k, v in res.items():
        print(f"{k:22s} params={v['params']:>9,}  lin={v['test_lin_mse']:.3e}  R2={v['test_r2']:.4f}  "
              f"dB={v['test_db_mse']:.3f}  phase={v['phase_mae_deg']:.2f}deg  MSDCNN-style={v['msdcnn_full']:.3e} "
              f"(9.5-12GHz {v['msdcnn_plg_band']:.3e})  ep={v['epochs_run']} {v['minutes']}min")


def vae():
    res = {}
    for d in sorted(glob.glob(os.path.join(C.OUTPUT_ROOT, "PHASE2", "vae*"))):
        dg, cf = load(os.path.join(d, "vae_diagnostics.json")), load(os.path.join(d, "vae_config.json"))
        if dg:
            res[os.path.basename(d)] = dict(classes=cf["classes"], latent=cf["latent_dim"], beta=cf.get("beta"),
                                            pix=dg["pixel_accuracy_mean"], exact=dg["exact_reconstruction_frac"],
                                            active=dg["active_latent_dims"],
                                            nbr=(dg["stats_data"]["neighbour_agreement"], dg["stats_prior_samples"]["neighbour_agreement"]),
                                            novelty=dg.get("prior_min_hamming_to_train_median"))
    OUT["vae"] = res
    print("\n== VAE ==")
    for k, v in res.items():
        print(f"{k:16s} {v}")


def mopso():
    """MOIPSO runs: hypervolume, best objective values, database baseline."""
    res = {}
    for d in sorted(glob.glob(os.path.join(C.OUTPUT_ROOT, "PHASE2", "mopso", "*"))):
        cp = load(os.path.join(d, "comparison.json"))
        if cp:
            res[os.path.basename(d)] = dict(seconds=round(cp["seconds"], 1), evals=cp["evaluations"],
                                            archive=cp["archive"], meeting_spec=cp["n_good"],
                                            best=dict(zip(cp["objective_names"], cp["best_value"])),
                                            hv=cp["hv"], hv_database=cp["database"]["hv_true"],
                                            hv_database_surrogate=cp["database"]["hv_surrogate"])
    OUT["mopso"] = res
    print("\n== MOPSO ==")
    for k, v in res.items():
        print(f"{k:16s} t={v['seconds']}s evals={v['evals']} meeting spec={v['meeting_spec']} HV={v['hv']:.3f} "
              f"(database {v['hv_database']:.3f}) best: " + ", ".join(f"{n} = {x:.3g}" for n, x in v["best"].items()))


def cst():
    OUT["cst_calibration"] = load(os.path.join(C.CST_DIR, "calibration", "calibration.json"))
    ver = {}
    for p in glob.glob(os.path.join(C.OUTPUT_ROOT, "PHASE2", "mopso", "*", "CST_verification", "cst_vs_surrogate.csv")):
        ver[p.split(os.sep)[-3]] = open(p).read()
    OUT["cst_verification"] = ver


if __name__ == "__main__":
    for f in (phase1, vae, mopso, cst):
        f()
    json.dump(OUT, open(os.path.join(ROOT, "paper", "results.json"), "w"), indent=1, default=str)
    print(f"\n-> paper/results.json")
