"""
compare_with_repo.py — compare the JSON results re-generated on this machine with the versions committed
in sutd-prcm-solverfree (git HEAD), number by number.

    python experiments/verify_solverfree/compare_with_repo.py fdtd_checks/validation.json data/baseline_cp_centered.json ...
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF = os.path.join(ROOT, "external", "sutd-prcm-solverfree")


def flat(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{pre}[{i}]")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield pre, float(d)


for rel in sys.argv[1:]:
    try:
        old = json.loads(subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=SF, capture_output=True, text=True,
                                        check=True).stdout)
    except subprocess.CalledProcessError:
        print(f"{rel}: not in git"); continue
    new = json.load(open(os.path.join(SF, rel)))
    a, b = dict(flat(old)), dict(flat(new))
    keys = [k for k in a if k in b]
    rel_err = [abs(b[k] - a[k]) / max(abs(a[k]), 1e-12) for k in keys]
    worst = sorted(zip(rel_err, keys), reverse=True)[:3]
    print(f"{rel}: {len(keys)} numbers, max rel. diff {max(rel_err) if rel_err else 0:.2e}; "
          f"{sum(e < 1e-3 for e in rel_err)} within 0.1 %; worst " +
          "; ".join(f"{k}: {a[k]:.4g} -> {b[k]:.4g}" for _, k in worst))
