"""
fr4_config.py — every setting of the FR-4 16x16 RDN dataset in one place.

Everything matches the SUTD-PRCM unit cell (Zhang et al., IEEE AWPL 2022, and
the CST model of E:\\SUTD_BENCHMARK_NOVEL_GNN, which reproduces the SUTD spectra
to |r| MSE ~1e-6) EXCEPT the substrate material, which is lossy FR-4.

Layouts use 1 = copper, 0 = no copper (the physical convention of the
surrogate; unlike the SUTD RDN files, nothing is inverted here).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # E:\SUTD_BENCHMARK_NOVEL_GNN: cell_common.py, context.py, cst_python_api
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── Designs (RDN class: every pixel copper with probability 0.5) ──────────
GRID_SIZE   = 16
N_DESIGNS   = 50_000          # distinct designs stored; simulate the first N_SIMULATE
N_SIMULATE  = 30_000          # target for now (indices 0 .. N_SIMULATE-1)
FILL_PROB   = 0.5             # SUTD RDN: mean fill 0.4995, per-design std 0.031
DESIGN_SEED = 2026
DESIGNS_FILE = os.path.join(HERE, "designs_rdn16_fr4_50k.npy")
DESIGNS_META = os.path.join(HERE, "designs_rdn16_fr4_50k.json")

# ─── Unit cell (SUTD-PRCM) ─────────────────────────────────────────────────
PATCH_SIZE_MM   = 0.5         # pixel pitch
PADDING_MM      = 1.0         # per side -> 10 mm unit cell
PATCH_T_MM      = 0.018       # copper patch thickness
GROUND_T_MM     = 0.18        # copper ground plane
CONDUCTOR       = "copper"    # sigma = 5.8e7 S/m (SUTD-matching model; not PEC)

# ─── Substrate: the one deliberate change from SUTD-PRCM ───────────────────
SUB_NAME  = "FR-4 (lossy)"
SUB_EPS   = 4.3               # CST library FR-4 (lossy)
SUB_TAND  = 0.025
SUB_T_MM  = 1.0               # SUTD-PRCM thickness (also a standard FR-4 board)

# ─── Sweep (SUTD-PRCM): 2-12 GHz, 1001 points, two Floquet modes ───────────
FREQ_MIN, FREQ_MAX, FREQ_SAMPLES = 2.0, 12.0, 1001

# ─── Output ────────────────────────────────────────────────────────────────
TAG         = f"fr4_t{SUB_T_MM:g}mm"
OUT_DIR     = os.path.join(HERE, f"samples_{TAG}")        # one .npz per design
LOCK_DIR    = os.path.join(OUT_DIR, "_locks")             # claims of running workers
LOG_DIR     = os.path.join(HERE, "logs")
PACKED_DIR  = os.path.join(HERE, f"packed_{TAG}")         # SUTD-format export
WORKSPACE_ROOT = os.path.join(HERE, "cst_workspaces")
