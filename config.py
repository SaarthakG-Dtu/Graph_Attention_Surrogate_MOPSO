"""
config.py — single source of configuration for the SUTD-PRCM pipeline.

  Phase 1 : PGA-ResNet + frequency cross-attention forward surrogate
            (layout 16x16 -> complex reflection at K=101 points, 2-12 GHz)
  Phase 2 : VAE design prior + latent-space MOPSO, scored by the frozen
            Phase 1 surrogate (no full-wave solver in the loop)
  CST     : full-wave verification of selected designs at the SUTD geometry

Everything a script needs is read from here; nothing is monkey-patched.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "SUTD_DATASET")
OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "outputs_new_pipeline")
CACHE_DIR    = os.path.join(DATA_ROOT, "cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def be_nice(threads: int = 2):
    """Cap CPU threads and lower process priority, so GPU jobs leave the CPU to
    CST simulations running on the same machine."""
    torch.set_num_threads(threads)
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (SUTD-PRCM)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE         = 16
FREQ_RANGE       = (2.0, 12.0)      # GHz
FREQ_POINTS_FULL = 1001
DOWNSAMPLE_TO    = 101              # every 10th point -> 0.1 GHz grid
CLASSES          = ["PLG", "PLR", "PTN", "RDN"]
POOLED_CLASS     = "ALL"            # union of the four classes (per-class splits kept)

RANDOM_SEED = 42

# Data protocol: every sample of a class, 2-12 GHz at 101 points (every 10th
# CST point, as in the edge-aware surrogate), seeded random 80/10/10 split per
# class. The surrogates use the MSDCNN protocol (dataset.msdcnn_split_indices).
N_POINTS = 101
SPLIT    = (0.80, 0.10, 0.10)
MSDCNN_N_TEST = 3000                    # MSDCNN protocol: test on the last 3,000 layouts of each class
MSDCNN_PLG_BAND = (9.5, 12.0)   # MSDCNN reports PLG on this sub-band only

# Target channel of the surrogate. "co": x->x reflection R (dataset channel 0),
# "cross": x->y reflection T (dataset channel 1). co/cross models share
# identical splits.
# Layout convention. The SUTD-PRCM RDN files store copper as 0, the other
# classes as 1 (established by CST: the inverted RDN layout reproduces the
# stored spectra, the layout as stored does not). With "physical", RDN layouts
# are inverted on loading so that 1 = copper for every class. Each run records
# the convention it was trained with (run_config.json).
LAYOUT_CONVENTION = "physical"          # "physical" | "file"
FILE_INVERTED_CLASSES = ["RDN"]

TARGET_CHANNELS = {"co": 0, "cross": 1}
OUTPUT_SUBDIR   = {"co": "COPOL", "cross": "XPOL"}


# Reported per-class runs, trained as in MSDCNN: 27,000 training layouts per class
# (RDN also 108,000), no validation set, last of 300 epochs; mirror augmentation + TTA.
P1_REPORTED_TAG = {c: "n27000" for c in ["PLG", "PLR", "PTN", "RDN"]}


def phase1_dir(class_name: str, target: str = "co", tag: str = "") -> str:
    """One Phase 1 run: OUTPUT_ROOT/<COPOL|XPOL>/<CLASS>[_tag]."""
    name = class_name + (f"_{tag}" if tag else "")
    return os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIR[target], name)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — forward surrogate architecture
# ─────────────────────────────────────────────────────────────────────────────
# Presets. "resnet3conv_pga" is the model used for every reported class.
# "resnet3conv" is the same network with all PGA blocks removed (ablation),
# "plain_cnn" is a conv/maxpool encoder with the same head (baseline).
P1_MODEL_PRESETS = {
    "resnet3conv_pga": dict(planes=[64, 128, 256], blocks=[3, 4, 3],
                            strides=[1, 2, 2], pga=[True, True, True]),
    "resnet3conv":     dict(planes=[64, 128, 256], blocks=[3, 4, 3],
                            strides=[1, 2, 2], pga=[False, False, False]),
    "plain_cnn":       dict(stem_ch=32, channels=[64, 256], final_ch=256),
}
P1_MODEL_TYPE = "resnet3conv_pga"

P1_STEM_ACT    = "tanh"   # stem activation: "tanh" | "relu"
P1_LEAKY_SLOPE = 0.2      # LeakyReLU slope inside BasicBlock

# PGA (pixel-wise graph attention) block
P1_PGA_LAYERS     = 3     # PGA blocks after every ResNet stage
P1_PGA_NEIGHBOR   = 8     # 8-neighbour grid adjacency
P1_PGA_NHEADS     = 1
P1_PGA_DOWN_RATIO = 8     # theta/phi bottleneck C -> C/8

# Frequency cross-attention head
P1_FREQ_QUERIES      = DOWNSAMPLE_TO
P1_CROSS_ATTN_HEADS  = 4
P1_CROSS_ATTN_LAYERS = 1

P1_DROPOUT_BACKBONE = 0.20   # Dropout2d on the backbone output
P1_DROPOUT_HEAD     = 0.20   # Dropout on the BiGRU output

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — loss and training
# ─────────────────────────────────────────────────────────────────────────────
P1_W_REIM    = 1.00
P1_W_MAG_DB  = 0.50
P1_W_MAG_LIN = 0.30
P1_W_PHASE   = 0.30
P1_W_PASS    = 0.08
P1_W_TV      = 0.06
P1_W_CURV    = 0.02
P1_W_KK      = 0.02

# Target-specific loss options (see phase1_loss.physics_loss_terms). The
# cross-pol response is ~0 over much of the band, so its dB term is floored
# and its phase term is weighted by the target magnitude.
P1_LOSS_OPTIONS = {"co":    dict(db_floor=None,  phase_mag_ref=None),
                   "cross": dict(db_floor=-40.0, phase_mag_ref=0.1)}

P1_BATCH_SIZE     = 256
P1_EPOCHS         = 500
P1_LR             = 2e-4
P1_WEIGHT_DECAY   = 1e-5
P1_GRAD_CLIP      = 1.0
P1_EARLY_STOP_PAT = 25
P1_CKPT_CRITERION = {"co": "db_mse", "cross": "lin_mse"}   # validation metric for the best checkpoint

BAND_EDGE_FRAC = 0.10            # band-edge |dB| error metric uses 10% each side
EVAL_N_PLOT    = 12              # sample panels in evaluation figures

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — VAE design space (port of the 12x12 conv_symmetric VAE)
# ─────────────────────────────────────────────────────────────────────────────
# Trained on the RDN layouts (Phase 1 splits). RDN pixels are i.i.d. fair coins
# and cannot be compressed: with mean-reduced losses the effective KL weight
# is beta * 256 / (W_BCE * d), and the VAE collapses to the prior when it
# exceeds 1 (d = 64: 2.0). d = 150 keeps the ratio of the 12x12 VAE
# (144 pixels, d = 84: 0.86).
VAE_TRAIN_CLASSES        = ["RDN"]
VAE_LATENT_DIM           = 150
VAE_CONV_CHANNELS        = [32, 64, 128]      # 16x16 -> 8x8 -> 4x4 and back
VAE_FINAL_ACTIVATION     = "sigmoid"          # "sigmoid" | "st" (0.5(1+tanh((x-0.5)m)))
VAE_ST_M                 = (1.0, 15.0, 50)    # ST sharpness start, end, anneal epochs
# total = W_BCE * BCE(mean) + beta * KL(mean over dims) + GAMMA * mean p(1-p)
VAE_W_BCE                = 2.0
VAE_BETA                 = 1.0
VAE_GAMMA                = 0.0
VAE_KL_FREE_EPOCHS       = 5
VAE_KL_ANNEAL_EPOCHS     = 30
VAE_KL_ANNEAL_SCHEDULE   = "cosine"           # "linear" | "cosine"
VAE_AUGMENT              = True               # random D4 rotation / flip
VAE_BATCH_SIZE           = 512
VAE_EPOCHS               = 300
VAE_LR, VAE_LR_MIN       = 1e-3, 1e-5
VAE_WEIGHT_DECAY         = 0.0
VAE_GRAD_CLIP            = 5.0
VAE_EARLY_STOP           = 30


def vae_dir(tag: str = "") -> str:
    return os.path.join(OUTPUT_ROOT, "PHASE2", "vae" + (f"_{tag}" if tag else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — VAE-latent MOIPSO scored by the frozen surrogate (no CST in the loop)
# ─────────────────────────────────────────────────────────────────────────────
# Co-pol surrogate ("CLASS[:tag]" of a Phase 1 run) and design VAE.
MOPSO_SURROGATE   = "RDN:n108000"    # RDN108000 of the MSDCNN comparison
MOPSO_TRUST       = 0.0              # max. disagreement with the checker surrogates (median on RDN test); 0 = off
MOPSO_TRUST_CHECKERS = ["RDN:n108000_nopga", "RDN:n27000"]
MOPSO_VAE_TAG     = "rdn_d200"
MOPSO_SEARCH_SPACE = "latent"     # "latent" (VAE) | "binary" (256 pixels, threshold 0.5)
MOPSO_NP          = 200           # swarm size
MOPSO_NR          = 50            # Pareto archive capacity
MOPSO_MAXGEN      = 150
MOPSO_W_START     = 0.9           # inertia, linear decay
MOPSO_W_END       = 0.4
MOPSO_U_MUT       = 0.5           # fraction of the swarm mutated per generation
LATENT_BOUND      = 1.5           # latent box [-B, B]^d
LATENT_MAXVEL     = 0.2           # per latent component
PIXEL_MAXVEL      = 0.25 / 6      # pixel space (positions in [0, 1]), best of the sweep
MOPSO_INIT        = "prior"       # "prior": z ~ N(0, I) clipped; "uniform"
MOPSO_SEED        = 0
MOPSO_SAVE_EVERY  = 25

# The SUTD-PRCM phase (and that of our CST model) is referenced to the Floquet
# port plane, PHASE_REF_SHIFT_MM above the top of the cell. Phase objectives
# use the phase moved to the cell surface: r_surface = r * exp(+j 2 k shift).
PHASE_REF_SHIFT_MM = 10.705     # from a CST simulation of the empty cell (bare grounded slab)

# Objective presets. Each objective has a value in physical units, an ideal
# and a worst value; the optimiser minimises the normalised penalty
# (value - ideal) / (worst - ideal) (0 = ideal, 1 = worst; hypervolume
# reference point 1). "good" is the value a design must reach to meet the spec.
# kinds (co-pol r_xx only):
#   inphase_fbw : fractional bandwidth (%) of the band around f0 where the
#                 surface reflection phase stays within +-90 deg (in-phase / AMC)
#   inband_db   : lowest |r_xx| (dB) inside that band (loss / unwanted resonances)
#   fragile     : number of fragile features (diagonal-only contacts + isolated pixels)
OBJECTIVE_PRESETS = {
    "amc_x": [                 # in-phase (AMC) reflector centred in X-band
        dict(name="In-phase bandwidth (%)", kind="inphase_fbw", f0=10.0, ideal=30.0, worst=0.0, good=10.0),
        dict(name="Lowest |r| in band (dB)", kind="inband_db", f0=10.0, ideal=0.0, worst=-20.0, good=-1.0),
        dict(name="Fragile features", kind="fragile", ideal=0.0, worst=60.0, good=5.0),
    ],
}
MOPSO_PROBLEM = "amc_x"


def mopso_dir(problem: str, tag: str = "") -> str:
    return os.path.join(OUTPUT_ROOT, "PHASE2", "mopso", problem + (f"_{tag}" if tag else ""))


# ─────────────────────────────────────────────────────────────────────────────
# CST full-wave verification (SUTD-PRCM unit cell, Zhang et al. 2021/2022)
# ─────────────────────────────────────────────────────────────────────────────
CST_GRID_SIZE        = 16
CST_PATCH_SIZE_MM    = 0.5      # pixel pitch
CST_PADDING_MM       = 1.0      # per side -> 10 mm unit cell
CST_PATCH_T_MM       = 0.018    # copper patch thickness
CST_GROUND_T_MM      = 0.18     # copper ground plane
CST_SUB_EPS          = 2.65
CST_SUB_TAND         = 0.003
CST_CONDUCTOR        = "copper"
# Substrate thickness is not stated in the SUTD-PRCM papers. CST search over
# 0.6-2.2 mm (cst_thickness_search.py, cst_thickness_grid.py) against the stored
# spectra of 6 layouts of all classes: 1.0 mm gives the lowest mean |r_xx| MSE
# (1.5e-3; PLG and PLR ~1e-6, phase within 0.1 deg). The phase reference
# (cst_reference.py) is 10.707 mm at 1.0 mm, the same as at 0.8 mm.
CST_SUB_T_MM         = 1.0
CST_DIR              = os.path.join(OUTPUT_ROOT, "CST")
