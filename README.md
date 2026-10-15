# SUTD-PRCM graph-attention surrogate + VAE-MOIPSO inverse design

Two-phase pipeline on the SUTD-PRCM dataset (16x16 pixelated, ground-backed
reflective metasurfaces, 2-12 GHz):

* **Phase 1** – forward surrogate: PGA-ResNet backbone (pixel-wise graph
  attention) + frequency cross-attention + BiGRU + passive polar head. Predicts
  the complex co-pol reflection `r_xx` at 101 points. One network (5.83 M
  parameters), trained on one class at a time with equal training sets
  (24,000 layouts for every class, 300 epochs) and tested on every class, as in MSDCNN Table I
  (`benchmark_phase1.py`).
* **Phase 2** – inverse design without a full-wave solver: a VAE of RDN layouts
  (latent d = 150: with the per-element loss, d must exceed 128 or the VAE ignores its input)
  and MOIPSO in its latent space, scored by the frozen RDN surrogate. Problem
  `amc_x`: X-band in-phase reflector (AMC) with three objectives in physical
  units: in-phase bandwidth around 10 GHz (%), lowest |r_xx| in that band (dB),
  number of fragile features (corner-only contacts + isolated pixels).
* **CST** – full-wave verification of the selected designs at the SUTD unit
  cell (`cst_verify.py`): substrate thickness 1.0 mm (CST search over
  0.6-2.2 mm against the stored spectra, `cst_thickness_search.py` and
  `cst_thickness_grid.py`), phase reference 10.705 mm above the cell (from the
  empty cell, `cst_reference.py`).

## Files

| file | purpose |
|---|---|
| `config.py` | every setting (paths, data, model, loss, VAE, MOPSO objectives, CST) |
| `dataset.py` | loader, cached arrays, reproducible per-class splits |
| `phase1_forward_model.py` | surrogate (`resnet3conv_pga`; ablations `resnet3conv`, `plain_cnn`) |
| `phase1_loss.py` | 8-term physics-guided loss, metrics |
| `train_phase1.py` | train / evaluate one surrogate (mirror augmentation, TTA, `--n_train`) |
| `benchmark_phase1.py` | every surrogate tested on every class (MSDCNN Table I format) |
| `phase2_vae.py` | VAE train + diagnostics |
| `phase2_surrogate.py` | co-pol surrogate wrapper, sanity check, SUTD database |
| `phase2_mopso.py` | latent / binary MOIPSO, objectives, Pareto outputs, database baseline |
| `cst_verify.py` | `calibrate` (thickness), `verify` (Pareto designs vs CST), `status` |
| `cst_thickness_search.py`, `cst_thickness_grid.py` | CST substrate-thickness search against the dataset |
| `cst_reference.py` | CST check of a thickness on layouts of every class + phase reference from the empty cell |
| `cell_common.py`, `context.py`, `cst_python_api/` | CST model construction |
| `utils.py` | journal-style plotting |
| `paper/` | `main.tex`, `make_figures.py` (data figures), `schematics.py` + `schematics/` (TikZ schematics), `collect_results.py` |
| `docs/` | objectives note and the earlier paper draft |
| `run_pipeline.bat` | runs every step (`run_pipeline.bat [step]`) |

## Running

```bat
run_pipeline.bat              :: everything
run_pipeline.bat phase1       :: or one step: phase1 vae mopso cst_calibrate cst_verify figures
python train_phase1.py --class_name RDN --epochs 300 --patience 1000 --augment --tta --msdcnn_train 108000 --tag n108000
python phase2_vae.py --mode both --tag rdn_d150
python phase2_mopso.py --swarm 1000 --gens 300 --tag long
python cst_verify.py verify --run_dir outputs_new_pipeline\PHASE2\mopso\amc_x_long --top 6
python paper\schematics.py data:architecture data:pipeline architecture pipeline unitcell
```

## Data

Every sample, 2-12 GHz at 101 points. MSDCNN protocol (`--msdcnn_train N`):
each class is put in a fixed seeded random order, a network trains on the first
N layouts (27,000 per class, 108,000 for RDN108000) with no validation set and
the last of 300 epochs is kept; as in MSDCNN, every network is tested on the
last 3,000 layouts of each class (`MSDCNN_N_TEST`). Layouts use 1 = copper for
every class: the RDN files store the opposite, so RDN layouts are inverted on
loading (`LAYOUT_CONVENTION = "physical"`); every run records its convention
in `run_config.json`.

## Outputs

```
outputs_new_pipeline/
  COPOL/<CLASS>_n27000/           PLG27000, PLR27000, PTN27000, RDN27000 (MSDCNN protocol)
  COPOL/RDN_n108000/              RDN108000, the inverse-design surrogate
  COPOL/RDN_n108000_nopga/        PGA ablation (RDN108000 without PGA blocks)
  benchmark/cross_class.json      MSDCNN-style cross-class table
  PHASE2/vae_rdn_d150/            design VAE (vae_rdn: collapsed d = 64 control)
  PHASE2/mopso/amc_x_<tag>/       MOIPSO runs, Pareto designs, CST_verification/
  CST/calibration/  CST/reference/  CST/workspaces/
```
