# Supplementary material: settings needed to reproduce the results

These are the settings not given in the paper. Each one was checked against the code and the run files. File names
refer to the code release (repository root unless stated).

## 1. Data
- **Source.** SUTD-PRCM, every tenth frequency (2.0, 2.1, ..., 12.0 GHz, K = 101); `dataset.py`.
- **Layouts.** 1 = copper, incident field along the image rows, as returned by `dataset.to_convention(..., "true")`.
- **Splits (MSDCNN protocol).** Each class has a fixed random order (seed 42); a network trained on N layouts uses the
  first N of it, and every network of a class is tested on the last 3,000 (`dataset.msdcnn_split_indices`). No
  validation set; the network after the last epoch is used.
- **Dataset baselines.** All 111,000 RDN layouts whose total reflected power `|r_xx|^2 + |r_yx|^2` stays below 1.01 at
  every frequency (110,843 layouts).

## 2. Surrogate (`phase1_forward_model.py`, `phase1_loss.py`, `train_phase1.py`)
- **Seed.** 42.
- **Encoder.** Stem, stages and PGA blocks as in Sec. IV. Channel dropout (`Dropout2d`, p = 0.2) on the 4 x 4 x 256
  map; the 16 tokens pass through layer normalization.
- **Frequency queries.** A learned query (256) per frequency plus the sinusoidal code
  `PE[k, 2i] = sin(k / 10000^(2i/256))`, `PE[k, 2i+1] = cos(...)`.
- **Attention.** `nn.MultiheadAttention(256, 4, dropout 0.1)` on the layer-normalized query, followed by layer
  normalization and a linear layer (one layer, no residual connections).
- **GRU.** A single-layer bidirectional GRU with 256 units per direction.
- **Output.** Dropout (p = 0.2), then a linear layer 512 -> 4 (a, c, b, b').
- **Loss.** Terms and weights of Table II. The dB term of r_xx uses `sqrt(Re^2 + Im^2 + 1e-7)`; that of r_yx clamps
  both magnitudes at -40 dB; the phase term of r_yx is weighted by `min(1, |S| / 0.1)`. The passivity term acts on the
  total power once.
- **Training.** `python train_phase1.py --class_name <C> --target both --convention true --msdcnn_train <N>
  --epochs 300 --patience 1000 --augment --tta --tag n<N>`. AdamW (lr 2e-4, weight decay 1e-5), batch 256, gradient
  clipping at 1.0, cosine schedule per epoch from 2e-4 to 2e-6, no warm-up, single precision. Each mirror (x and y) is
  applied per layout with probability 0.5; r_yx changes sign when exactly one is applied.
- **Test-time averaging.** Over the four mirror images; magnitudes and unit phasors are averaged separately.
- **PGA ablation.** The same command with `--model_type resnet3conv` (no PGA blocks).

## 3. VAE (`phase2_vae.py`)
`python phase2_vae.py --mode both --latent_dim 200 --tag rdn_d200` (and `--latent_dim 64 --tag rdn_d64`), RDN
layouts only, true layout orientation. Pixel accuracy on the RDN test layouts: 1.000 (d = 200, stopped at epoch 79)
and 0.501 (d = 64, stopped at epoch 211).

## 4. Objectives (`phase2_devices.py`; edge cases)
- **Band edges.** Interpolated linearly between grid points (in dB for the coding pair). A band stops at the ends of
  the grid.
- **Band quality.** Averaged over the grid points inside the band: mean AR in dB for the converter, mean `|r_yx|^2`
  for the rotator, mean reduction for the coding pair.
- **Caps.** AR is capped at 40 dB; the reduction is capped at 30 dB (`s >= 1e-3`).
- **No band at f0.** The quality is taken at f0. For the coding pair, the largest reduction is used.
- **D.** Corner contacts are counted as 2 x 2 windows with exactly one diagonal pair of copper pixels. An isolated pixel
  is a copper pixel with no copper among its 4 edge neighbours. For a coding pair, D is the sum over both cells.

## 5. MOIPSO (`phase2_mopso.py --problem cp | rotator | coding_diag | coding_mirror | coding_free`)
- **Initialization.** Pixel particles uniform in [0, 1]^D; VAE particles z ~ N(0, I), clipped to [-1.5, 1.5]^200.
  Velocities start at zero.
- **Velocity and acceleration.** After each move, the velocity is clipped per component to +-0.25/6 (pixels) or +-0.2
  (VAE). c1 and c2 are redrawn every generation (one rho1, rho2 per generation).
- **Mutation.** Each particle is mutated with probability 0.5: Gaussian, sigma = max(0.02, 0.1 (1 - t/T)) x range.
- **Archive and leaders.** 50 members, one per distinct layout; 7 grid divisions per objective spanning the archive's
  current range; a box is drawn with probability proportional to 1/occupancy and a leader at random in it; when full,
  the members with the smallest crowding distance are removed.
- **Personal best.** Replaced when the new position dominates it, and with probability 0.5 when neither dominates.
- **Four-mirror average.** Every prediction in the loop uses the test-time averaging of Sec. 2.
- **Seeds.** 0, 1, 2 for the three standard runs; 0 for the long runs and the coding runs.
- **Hypervolume.** Monte Carlo with reference point 1: 200,000 points at the end of a run, 20,000 per generation.

## 6. Selection (`experiments/devices/verify_designs.py`, `verify_coding.py`)
- **Converter and rotator.**
  1. The 6 designs of the best-HV run with the smallest penalty sum.
  2. Then, from all runs, the designs predicted to meet the search target (symmetric band, AR < 2.5 dB or PCR > 0.93),
     sorted by penalty sum.
  3. Duplicates are removed, and at most 12 designs are kept.
- **Coding.** The non-dominated pairs of all runs; the 3 widest predicted bands with D <= 10, then filled to 6 pairs by
  penalty sum. The board pair is pair 0 (widest predicted band).
- **Coding library baseline** (`experiments/devices/coding_baseline.py`). Mirror pairs: all of them, scored exactly.
  General pairs: for each cell A, the 16 partners nearest to -A, by Euclidean distance on Re/Im of both channels at
  every second frequency, in each of four windows (4-12, 6-12, 8-12, 5-10 GHz); all 7.1 M candidates are then scored
  exactly with (9) for x-incidence.

## 7. FDTD (`fdtd_solver/`)
- **Geometry.** Air is 5 mm from the cell surface to the source, 2 mm from the source to the monitor, and 2 mm from the
  monitor to the PML.
- **PML.** UPML with 12 cells, cubic grading, R0 = 1e-8. For boards, CPML with 20 cells: sigma_max = 0.8 (m+1)/(eta0 h),
  m = 3, kappa_max = 5, alpha falling linearly from 2 pi 3 GHz eps0 at the interface.
- **Source.** A soft Ex source: a Gaussian-modulated sine with centre 7 GHz, sigma_t = 1/(2 pi 2.33 GHz), and
  t0 = 6 sigma_t.
- **Reflection.** The plane average of the field at the monitor, minus the incident field of a 1-D FDTD run with the same
  h and dt. The phase is moved with the numerical wavenumber and then to the port at Delta = 10.705 mm.
- **Stopping.** Every 2000 steps after 3 ns, once the largest |signal| over the last 2000 steps is below 1e-4 of the
  co-polarized peak, for both channels; at most 30 ns for unit cells and 20 ns for boards. Board spectra use the first
  15 ns with a Hann taper over its last quarter.
- **Copper.** Pixels: sheets with conductivity 1/(R_s h) + sigma_sub/2 in the averaged permittivity, updated with
  exponential time stepping. Ground: Ex(0) = -R_s Hy(h/2), Ey(0) = R_s Hx(h/2). R_s = 0.026 Ohm, constant over frequency.
- **Mesh extrapolation.** The scale s minimizes the MSE of |r_xx| and |r_yx| between the 0.125 mm spectrum and the
  warped 0.25 mm spectrum (s = 0.97 to 1.08 in steps of 5e-4, Re/Im interpolated linearly); delta_0.125 = (s - 1)/3.
  Points of the warped spectrum that fall outside the grid (near 2 GHz) take the 0.125 mm values.
- **Tolerance study** (`experiments/devices/mc_tolerance.py`). Material parameters drawn with `numpy` seed 2026; the
  corner-contact draws are keyed by (seed 7, physical cell, contact position), so a cell and its transpose open the same
  contacts. Converter design 5, rotator designs 5 and 2 (`rotator=2`), coding pair 0.

## 8. CST (`cst_verify.py`, `cell_common.py`, `experiments/devices/cst_designs.py`, `cst_coding.py`)
- CST Studio Suite 2024, frequency-domain solver, 2-12 GHz; unit-cell boundaries in x and y, Floquet ports with two
  modes; geometry and materials of Sec. II, substrate 1.0 mm; copper pixels and ground.
- r_xx is S(mode 1 -> mode 1). r_yx is S(mode 2 -> mode 1) multiplied by `config.CST_RYX_SIGN` (-1 for layouts placed
  as in the dataset), which reproduces the dataset's r_yx of PLG and PLR cells to an MSE of about 1e-5.
- For coding pairs B = mirror(A), so only A and A^T are simulated (A alone for the diagonal pairs).

## 9. Software and hardware
- Python 3.11.9, PyTorch 2.13.0 (CUDA 13.0), NumPy 2.4.6, CST Studio Suite 2024.
- One NVIDIA RTX PRO 4000 (24 GB) for training, searches and FDTD, shared between jobs.
- Figures: `paper/figtools/*.py` (data and model figures, schematic data); schematics compiled by
  `paper/figtools/make_schematics.py` into `paper/schematics_v2/`.
