# Re-running sutd-prcm-solverfree on this machine (17-19 Oct 2026)

Clone: `external/sutd-prcm-solverfree` (commit c1e8a6f), paths adapted, run on an NVIDIA RTX PRO 4000. For internal use;
none of the dataset findings below go into the paper.

## Reproduced

| Claim / result | Repo value | Here | Evidence |
|---|---|---|---|
| Training data counts (`data.py`, `data_all.py`) | 104,252 / 245,052 | same (after using `data/quality_untrustworthy_rows.json` as the quality list) | `logs/data*.log` in the clone |
| FDTD solver checks (`test_fdtd_mc.py`) | slab 6.6e-4 / 5.6e-4, batch < 1e-7 | same to 1e-8 | `fdtd_checks/fdtd_mc_check.json` |
| Empty cell vs analytic slab (`test_fdtd_slab.py`) | 6e-4 (0.25 mm) | 1e-4 (0.125 mm), as stated | `logs/verify_chain.log` |
| 96 dataset cells, FDTD vs CST, all copper lossy | MSE 1.11e-3; min. reflected power 0.8e-3 (RDN 5e-3) | 1.105e-3; 0.76e-3 (RDN 5.4e-3) | `fdtd_checks/copper_loss.json` |
| Corner contacts must conduct (`test_corner_and_fix.py`) | open corners better on 1 of 24 | same | `fdtd_checks/corner_and_fix.json` |
| Frequency shift (`freq_shift.py`) | per-cell shifts | same to 1e-5 | `fdtd_checks/freq_shift_validation.json` |
| Dataset baselines (converter, rotator, coding) | HV 0.1747 / 0.1490 / 0.0913 | 0.1753 / 0.1489 / 0.0917 (Monte Carlo HV) | `data/baseline_*.json` |
| 12 converters, 12 rotators in FDTD (h -> 0) | 12/12, 14.5-17.8 %; 12/12, 14.3-16.0 % | identical; spectra to 1e-7 | `fdtd_designs.json` here |
| Monte Carlo, 64 boards | 64/64, 64/64, 73 % | identical to 1e-6 | `search/mc_v2/mc.json` |
| Checkerboard boards (7 runs) | 8x8 tiles: 9.81-11.89 GHz | identical to 1e-7 | `search/coding_v2/board_*` |
| Mirror sign of r_yx, KK sign | as stated | confirmed (KK residual 0.08 vs 3.28 on our data) | |
| RDN stored inverted and mirrored | `flipud(1 - stored)` | confirmed by FDTD on normal rows | `experiments/dataset_checks/rdn_block_orientation.json` |

## Not reproduced or not correct

1. **"3,601 RDN rows carry spectra of other layouts" is wrong.** The rows (Data_081 r816 - Data_083 r903,
   Data_086 r612 - Data_088 r124; our mask `outputs_new_pipeline/notes/rdn_not_inverted_mask.npy`, 3,600 rows) are
   stored like the other classes: copper = 1, not inverted, not mirrored. Read that way, FDTD reproduces their stored
   spectra (|r_xx| MSE down to 6e-5, r_yx with the right sign; 8 rows, `rdn_block_orientation.json`). They are good data
   read with the wrong convention; the repo drops them.
2. **"16 RDN cells that rotate polarization" (paper v2, Sec. VIII) is 4.** Only 4 cells of all 111,000 have
   PCR > 0.9 over 9.8-10.6 GHz; the repo's own `fdtd_checks/rotator_cells.json` has `n = 4`. FDTD - CST = -0.017 there
   (paper: -0.010).
3. **Etching worst cases (`mc_tolerance.py etch`) fail the repo's own check.** The nominal 0.0625 mm run gives
   B = 19.08 / 16.65 / 20.5 % instead of the converged 17.84 / 16.02 / 20.07 % (handover: within 0.3 %). Over-etching
   by 62.5 um removes the converter's band. Not used in our paper.
4. **Stale committed results.** `fdtd_checks/validation.json` (r_yx complex MSE of RDN, before the mirror sign was
   applied) and `fdtd_checks/copper_loss.json` (before the ground loss) do not match the current code; the paper's
   numbers match the current code.
5. **`test_ground_loss.py`** needs an argument (`slab` or `cells`); without it it fails.
6. **Trained networks are not in the repo** (`runs/all2`, `runs/cp2` empty), so the network-dependent results
   (Table III two-output rows, searches) cannot be re-run from the repo alone.

## Our version (in this repository)

Per-class four-output networks (no all-class network), our data handling (block rows kept, read as stored), our
MOIPSO; designs checked in FDTD and CST. Results in `experiments/devices/` and the paper (`paper/main.tex`).
