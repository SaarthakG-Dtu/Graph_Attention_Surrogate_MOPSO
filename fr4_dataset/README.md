# FR-4 16×16 RDN dataset (SUTD-PRCM cell, FR-4 substrate)

Random 16×16 pixel layouts simulated in CST exactly like SUTD-PRCM, except that the
substrate is lossy FR-4. Meant for fine-tuning the surrogate on a new substrate.

| Setting | Value |
|---|---|
| Layouts | RDN class: every pixel copper with probability 0.5; 50,000 distinct designs (seed 2026), no all-empty/all-copper; **1 = copper** |
| Cell | 16×16 pixels of 0.5 mm, 1 mm padding (10 mm cell), copper patches 0.018 mm, copper ground 0.18 mm |
| Substrate | FR-4 (lossy), εr = 4.3, tan δ = 0.025, t = 1.0 mm (`fr4_config.py`) |
| Sweep | 2–12 GHz, 1001 points, normal incidence, two Floquet modes, frequency-domain solver |
| Saved per design | layout, frequency, `s11_co` (r_xx), `s11_cross`, `s12`, `s22` (complex), substrate data, CST time |

The CST cell is the repository's `cell_common.py`, which reproduces the SUTD-PRCM spectra
(|r| MSE ≈ 1e-6 with the SUTD substrate); only the substrate parameters differ.

## Files

| File | Purpose |
|---|---|
| `fr4_config.py` | every setting |
| `generate_designs.py` | makes `designs_rdn16_fr4_50k.npy` (+ `.json` with statistics and checksum); refuses to overwrite |
| `run_dataset.py` | CST sweep; one `samples_fr4_t1mm/sample_XXXXXX.npz` per design |
| `pack_dataset.py` | progress report; `--pack` exports finished blocks of 1000 in SUTD format |

## Running

```bat
python run_dataset.py                          :: indices 0 .. 29,999 on this PC
python run_dataset.py --start 0 --end 15000    :: or split the range across PCs
python run_dataset.py --start 15000 --end 30000 --worker pc2
python pack_dataset.py                         :: progress and time left
python pack_dataset.py --pack                  :: SUTD-format export for training
```

Stop at any time (Ctrl+C, closing the window, a crash or a power cut) and run the same
command again; it continues where it stopped:

* every design is saved the moment it finishes, to a temporary name that is renamed only
  when the file is complete, so no half-written sample can exist;
* finished designs are skipped; `--verify` re-checks every saved file and re-simulates damaged ones;
* lock files in `samples_fr4_t1mm/_locks` stop two workers from simulating the same design;
  locks of a crashed worker are taken over after 2 h;
* failed designs are listed in `logs/failed.csv` and retried on later runs (up to 3 times);
* the sample folder is tied to the design file and the settings (`_dataset.json`), so
  samples of different substrates or design files can never be mixed.

CST time is 2.5–11 min per design on one PC (CPU, 8 threads): about 3,400 h for 30,000
designs on one worker, so split the range across several PCs.

## Mesh

CST's default adaptive mesh refinement is used, as for SUTD-PRCM. Zero-thickness PEC
patches, fixed meshes and a capped refinement were tested on designs 0-6 and rejected:
they either changed the spectra by about as much as the surrogate's test error or were
not faster.
