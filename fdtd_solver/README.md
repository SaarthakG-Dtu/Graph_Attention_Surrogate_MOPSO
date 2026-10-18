# FDTD solver for the SUTD-PRCM unit cell

Taken from the sutd-prcm-solverfree repository (A. Kumar), used here to check designs without a commercial
solver. Validated on this machine against the dataset's CST spectra (experiments/verify_solverfree/).

- `fdtd.py` — 3-D Yee FDTD of one periodic cell at normal incidence (UPML/CPML top, lossy copper: pixels as
  resistive sheets, ground as surface impedance, R_s = 0.026 ohm), plane-averaged specular reflection, and the
  periodic checkerboard supercell. `to_port` moves the reflection to the dataset's port plane (10.705 mm).
- `fdtd_mc.py` — the same cell with per-sample materials and thickness, for tolerance studies.
- `extrapolate.py` — two meshes (0.25 and 0.125 mm) to h -> 0 by a frequency-warp Richardson extrapolation.
- `convention.json` — layouts in our "true" convention (1 = copper, incident field along image rows) are
  passed to the solver transposed.

Stop all FDTD jobs gracefully by creating `fdtd_solver/STOP_FDTD`.
