"""
cell_common.py — clean build/simulate module for a unit cell with a
PROPERLY CONSTRUCTED ground plane (an actual brick, built as part of the
cell -- not a boundary-condition substitute).

Construction (bottom to top):
    z < 0                : open half-space below the ground plane
    z in [-GROUND_T, 0]  : ground plane brick (conductor)
    z in [0, t_sub]      : substrate brick (dielectric)
    z in [t_sub, t_sub+PATCH_T] : patch bricks, one per '1' pixel (conductor)
    z > t_sub+PATCH_T    : open half-space above the structure

Boundaries: unit cell in x/y, "expanded open" in z on BOTH faces (standard
CST setup for a periodic array with a floating conductor backing -- the
ground plane brick is what physically blocks transmission, the boundary
itself stays open on both sides). Floquet port only meaningfully returns
energy from the top (zMax) side since nothing gets through the ground
plane; s11_co/s11_cross are read from that port.

Root-cause note (why earlier versions of this failed intermittently):
the actual bug was NOT the boundary/brick combination -- it was that the
same CST_Workspace directory gets reused across many sequential designs,
and copying the template into "worker.cst" left the OLD worker/ subfolder
(mesh + result cache from the PREVIOUS design) in place, merging fresh
data on top of stale data via copytree(..., dirs_exist_ok=True). That
caused intermittent, design-dependent result-tree corruption. Fixed here
by deleting worker.cst + its subfolder completely before every design.

STALE-TEMPLATE GUARD (the padding bug): build_base_template() skips work if
base_template.cst already exists. If the geometry changed since that
template was built -- e.g. padding went 1.0 -> 0.0, shrinking the cell from
10mm to 8mm -- the OLD 10mm substrate/ground bricks and the OLD 10mm unit
cell boundary get silently reused, while the patches are laid out over the
new 8mm span. Result: a leftover strip on the +x and +y sides (the patch
array is anchored at the origin, not centred) and S-parameters computed at
the wrong period. Every build now writes a geometry.json next to the
template and refuses to reuse a template whose spec does not match.

Geometry/material are parameters, not hardcoded, so this one module covers
whatever configuration you need without maintaining N near-duplicate files:

    grid_size           : e.g. 16
    patch_size           : mm, per-pixel size
    padding               : mm, PER SIDE (0 = patches fill the cell edge-to-edge)
    unit_cell_size(...)   : derived = grid_size*patch_size + 2*padding
    substrate_eps, substrate_tand : dielectric properties
    substrate_thickness  : mm
    ground_thickness      : mm, ground plane brick thickness
    patch_thickness       : mm, must be > 0 (zero-height bricks are degenerate)
    conductor             : "PEC" or "copper" (sigma=5.8e7 S/m), applies to
                             BOTH the ground plane and the patches
"""

import os
import json
import shutil
import time
import numpy as np
import context               # CST environment shim from your project
import cst_python_api as cpa # CST Python API

FREQ_MIN = 2.0
FREQ_MAX = 12.0
FREQ_SAMPLES = 1001

COPPER_SIGMA = 5.8e7   # S/m, annealed copper

GEOMETRY_SPEC_NAME = "geometry.json"


def unit_cell_size(grid_size, patch_size, padding):
    return grid_size * patch_size + 2 * padding


def _geometry_spec(grid_size, patch_size, padding, substrate_eps,
                    substrate_tand, substrate_thickness, ground_thickness,
                    conductor):
    """Everything baked into base_template.cst. If any of this changes, the
    cached template is invalid and must not be reused."""
    return {
        "grid_size": int(grid_size),
        "patch_size": float(patch_size),
        "padding": float(padding),
        "unit_cell_size": float(unit_cell_size(grid_size, patch_size, padding)),
        "substrate_eps": float(substrate_eps),
        "substrate_tand": float(substrate_tand),
        "substrate_thickness": float(substrate_thickness),
        "ground_thickness": float(ground_thickness),
        "conductor": str(conductor),
        "freq_min": FREQ_MIN,
        "freq_max": FREQ_MAX,
        "freq_samples": int(FREQ_SAMPLES),
    }


def build_base_template(workspace_dir, grid_size, patch_size, padding,
                         substrate_eps, substrate_tand, substrate_thickness,
                         ground_thickness=0.18, conductor="PEC",
                         template_name="base_template.cst"):
    """Builds (or skips, if already present AND the geometry matches) the CST
    template: ground plane brick + substrate brick. Patches are added
    per-design in simulate_design()."""
    template_path = os.path.join(workspace_dir, template_name)
    spec_path = os.path.join(workspace_dir, GEOMETRY_SPEC_NAME)
    spec = _geometry_spec(grid_size, patch_size, padding, substrate_eps,
                           substrate_tand, substrate_thickness,
                           ground_thickness, conductor)

    if os.path.exists(template_path):
        cached = None
        if os.path.exists(spec_path):
            with open(spec_path) as fh:
                cached = json.load(fh)
        if cached == spec:
            return
        differences = "unknown (no geometry.json -- template predates this check)"
        if cached is not None:
            differences = ", ".join(
                f"{k}: cached={cached.get(k)!r} wanted={v!r}"
                for k, v in spec.items() if cached.get(k) != v
            )
        raise RuntimeError(
            f"{template_path} was built for a DIFFERENT geometry and cannot be "
            f"reused.\n  Mismatch: {differences}\n"
            f"Delete {workspace_dir} (or run with a workspace name that encodes "
            f"the full geometry) and re-run."
        )

    cell = spec["unit_cell_size"]
    print(f"--- Building template: cell={cell:g}mm "
          f"({grid_size}x{patch_size:g}mm + 2x{padding:g}mm padding), "
          f"t_sub={substrate_thickness:g}mm, conductor={conductor} ---")

    myCST = cpa.CST_MicrowaveStudio(workspace_dir, template_name)

    myCST.Solver.setBoundaryCondition(
        xMin="unit cell", xMax="unit cell",
        yMin="unit cell", yMax="unit cell",
        zMin="expanded open", zMax="expanded open",
    )
    myCST.Solver.defineFloquetModes(nModes=2, theta=0.0, phi=0.0, forcePolar=False, polarAngle=0.0)

    myCST.Build.Material.addNormalMaterial(
        "Substrate Dielectric", substrate_eps, 1.0,
        colour=[0.2, 0.5, 0.2], tanD=substrate_tand,
    )

    if conductor == "copper":
        myCST.Build.Material.addNormalMaterial(
            "Copper (annealed)", 1.0, 1.0,
            colour=[0.8, 0.5, 0.2], sigma=COPPER_SIGMA,
        )
        conductor_name = "Copper (annealed)"
    else:
        conductor_name = "PEC"

    # Ground plane -- an actual constructed brick, part of the cell.
    myCST.Build.Shape.addBrick(
        xMin=0.0, xMax=cell, yMin=0.0, yMax=cell,
        zMin=-ground_thickness, zMax=0.0,
        name="GroundPlane", component="component1",
        material=conductor_name,
    )

    # Substrate, directly on top of the ground plane.
    myCST.Build.Shape.addBrick(
        xMin=0.0, xMax=cell, yMin=0.0, yMax=cell,
        zMin=0.0, zMax=substrate_thickness,
        name="Substrate", component="component1",
        material="Substrate Dielectric",
    )

    myCST.Solver.setFrequencyRange(FREQ_MIN, FREQ_MAX)
    myCST.Solver.changeSolverType("HF Frequency Domain")
    myCST._CST_MicrowaveStudio__MWS.FDSolver.setNumberOfResultDataSamples(int(FREQ_SAMPLES))
    myCST.saveFile(workspace_dir, template_name)
    myCST.closeFile()

    with open(spec_path, "w") as fh:
        json.dump(spec, fh, indent=2)

    time.sleep(2)


def simulate_design(workspace_dir, matrix, grid_size, patch_size, padding,
                     substrate_thickness, patch_thickness=0.018, conductor="PEC",
                     template_name="base_template.cst", worker_name="worker.cst",
                     return_all_modes=False, grow=0.0):
    """Copies the template into a CLEAN worker project (no leftover state
    from a previous design -- see module docstring), adds this design's
    patches, runs the sim, returns (freq, s11_co, s11_cross)."""
    if matrix.shape != (grid_size, grid_size):
        raise ValueError(f"Expected a {grid_size}x{grid_size} design, got {matrix.shape}")

    if patch_thickness <= 0:
        raise ValueError(
            f"patch_thickness must be > 0 (got {patch_thickness}); a zero-height "
            f"brick is degenerate and CST will drop it or fail meshing."
        )

    cell = unit_cell_size(grid_size, patch_size, padding)
    conductor_name = "Copper (annealed)" if conductor == "copper" else "PEC"

    # Wipe any leftover worker.cst + subfolder from a PREVIOUS design before
    # starting this one -- this is the actual fix for the intermittent
    # "specified result item does not exist" failures.
    worker_path = os.path.join(workspace_dir, worker_name)
    worker_subdir = os.path.join(workspace_dir, os.path.splitext(worker_name)[0])
    if os.path.exists(worker_path):
        os.remove(worker_path)
    if os.path.exists(worker_subdir):
        shutil.rmtree(worker_subdir)

    shutil.copy2(os.path.join(workspace_dir, template_name), worker_path)
    base_subdir = os.path.join(workspace_dir, os.path.splitext(template_name)[0])
    if os.path.exists(base_subdir):
        shutil.copytree(base_subdir, worker_subdir)

    # 17 edges for 16 cells, spanning padding -> cell-padding on both axes.
    # With padding=0 this is 0 -> cell, i.e. patches fill the cell edge to
    # edge and the outermost column/row becomes electrically continuous with
    # the neighbouring cell under the periodic boundary. That is a valid but
    # physically DIFFERENT structure from the 1mm-padded SUTD one.
    x, y = np.meshgrid(
        np.linspace(padding, cell - padding, grid_size + 1),
        np.linspace(padding, cell - padding, grid_size + 1),
    )
    y = np.flipud(y)

    myCST = None
    try:
        myCST = cpa.CST_MicrowaveStudio(workspace_dir, worker_name)

        z0 = substrate_thickness
        z1 = substrate_thickness + patch_thickness
        brick_count = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if matrix[r, c] == 1:
                    brick_count += 1
                    # grow > 0 enlarges every patch so that diagonally touching
                    # pixels overlap and conduct (corner-contact convention)
                    myCST.Build.Shape.addBrick(
                        xMin=float(x[r, c]) - grow,     xMax=float(x[r, c + 1]) + grow,
                        yMin=float(y[r + 1, c]) - grow, yMax=float(y[r, c]) + grow,
                        zMin=z0, zMax=z1,
                        name=f"Patch_{brick_count}", component="component1",
                        material=conductor_name,
                    )

        myCST.saveFile(workspace_dir, worker_name)
        myCST.Solver.runSimulation()

        # Reads from the Zmax port only -- the TOP of the stack, above the
        # patches, where the boundary is open and the incident plane wave
        # comes from (matches the paper: "x-polarized plane wave incident
        # normally from the top"). Zmin (below the ground plane) never
        # contributes a meaningful result since the ground brick blocks it.
        # So s11_co/s11_cross are reflection measured from the patches
        # side, not the ground-plane side.
        freq, s11_co    = myCST.Results.getSParameters(0, 0, 1, 1)   # Zmax, mode1->mode1 (co-pol)
        _,    s11_cross = myCST.Results.getSParameters(0, 0, 2, 1)   # Zmax, mode2->mode1 (cross-pol)
        if return_all_modes:
            _, s12 = myCST.Results.getSParameters(0, 0, 1, 2)        # mode1 <- mode2
            _, s22 = myCST.Results.getSParameters(0, 0, 2, 2)        # mode2 <- mode2
            return freq, s11_co, s11_cross, s12, s22

        return freq, s11_co, s11_cross

    finally:
        if myCST:
            try:
                myCST.closeFile()
            except Exception:
                pass
        time.sleep(1.0)