# Figure Contract — 03 AIMD and the SCF Loop

## Output

- Canonical stem: `03_aimd_scf`.
- Formats: A4-width PNG/SVG and independent 16:9 MP4.
- Final files: `figures/03_aimd_scf.png`, `.svg`, `videos/03_aimd_scf.mp4`.

## Target Size

- Static: 297 × 210 mm, 3508 × 2480 px at 300 dpi; minimum text 10 pt.
- Video: 1920 × 1080, 24 fps; minimum text 18 pt.
- Static is a separate explanatory composition, not a video screenshot.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | outer MD loop | Empty position → acceleration/force → velocity loop | no molecule or density inside |
| b | concrete quantum calculation | Large 3D H2O dimer plus large electron-density isosurface and large force/motion arrows | fixed camera and isotropic scale |
| c | inner SCF loop | Small independent density → Fock → solve → new density loop with convergence readout | no structure inside the loop |

## Required Elements

- Real 19-iteration generalized RHF/STO-3G SCF history and finite-difference forces.
- Density isosurface derived from the stored converged density matrix/basis, not an illustrative blob.
- The SCF loop is visibly nested inside one outer force evaluation without geometric overlap.
- Atom motion after the force evaluation is visible and arrows have large heads.

## Forbidden Changes

- No structure inside either loop and no 2D density contour substituted for the requested 3D view.
- No claim that RHF/STO-3G is production DFT; label the pedagogical electronic-structure level.
- No independent camera/scale changes between SCF stages and no invented forces.

## QA Plan

- MatterVis inspect/preflight/source render for the dimer; verify density grid and isosurface bounds separately.
- Static strict QA, geometry endpoints, source/composite clipping, five-problem review.
- Every video frame: visualize-data whitespace, clipping, boundaries, colors, text/arrow bounds, SCF-iteration/state consistency, and motion extrema.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Static strict QA and review hash pass.
- [ ] Every video frame passes.
- [ ] SCF and force provenance is verified.
- [ ] Final static and motion are visually accepted.
