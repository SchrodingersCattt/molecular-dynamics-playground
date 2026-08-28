# Figure Contract — Multi-step AIMD RHF–SCF

## Output

- Canonical stem: `03_aimd_scf`
- Formats: PNG, SVG, MP4
- Final files: `figures/03_aimd_scf.{png,svg}`, `videos/03_aimd_scf.mp4`
- Intermediate files: MatterVis frames, density-plane composites and checks under `_qa/03_aimd_scf/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps, minimum 18 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | outer MD loop | same empty `r → a → v` loop | fixed across the series |
| b | electronic solve and nuclear update | large MatterVis dimer, one evolving `ρᵏ(r)`, native vectors and a separate small SCF loop | fixed camera, viewport, grid and scale across every frame |

## Required Elements

- Seven real RHF/STO-3G water-dimer geometries linked by six 0.5 fs Velocity Verlet updates.
- The first ten seconds contain two complete SCF cycles; the final five seconds contain four faster ionic updates.
- Every SCF frame uses its real density on one shared molecular-plane grid. Residual-driven contour count, tone and blur encode coarse-to-converged state without replacing the density.
- After each detailed SCF convergence: pause, show the nuclear gradient/force, update the half-step velocity, then update position.
- MatterVis owns every atom, two-colour bond and atom-centred world-space vector. Density contours remain outside the SCF loop.

## Forbidden Changes

- No fabricated density, molecule inside either loop or per-frame camera fitting.
- Do not present display sharpness as an adaptive electronic-structure basis or grid.
- Do not interpolate unrelated ionic geometries into a fake continuous electronic trajectory.

## QA Plan

- Verify all density arrays, the shared grid, residual-to-contour mapping, Velocity Verlet identities and force provenance.
- MatterVis inspect/render every source; strict source/composite and every-frame checks.
- Inspect representative frames at every phase boundary and the final-size animation for visible coarse-to-fine evolution, vector anchoring and nuclear motion.

## Delivery Gate

- [ ] Seven ionic geometries and every real SCF density frame have provenance.
- [ ] Two full electronic cycles and four rapid ionic updates are readable in motion.
- [ ] Strict static and every-frame QA pass.
- [ ] Five-problem review matches the latest render hash.
