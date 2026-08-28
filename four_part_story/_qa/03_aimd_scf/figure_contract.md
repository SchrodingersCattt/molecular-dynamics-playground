# Figure Contract — Multi-step AIMD RHF–SCF

## Output

- Canonical stem: `03_aimd_scf`
- Formats: PNG, SVG, MP4
- Final files: `figures/03_aimd_scf.{png,svg}`, `videos/03_aimd_scf.mp4`
- Intermediate files: MatterVis frames, density-plane composites and checks under `_qa/03_aimd_scf/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps, every font 10–16 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| A | Velocity Verlet loop | large `r → a → v` circle with the active equation inside | fixed size across the series |
| B | molecular case | large MatterVis dimer, one evolving `ρᵏ(r)`, native vectors, one physical-action line and fixed lower-left `Simulation step NN` | fixed camera, viewport, grid and scale across every frame; no SCF iteration text |
| C | electronic convergence | real `|Eᵏ-E*|`, `Iteration NN / total` and the active electronic action | fixed logarithmic energy scale; only panel with electronic-step text |
| D | SCF loop | separate `F → C → ρ → ?` electronic loop without a duplicate iteration counter | never contains the molecule or density |

## Required Elements

- Seven real RHF/STO-3G water-dimer geometries linked by six 0.5 fs Velocity Verlet updates.
- The first ten seconds contain two complete SCF cycles; the final five seconds contain four faster simulation steps.
- Every SCF frame uses its real density on one shared molecular-plane grid. One fixed eight-level contour set, the true density change and residual-driven blur encode coarse-to-converged state without replacing the density or popping contour bands.
- Every simulation step uses its real RHF energy sequence in panel C; a navy trace and marker reveal only progress through the current SCF solve.
- After each detailed SCF convergence: pause, show the nuclear gradient/force, update the half-step velocity, then update position.
- The convergence pause is doubled relative to the previous cut while the total animation remains 15 seconds.
- During position drift, the previous and updated MatterVis structures coexist briefly as a controlled ghosted transition.
- MatterVis owns every atom, two-colour bond and atom-centred world-space vector. Density contours remain outside the SCF loop.

## Forbidden Changes

- No fabricated density, molecule inside either loop or per-frame camera fitting.
- Do not present display sharpness as an adaptive electronic-structure basis or grid.
- Do not interpolate unrelated simulation-step geometries into a fake continuous electronic trajectory.

## QA Plan

- Verify all density arrays, the shared grid, residual-to-contour mapping, Velocity Verlet identities and force provenance.
- MatterVis inspect/render every source; strict source/composite and every-frame checks.
- Inspect representative frames at every phase boundary and the final-size animation for visible coarse-to-fine evolution, vector anchoring and nuclear motion.

## Delivery Gate

- [ ] Seven simulation-step geometries and every real SCF density frame have provenance.
- [ ] Two full electronic cycles and four rapid simulation updates are readable in motion.
- [ ] Strict static and every-frame QA pass.
- [ ] Five-problem review matches the latest render hash.
