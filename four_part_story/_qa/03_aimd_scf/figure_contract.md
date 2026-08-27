# Figure Contract — AIMD RHF–SCF

## Output

- Canonical stem: `03_aimd_scf`
- Formats: PNG, SVG, MP4
- Final files: `figures/03_aimd_scf.{png,svg}`, `videos/03_aimd_scf.mp4`
- Intermediate files: Cube files, MatterVis frames and checks under `_qa/03_aimd_scf/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps, minimum 18 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | outer MD loop | same empty `r → a → v` loop | fixed across the series |
| b | electronic solve | large MatterVis dimer and one evolving `ρᵏ(r)` plus separate small SCF loop | fixed camera, viewport, grid and nuclei for every iteration |

## Required Elements

- All 19 real RHF/STO-3G density iterates on one Cartesian grid.
- One density object evolving from broad/blurred to crisp; no convergence curve or competing Δρ object.
- Blur is a disclosed residual-driven visual encoding applied to each real `ρᵏ`, not a change in basis or SCF grid.
- Converged forces appear before nuclear displacement, never simultaneously.

## Forbidden Changes

- No fabricated density, molecule inside either loop or per-frame camera fitting.
- Do not present render sharpness as adaptive electronic-structure resolution.

## QA Plan

- Verify density matrices, grids, Cube hashes, residual-to-blur mapping and force provenance.
- MatterVis inspect/render every source; strict source/composite and every-frame checks.
- Inspect animation for genuinely visible coarse-to-fine evolution.

## Delivery Gate

- [ ] Nineteen iteration densities and provenance exist.
- [ ] Strict static and every-frame QA pass.
- [ ] Five-problem review matches the latest render hash.

