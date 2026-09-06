# Figure Contract — 02 AIMD SCF

## Output

- Canonical stem: `02_aimd_scf`
- Formats: PNG, SVG, MP4
- Final files: `figures/02_aimd_scf.png`, `figures/02_aimd_scf.svg`, `videos/02_aimd_scf.mp4`
- Intermediate files: source clipping, strict QA, review reports, and extracted frames under `_qa/02_aimd_scf/`.

## Target Size

- Target context: 16:9 PowerPoint slide and independent video.
- Physical size: 338.7 mm wide.
- Raster rule: 1920 × 1080 PNG; SVG vector; 1920 × 1080 H.264 at 24 fps.
- Minimum text size at final size: 18 pt; title 39 pt.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | AIMD force-query expansion | Faded outer Verlet loop, active force query, five-stage SCF loop, real H2O dimer density and forces | Reuse slide 01 outer-loop grammar; SCF loop owns the right-side focus |

## Required Elements

- Data layers: real RHF/STO-3G H2O dimer SCF energies, density matrices, electron-density grids, and finite-difference forces.
- Highlights: outer force-query node remains dark; the current SCF stage alone changes from light grey to black.
- Labels: density, build Fock matrix, solve orbitals, update density, convergence test.
- Leaders/arrows: clockwise arrows in both loops; geometry-anchored atomic force arrows only after convergence.
- Animation: sampled real iterations preserve iteration order and end at the actual converged result.

## Forbidden Changes

- Do not replace the real SCF history with a hand-drawn exponential curve.
- Do not omit the backward edge from a failed convergence test to the next density.
- Do not imply that the electron density or display-normalized force arrows are at an absolute common scale.
- Do not crop atoms, density contours, equations, or loop arrows.
- Do not non-proportionally stretch the dimer or density grid.

## Style

- Font family: DejaVu Sans.
- Font sizes: 18–39 pt.
- Line widths: 1.5–3.0 pt; bonds 3.0–4.0 pt.
- Palette: white background; grey inactive logic; black active logic; sparse navy/crimson atoms and force; green only for final convergence.
- Shared viewport: 1920 × 1080.

## QA Plan

- Programmatic helpers: contract, source clipping, strict QA, review hash check.
- Safety bands: four-edge clearance; a blank connector corridor separates the two loops except for the declared connecting arrow.
- Geometry validation: bonds and force origins derive from the same projected atom coordinates.
- Raster/vector inspection: original-resolution PNG and MP4 frames at early, middle, pre-converged, and converged states.
- Human self-review: `self_review.md`.
- Outstanding deviations: none.

## Delivery Gate

- [x] Canonical outputs exist with requested names.
- [x] No unwanted duplicate variants remain.
- [x] Programmatic QA passes.
- [x] Final image was inspected at intended slide size.
- [x] Five-problem self-review matches the latest render.
