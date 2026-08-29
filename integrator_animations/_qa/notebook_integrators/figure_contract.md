# Figure Contract

## Output

- Canonical stem: integrator_comparison
- Formats: PNG and PDF
- Final files: integrator_animations/figures/integrator_comparison.png; integrator_animations/figures/integrator_comparison.pdf
- Intermediate files with leading underscore: QA raster renders and reports only
- Files that must be removed before delivery: exploratory variants and temporary page renders

## Target Size

- Target context: A4 landscape / double-column reusable figure
- Physical size: 297 x 210 mm
- DPI or vector export rule: vector PDF plus 300 dpi PNG
- Minimum text size at final size: 10 pt; maximum 16 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| rows 1-3 | dt = 0.05, 0.10, 0.50 | identical methods and diagnostics | shared physical and semantic scales |
| column 1 | short-time position | analytical reference and four representative methods | common time and position limits |
| column 2 | phase-space geometry | equal aspect; explicit divergence marker | common q/p limits |
| column 3 | long-time energy error | absolute relative error on log scale | common time and energy limits |

## Required Elements

- Data layers: analytical, Explicit Euler, Symplectic Euler, Velocity Verlet, classical RK4
- Highlights: boundary marker where a trajectory leaves the shared phase-space window
- Labels: dt row labels, three column titles, physical axis labels
- Leaders/arrows: none
- Legends: one figure-level legend in reserved top whitespace
- Panel letters: none; rows and columns are self-identifying
- Animation labels or frame timing: not applicable

## Forbidden Changes

- Do not drop the diverging Explicit Euler trajectory.
- Do not rename fixed-step RK4 as RK45.
- Do not crop away divergence markers, axes, legends, or titles.
- Do not smooth, normalize, or independently rescale comparable rows.
- Do not use non-proportional stretching.
- Do not replace computed trajectories with illustrative curves.

## Style

- Font family: DejaVu Sans
- Font sizes: 10-14 pt
- Line widths: 0.8-1.8 pt
- Marker sizes: 26 pt^2 for divergence marker
- Palette: analytical #8E8E8E; Euler #A99C50; symplectic Euler #2F8562; Velocity Verlet #4E9BB5; RK4 #183153
- Background: #FFFFFF
- Shared rules: fixed axes within each diagnostic; equal aspect for every phase panel

## QA Plan

- Programmatic helpers: strict visualize-data QA, clipping, print, colors, consistency and quantitative checks
- Safety bands: top legend band and all row/column gaps
- Label/leader validation: no leaders; verify legend and labels occupy whitespace
- Raster/vector inspection: render PDF at 300 dpi and compare with canonical PNG
- Human self-review: self_review.md
- Outstanding deviations: none unless recorded in the manifest

## Delivery Gate

- [ ] Canonical outputs exist with requested names.
- [ ] No unwanted duplicate variants remain.
- [ ] Programmatic QA passes or deviations are logged and approved.
- [ ] Final image was inspected at intended physical size.
- [ ] Five-problem self-review was completed after the latest render.
