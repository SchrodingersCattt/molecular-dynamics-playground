# Figure Contract — Classical Lennard–Jones MD

## Output

- Canonical stem: `02_classical_lj`
- Formats: PNG, SVG, MP4
- Final files: `figures/02_classical_lj.{png,svg}`, `videos/02_classical_lj.mp4`
- Intermediate files: `_qa/02_classical_lj/source/` and `_qa/02_classical_lj/_qa/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps, minimum 18 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | abstract integrator | same empty `r → a → v` loop | identical to panel a in the other stories |
| b | concrete empirical potential | MatterVis H2O dimer, O···O distance, LJ forces and one rigid-body VV step | same camera and viewport throughout |

## Required Elements

- TIP3P O–O LJ term applied to a real water-dimer geometry.
- Direct structural mapping `rOO → U_LJ → F → a`; no standalone curve.
- MatterVis standard atom colours and two-colour bonds.

## Forbidden Changes

- No Ar pair, independent potential graph, hand-drawn structure or molecule inside the loop.
- Do not imply that the highlighted LJ term is the complete water force field.

## QA Plan

- Analytic LJ force must agree with central finite difference.
- O···O line and force origins use the same projected world coordinates as MatterVis.
- Run strict source/composite and every-frame checks, then final-size review.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Scientific derivative and strict visual QA pass.
- [ ] Five-problem review matches the latest render hash.

