# Figure Contract — Classical Lennard–Jones MD

## Output

- Canonical stem: `02_classical_lj`
- Formats: PNG, SVG, MP4
- Final files: `figures/02_classical_lj.{png,svg}`, `videos/02_classical_lj.mp4`
- Intermediate files: `_qa/02_classical_lj/source/` and `_qa/02_classical_lj/_qa/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps; all text 10–16 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| A | abstract integrator | same `r → a → v` loop with the active equation inside | identical to panel A in the other stories |
| B | concrete empirical potential | MatterVis H2O dimer, top physical-action label and lower-left `Simulation step` | same camera and viewport throughout |
| C | LJ relation | compact `rOO → U_LJ → F → a` map and scope note | fixed upper-right slot |
| D | LJ loop | compact `rOO → U_LJ → F` evaluation loop | fixed lower-right slot |

## Required Elements

- TIP3P O–O LJ term applied to a real water-dimer geometry.
- Direct structural mapping `rOO → U_LJ → F → a`; no standalone curve.
- MatterVis standard atom colours and two-colour bonds.

## Forbidden Changes

- No Ar pair, independent potential graph, hand-drawn structure or molecule inside the loop.
- Do not imply that the highlighted LJ term is the complete water force field.
- No page header; one type family throughout; every label is between 10 and 16 pt.

## QA Plan

- Analytic LJ force must agree with central finite difference.
- O···O line and force origins use the same projected world coordinates as MatterVis.
- Run strict source/composite and every-frame checks, then final-size review.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Scientific derivative and strict visual QA pass.
- [ ] Five-problem review matches the latest render hash.
