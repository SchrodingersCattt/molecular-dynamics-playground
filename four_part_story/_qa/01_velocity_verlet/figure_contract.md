# Figure Contract — Velocity Verlet

## Output

- Canonical stem: `01_velocity_verlet`
- Formats: PNG, SVG, MP4
- Final files: `figures/01_velocity_verlet.{png,svg}`, `videos/01_velocity_verlet.mp4`
- Intermediate files: `_qa/01_velocity_verlet/source/` and `_qa/01_velocity_verlet/_qa/`
- Files that must not be delivered: failed frames and exploratory suffix variants

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps; all text 10–16 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| A | abstract integrator | exact `r → a → v` loop with the active equation inside the circle | fixed geometry across all four stories |
| B | concrete step | MatterVis H2O, top physical-action label and lower-left `Simulation step` | one camera, viewport, atom IDs and physical geometry |
| C | known state | compact `r_n`, `a_n`, `v_n` state card | fixed upper-right slot |
| D | updated state | compact `r_{n+1}`, `a_{n+1}`, `v_{n+1}` state card | fixed lower-right slot |

## Required Elements

- MatterVis red O, white H and two-colour O–H half-bonds.
- Navy displacement, crimson acceleration and green velocity arrows.
- Old pose ghost and new pose solid; arrows are visually amplified without numeric labels.

## Forbidden Changes

- No hand-drawn atoms or bonds; no molecule inside the loop.
- No incorrect `r → v → a` ordering and no non-proportional image stretching.
- Do not alter stored coordinates, velocities or accelerations for readability.

## Style

- White background; inactive loop light grey; active stage charcoal/black.
- No page header; one type family throughout; every label is between 10 and 16 pt.
- MatterVis orthographic camera direction `[1.55,-1.0,0.62]`, up `[0,0,1]`.
- Arrow shafts at least 8 px and heads at least 20 px in video.

## QA Plan

- MatterVis inspect/render JSON, source and composite clipping, strict manifest QA.
- Every encoded frame receives layout, whitespace, clipping, boundary and colour checks.
- Inspect full-frame contact sheets and the video as motion at final display size.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Strict static and every-frame video QA pass.
- [ ] Final-size five-problem review matches the latest render hash.
