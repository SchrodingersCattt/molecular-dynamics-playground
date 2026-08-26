# Figure Contract — 01 Velocity Verlet

## Output

- Canonical stem: `01_velocity_verlet`
- Formats: A4-width PNG/SVG plus an independent 16:9 MP4.
- Final files: `figures/01_velocity_verlet.png`, `.svg`, and `videos/01_velocity_verlet.mp4`.
- Intermediate files: source panels, MatterVis provenance, geometry metadata, decoded QA frames, and reports only under `_qa/01_velocity_verlet/`.

## Target Size

- Static: A4 landscape, 297 × 210 mm, 3508 × 2480 px at 300 dpi; minimum text 10 pt.
- Video: 1920 × 1080, H.264/yuv420p, 24 fps; minimum text 18 pt.
- Static and video have separate layouts; the static figure is not a captured video frame.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | abstract integrator | Empty three-node loop: position → acceleration → velocity; compact equations in reserved centre whitespace | no atoms or molecular geometry in the loop |
| b | concrete step | Large 3D H2O before/after state, displacement, acceleration, and velocity arrows | one fixed asymmetric orthographic camera and isotropic scale |

## Required Elements

- Exact three-step Velocity Verlet equations and one numerically verified step.
- Stable O/H atom IDs and stored O-H bonds.
- Position stage: old state as ghost, new state solid, large displacement arrows.
- Acceleration stage: `a = F/m` arrows anchored at the new positions.
- Velocity stage: old and new velocity arrows with a clear update.
- Active stage changes light grey to black; physical accents remain navy/crimson/green.

## Forbidden Changes

- No structure inside the abstract loop.
- No Euler update substituted for Velocity Verlet; no missing second acceleration.
- No non-proportional scaling, independent camera fitting, or screen-distance bonds.
- Arrow display scaling must be declared and constant within each vector type.

## QA Plan

- MatterVis inspect, CPU preflight, explicit-camera source render, output/hash verification.
- Static: source clipping, composite clipping, boundaries, colors, 297 mm print scale, strict QA, five-problem review.
- Video: decode every frame; run visualize-data whitespace, clipping, boundary, and color gates on every frame; validate registered text/arrow bounds and inspect motion extrema.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Static strict QA and review hash pass.
- [ ] Every video frame passes the declared gates.
- [ ] Minimum font sizes and arrowhead sizes pass metadata checks.
- [ ] Final static and full motion were inspected at delivery size.
