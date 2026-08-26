# Figure Contract — 02 Classical Lennard-Jones Potential

## Output

- Canonical stem: `02_classical_lj`.
- Formats: A4-width PNG/SVG and independent 16:9 MP4.
- Final files: `figures/02_classical_lj.png`, `.svg`, `videos/02_classical_lj.mp4`.

## Target Size

- Static: 297 × 210 mm, 3508 × 2480 px at 300 dpi; minimum text 10 pt.
- Video: 1920 × 1080, 24 fps, minimum text 18 pt.
- Static and video use separate compositions.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | potential definition | Large Ar 12-6 LJ curve with repulsive, equilibrium, attractive regions and moving exact `(r,U)` point | axes and units fixed across animation |
| b | concrete interaction | Large 3D Ar pair, separation bracket and large force arrows | fixed asymmetric orthographic camera and physical sphere scale |

## Required Elements

- `U(r)=4 epsilon[(sigma/r)^12-(sigma/r)^6]` and `F=-dU/dr`.
- Argon parameters `sigma=3.405 Å`, `epsilon=0.0103 eV`, equilibrium `r_m=2^(1/6)sigma`.
- Force arrows reverse direction on opposite sides of equilibrium and vanish at equilibrium.
- Exact curve point, separation, force sign, and displayed atom positions share one `r` value per frame.

## Forbidden Changes

- No softened or mock LJ values and no force direction chosen from screen position.
- No hidden axis truncation; the plotted domain and clipped repulsive branch are explicitly stated.
- No non-proportional scaling or independent camera fitting.

## QA Plan

- MatterVis inspect/preflight/source render for the Ar pair.
- Static strict QA at 297 mm and five-problem review.
- Every video frame: visualize-data whitespace, clipping, boundaries, colors, registered text/arrow bounds, and exact `r-U-F` consistency.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Static strict QA and review hash pass.
- [ ] Every video frame passes.
- [ ] LJ signs and equilibrium are numerically verified.
- [ ] Final static and motion are visually accepted.
