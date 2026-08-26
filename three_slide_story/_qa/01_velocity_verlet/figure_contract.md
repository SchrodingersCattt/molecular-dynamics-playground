# Figure Contract — 01 Velocity Verlet

## Output

- Canonical stem: `01_velocity_verlet`
- Formats: PNG, SVG, MP4
- Final files: `figures/01_velocity_verlet.png`, `figures/01_velocity_verlet.svg`, `videos/01_velocity_verlet.mp4`
- Intermediate files: QA reports and extracted inspection frames stay under `_qa/`.
- Files that must be removed before delivery: contact sheets and ad-hoc frame grabs outside `_qa/`.

## Target Size

- Target context: 16:9 PowerPoint slide and full-screen video.
- Physical size: 338.7 mm wide (13.333 in widescreen slide).
- Raster rule: 1920 × 1080 PNG; SVG remains vector; video is 1920 × 1080 at 24 fps.
- Minimum text size at final size: 18 pt; title 39 pt.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | Single-slide Verlet loop | Five ordered stages, H2O state, position/velocity/force semantics | Same ring, typography, node sizes, and palette as slides 02–03 |

## Required Elements

- Data layers: one real H2O trajectory step from `md_workflows/classical_data.npz`.
- Highlights: one active node at a time in the MP4; static PNG leaves all nodes neutral.
- Labels: state, half kick, drift, force query, half kick; equations for all update stages.
- Leaders/arrows: clockwise ring arrows; geometry-anchored velocity and force arrows.
- Legend: position = navy, velocity = green, force = crimson.
- Animation timing: five equal stages across 12.0 s at 24 fps.

## Forbidden Changes

- Do not drop or reorder any of the five Velocity Verlet stages.
- Do not crop the molecular geometry, node circles, equations, or title.
- Do not imply that display-normalized arrows encode absolute vector magnitude.
- Do not stretch the structure non-proportionally.
- Do not replace the real trajectory step with a decorative motion path.

## Style

- Font family: DejaVu Sans.
- Font sizes: 18–39 pt.
- Line widths: 1.5–3.0 pt; bonds 4.0 pt.
- Marker sizes: stage nodes 28 pt diameter-equivalent; atom radii follow element class.
- Palette: white background; grey loop; black active stage; navy position/H; green velocity; crimson force/O.
- Shared viewport: 1920 × 1080; single circular loop centered on the slide.

## QA Plan

- Programmatic helpers: contract, source clipping, strict QA, review check.
- Safety bands: full-image four-edge clearance; no internal panel bands because this is one panel.
- Label/leader validation: all vectors originate at projected atom coordinates; no free callout leaders.
- Raster/vector inspection: PNG at original 1920 × 1080 plus extracted MP4 stage frames.
- Human self-review: `self_review.md`.
- Outstanding deviations: none.

## Delivery Gate

- [x] Canonical outputs exist with requested names.
- [x] No unwanted duplicate variants remain.
- [x] Programmatic QA passes.
- [x] Final image was inspected at intended slide size.
- [x] Five-problem self-review matches the latest render.
