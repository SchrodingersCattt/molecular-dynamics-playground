# Figure Contract — 03 Deep Potential MD

## Output

- Canonical stem: `03_deep_potential_md`
- Formats: PNG, SVG, MP4
- Final files: `figures/03_deep_potential_md.png`, `figures/03_deep_potential_md.svg`, `videos/03_deep_potential_md.mp4`
- Intermediate files: MatterVis render/provenance, job diagnostics, strict QA reports, and extracted frames remain under `_qa/03_dpmd/`.

## Target Size

- Target context: 16:9 PowerPoint slide and independent video.
- Physical size: 338.7 mm wide.
- Raster rule: 1920 × 1080 PNG; SVG vector; 1920 × 1080 H.264 at 24 fps.
- Minimum text size at final size: 17 pt; title 39 pt.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | DPMD force-query expansion | Mini Verlet loop, real periodic 64-water box, chemical-bond view, 6 Å local sphere, in/out rendering, descriptor, shared atomic network, energy sum and force derivative | Reuse slides 01–02 typography, outer-loop grammar, white canvas, and semantic palette |

## Required Elements

- Data layers: 64 H2O molecules, 192 stable atom IDs, 12.4296 Å cubic PBC box at 0.997 g cm^-3.
- Local selection: source atom 126 at the box centre; 83 other atoms selected by exact minimum-image distance `r <= 6.0 Å`.
- 3D views: fixed asymmetric orthographic camera `[1.55, -1.0, 0.62]`; chemical bonds and neighbor-centric views use the same physical scale.
- Pipeline: structure → neighbors → descriptor `D_i` → shared atomic network `epsilon_i` → `E = sum_i epsilon_i`, `F = -grad_R E`.
- Animation: current pipeline step alone changes from light grey to black; the outer force-query node remains dark.

## Forbidden Changes

- Do not use a symmetric 111 camera or independently auto-fit stage views.
- Do not infer neighbors from screen distance; use stable IDs and minimum-image 3D distances.
- Do not omit atoms inside `r_cut` or include atoms outside it to improve appearance.
- Do not draw quantitative DPMD force arrows without successful model inference.
- Do not claim the prepared periodic box is an equilibrated MD frame.
- Do not stretch the cell or molecular geometry non-proportionally.

## Style

- Font family: DejaVu Sans.
- Font sizes: 17–39 pt.
- Line widths: 1.2–3.0 pt; box/bond/neighbor lines preserve hierarchy.
- Palette: white final canvas; grey context and logic; navy H/cutoff; crimson O; green learned atomic energy.
- Shared viewport: fixed 1920 × 1080 and fixed asymmetric orthographic camera.

## QA Plan

- MatterVis: inspect public extxyz input, preflight CPU render, render with explicit unit-cell selection and camera, verify output/hash/warnings.
- Figure QA: contract, source clipping, strict QA, review hash check.
- Geometry validation: exact bond endpoints, cell corners, central source ID, minimum-image distances, and isotropic camera transform.
- Video inspection: structure, cutoff, neighbor, network, and energy/force stages at original resolution.
- Human self-review: `self_review.md`.
- Outstanding deviations: no quantitative DPMD result is asserted because the Windows bohr uploader failed before job creation.

## Delivery Gate

- [x] Canonical outputs exist with requested names.
- [x] No unwanted duplicate variants remain.
- [x] Programmatic QA passes.
- [x] Final image was inspected at intended slide size.
- [x] Five-problem self-review matches the latest render.
