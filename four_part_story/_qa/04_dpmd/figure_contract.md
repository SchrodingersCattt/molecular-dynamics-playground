# Figure Contract — 04 Deep Potential Molecular Dynamics

## Output

- Canonical stem: `04_deep_potential_md`.
- Formats: A4-width PNG/SVG and independent 16:9 MP4.
- Final files: `figures/04_deep_potential_md.png`, `.svg`, `videos/04_deep_potential_md.mp4`.

## Target Size

- Static: 297 × 210 mm, 3508 × 2480 px at 300 dpi; minimum text 10 pt.
- Video: 1920 × 1080, 24 fps; minimum text 18 pt.
- Static and video have independent layouts.

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | MD loop | Empty outer position → force/acceleration → velocity loop | no structure inside |
| b | concrete local environment | Large 3D 64-water periodic box, central O, real 6 Å sphere, exact neighbors, bond/neighbor views | fixed asymmetric camera and isotropic scale |
| c | learned potential | Large but simple descriptor → shared network → atomic energy → total energy/forces path | no decorative dashboard elements |

## Required Elements

- Prepared 64-water box with PBC and stable source IDs; explicitly not claimed equilibrated.
- Exact minimum-image 6.0 Å selection: 83 neighbors plus central atom.
- Chemical-bond and neighbor-centric views use stored topology and one camera.
- `D_i → epsilon_i`, `E=sum_i epsilon_i`, `F=-grad_R E`.
- Numerical energy and force marks must come only from the retained DeepMD inference result.

## Forbidden Changes

- No 111 camera, no independent auto-fit, no projected-distance neighbors.
- No structure inside the MD loop and no fake quantitative force arrows.
- No omission/addition of neighbors for appearance and no non-proportional stretching.

## QA Plan

- MatterVis inspect/preflight/render/verify with explicit asymmetric camera.
- Static source/composite clipping, boundaries, colors, 297 mm print scale, strict QA, five-problem review.
- Every video frame: visualize-data whitespace, clipping, boundaries, colors, registered text/arrow bounds, exact stable-ID selection and motion extrema.
- DeepMD evidence: `data/dpmd_water_box_results.npz`, `data/dpmd_eval.json`, and `bohrium/sandbox_run.json` must agree on atom count, energy, forces, cutoff, model hash, and net force.

## Delivery Gate

- [ ] Canonical outputs exist.
- [ ] Static strict QA and review hash pass.
- [ ] Every video frame passes.
- [ ] Neighbor and camera provenance is verified.
- [ ] Real DeepMD energy and force provenance is verified.
- [ ] Final static and motion are visually accepted.
