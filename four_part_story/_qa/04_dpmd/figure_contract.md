# Figure Contract — Deep Potential MD

## Output

- Canonical stem: `04_deep_potential_md`
- Formats: PNG, SVG, MP4
- Final files: `figures/04_deep_potential_md.{png,svg}`, `videos/04_deep_potential_md.mp4`
- Intermediate files: Bohrium logs, trajectory, model provenance and MatterVis frames under `_qa/04_dpmd/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps, minimum 18 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| a | abstract integrator | same empty `r → a → v` loop | fixed across the series |
| b | learned local potential | equilibrated MatterVis 64-water box, selected atom, true 6 Å sphere, neighbour and model views | stable source IDs, one camera and viewport |

## Required Elements

- Bohrium DeepMD minimisation, 300 K NVT trajectory and one retained VV step.
- Chemical-bond view followed by neighbour-vector view; inside/outside sphere rendered differently.
- Real model energy/force arrays and exact minimum-image neighbour selection.

## Forbidden Changes

- No synthetic force arrows, screen-space neighbour selection, hand-drawn atoms/bonds or `[1,1,1]` camera.
- Do not destroy a Bohrium sandbox before all results and hashes are downloaded.

## QA Plan

- Verify model/type map, trajectory stability, atom IDs, PBC, neighbour mask and force arrays.
- MatterVis inspect/render JSON plus strict source/composite and every-frame checks.
- Inspect full contact sheets and final video motion.

## Delivery Gate

- [ ] Equilibrated trajectory and real inference provenance exist.
- [ ] Strict static and every-frame QA pass.
- [ ] Five-problem review matches the latest render hash.

