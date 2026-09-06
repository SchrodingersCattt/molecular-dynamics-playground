# Figure Contract — Deep Potential MD

## Output

- Canonical stem: `04_deep_potential_md`
- Formats: PNG, SVG, MP4
- Final files: `figures/04_deep_potential_md.{png,svg}`, `videos/04_deep_potential_md.mp4`
- Intermediate files: Bohrium logs, trajectory, model provenance and MatterVis frames under `_qa/04_dpmd/`

## Target Size

- Static: A4 landscape, 297 × 210 mm, 300 dpi, minimum 10 pt
- Video: 1920 × 1080, 24 fps; all text 10–16 pt

## Panels

| Panel | Role | Required content | Comparison constraints |
|---|---|---|---|
| A | abstract integrator | same `r → a → v` loop with the active equation inside | fixed across the series |
| B | learned local potential | 64-water periodic box, top physical-action label and lower-left `Simulation step` | stable source IDs, one camera and viewport |
| C | local environment | selected centre, 6 Å cutoff and minimum-image neighbour count | fixed upper-right slot |
| D | learned model | `descriptor → shared network → atomic energy → energy/force` | fixed lower-right slot |

## Required Elements

- Bohrium DeepMD minimisation, 300 K NVT trajectory and one retained VV step.
- Chemical-bond view followed by neighbour-vector view; inside/outside sphere rendered differently.
- Real model energy/force arrays and exact minimum-image neighbour selection.

## Forbidden Changes

- No synthetic force arrows, screen-space neighbour selection, hand-drawn atoms/bonds or `[1,1,1]` camera.
- Do not destroy a Bohrium sandbox before all results and hashes are downloaded.
- No page header; one type family throughout; every label is between 10 and 16 pt.

## QA Plan

- Verify model/type map, trajectory stability, atom IDs, PBC, neighbour mask and force arrays.
- MatterVis inspect/render JSON plus strict source/composite and every-frame checks.
- Inspect full contact sheets and final video motion.

## Delivery Gate

- [ ] Equilibrated trajectory and real inference provenance exist.
- [ ] Strict static and every-frame QA pass.
- [ ] Five-problem review matches the latest render hash.
