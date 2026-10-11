# Figure Contract — 03b UKS Reactive AIMD

## Output

- Canonical stem: `03b_uks_reaction`
- Static: `product/figures/03b_uks_reaction.{png,svg}`
- Video: `product/videos/03b_uks_reaction.mp4` (30 s, 24 fps)
- Source data: `product/data/uks_tnt_reaction.npz/.json` plus `product/data/uks_tnt_reaction_density3d.npz`

## Scientific contract

- Reaction label: `TNT → aryl radical + ·NO2`, defined as C2–NO2 homolytic dissociation.
- Total charge: 0; total electron count: 116; BS singlet convention: `spin=0`.
- The aryl fragment and NO2 receive mass-weighted opposite initial velocities so the centre of mass remains stationary.
- A real dataset must have `backend=pyscf_uks`. `backend=analytic_demo` is a renderer/layout fixture and must remain visibly labelled.
- Real UKS data must contain converged alpha/beta density matrices, UKS gradients, SCF histories, `<S²>`, and fixed-grid 3-D alpha/beta density fields in electron/Angstrom^3. Cube exports convert the same field to electron/Bohr^3 without changing atom coordinates.

## Visual contract

- Left: the same velocity-Verlet rail and active stage logic as 03 AIMD.
- Centre: fixed-camera native MatterVis TNT structure, with line-only alpha/beta contours from a fixed-grid 3-D maximum projection along the camera normal, or a force/velocity/position asset. The projected density and nuclei share the same Cartesian origin and axes; it is not a single aromatic-plane contour slice or a filled volume.
- Upper right: the current ionic step's UKS SCF residual curve, held during force/velocity/position updates.
- Lower right: the alpha/beta UKS self-consistency loop.
- Static size is 3508 × 2480 px; video size is 1920 × 600 px at 24 fps.

## QA gates

- Dataset invariant checks pass for electron count, charge, centre-of-mass velocity, NO2 continuity, and monotonic C2–N2 kick direction.
- Static layout validation passes with no text overlap or clipping.
- The fast-export MP4 is checked at representative times in `product/qa/03b_uks_reaction/_qa/sampled_frame_qa.json`; a full strict 720-frame audit remains a separate optional pass.

