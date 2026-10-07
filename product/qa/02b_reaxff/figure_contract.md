# Figure Contract — 02b Schematic ReaxFF (TNT C2–NO2 departure)

## Output

- Canonical stem: `02b_reaxff`
- Static: `product/figures/02b_reaxff.{png,svg}`
- Video: `product/videos/02b_reaxff.mp4` (21.6 s, 24 fps)
- Source data: `product/data/02b_reaxff.npz/.json` (generator `scripts/build_box/generate_reaxff_tnt.py`)
- Renderer: `scripts/md_visuals/render_reaxff.py`

## Scientific contract

- Same TNT atom map, starting geometry and initial kick as 03b (`generate_uks_tnt.initial_geometry`, `kick_velocities`, 0.05 Å/fs relative speed, `dt = 0.1 fs`, 500 steps).
- The energy is a **schematic ReaxFF-style model**: sigma + pi bond order, `E_bond = -De BO exp[p_be1 (1 - BO^p_be2)]`, over-coordination penalty from `Δ_i = Σ_j BO_ij − Val_i`, valence-angle energy weighted by two bond orders, EEM charges with shielded Coulomb, exponential core repulsion.
- It is not a published ReaxFF parameter set; no LAMMPS run, no UKS rerun. The word `Schematic` stays visible in the energy panel header.
- Forces are exact autograd gradients (float64), checked against central differences (< 1e-8 eV/Å). The trajectory is velocity Verlet on those forces; the total energy drift stays below 1e-4 eV.
- The selected C2–N2 bond is the weak teaching bond and fades under the kick while the aryl/NO2 radical sites appear; this remains a schematic model, not a published ReaxFF parameter set.

## Visual contract

- Left: shared velocity-Verlet loop with the active stage.
- Centre: fixed-camera MatterVis ball-and-stick with native bonds at the 03/03b sizes (atom scale 0.90, bond radius 0.102 Å). Only the breaking Cl–OH bond is restyled: dark ink, opacity and thickness follow its bond order, and it is kept past MatterVis's distance cutoff until BO < 0.05. Energy-term cues share one teal: dashed coordination rings on under-coordinated Cl and O, the BO-weighted Cl–O–H angle arc, and `δ+` / `δ−` labels for the sign of the EEM charges (placed in the widest gap between each atom's bonds). A dimension line marks `r_ij`; orange arrows mark forces, purple arrows mark velocities, with a colour key at the bottom right.
- Upper right: bond order versus distance with one dot per bond and a legend (`aromatic C–C` grey, `C2–NO2 (breaking)` dark ink); top and right spines hidden, black axes.
- Lower right: `Schematic ReaxFF energy` written out term by term — `BO_ij(r_ij)`, `E_bond(BO)`, `E_over(Δ_i)`, `E_angle(BO, θ)`, `E_Coul(q, r)` and `F_i = −∂E/∂r_i, E = Σ E_term`; the line for the active stage is highlighted.
- Timeline: 9.6 s detailed first step (Distances, Bond orders, Energy terms, Forces, Update velocity, Update position), then 15 fast steps of 0.8 s.
- Static size 3508 × 2480 px (energy lines in a band under the scene and plot); video 1920 × 600 px at 24 fps; Arial, PPT points 14 / 16 / 18 / 24, no bold except vectors.

## QA gates

- Dataset checks: finite-difference force error, energy drift, BO(C2–N2) final < 0.01, `r_C2N2` monotonic increase, and no spurious O–O topology in the mat-vis structure preflight.
- Static layout validation passes; keyframe contact sheet reviewed; decoded-frame sampled QA of the MP4 in `product/qa/02b_reaxff/_qa/sampled_frame_qa.json` (all samples pass). The every-frame `--strict-video` path is kept for regression and shares 03b's bottom-label whitespace behaviour.
