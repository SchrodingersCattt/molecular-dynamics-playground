# Four-part molecular-dynamics story

This directory contains independent figures and independent 16:5 videos. Each movie is rendered from the same scientific snapshot chain as its still, but is composed as its own visual explanation; no movie is a stitched montage or a screenshot of a still.

## Canonical outputs

| Part | Static figure | Video | Concrete case |
|---|---|---|---|
| 01 Velocity Verlet | `product/figures/01_velocity_verlet.png` / `.svg` | `product/videos/01_velocity_verlet.mp4` | one exact H₂O integration step |
| 02 Classical potential | `product/figures/02_classical_lj.png` / `.svg` | `product/videos/02_classical_lj.mp4` | TIP3P water dimer, O···O Lennard–Jones subterm (σ = 3.15061 Å, ε = 0.00659568 eV) |
| 03 Ab initio MD | `product/figures/03_aimd_scf.png` / `.svg` | `product/videos/03_aimd_scf.mp4` | H₂O dimer, RHF/STO-3G SCF density on a fixed molecular-plane grid, seven ionic geometries |
| 04 Deep Potential MD | `product/figures/04_deep_potential_md.png` / `.svg` | `product/videos/04_deep_potential_md.mp4` | 64-water periodic box, O126 and its 83 minimum-image neighbours inside 6.0 Å; six DeepMD (DeepPot-SE) states, five velocity-Verlet updates, end to end from positions to the next positions |
| 04_4c DPA4C MD | `product/figures/04_4c_dpa4c.png` / `.svg` | `product/videos/04_4c_dpa4c.mp4` | same box, velocities and time step; force provider swapped for DPA4C-Neo-OMat24 (equivariant descriptor), evaluated live on Bohrium |
| 05 Well-tempered metadynamics | `product/figures/05_well_tempered_metadynamics.png` / `.svg` | `product/videos/05_well_tempered_metadynamics.mp4` | one-dimensional double well, Langevin residence, tempered Gaussian hills, recovered free energy |
| 06 Rigid-water symmetry | `product/figures/06_rigid_water_descriptor_invariance.png` / `.svg` | `product/videos/06_rigid_water_descriptor_invariance.mp4` | one H₂O translated and rotated as a rigid body; full Cartesian coordinates change while the DeepMD environment matrix Rᵢ stays invariant |
| 06 Rigid-water invariance | `product/figures/06_rigid_water_descriptor_invariance.png` / `.svg` | `product/videos/06_rigid_water_descriptor_invariance.mp4` | one real H₂O rigidly translated and rotated in a periodic box; Cartesian coordinates change while the minimum-image descriptor is constant |

## Scientific evidence

- Velocity Verlet uses the exact three-stage form: update position with `a_n`, evaluate the new acceleration from the potential, then update velocity with the average acceleration.
- The classical-potential case uses the analytic 12–6 Lennard–Jones O–O term for the stored water-dimer separation. The displayed force is deliberately only this term; TIP3P also has electrostatics, which are not included in this panel.
- The AIMD case retains the complete per-ionic-step RHF/STO-3G density history on one fixed molecular-plane grid, the corresponding SCF residuals, seven nuclear geometries, and central-difference nuclear forces. The contour layer is therefore a labelled 2-D density slice, not a 3-D isosurface or a production DFT calculation.
- The Deep Potential case uses `product/data/dpmd_water_box_trajectory.npz`: six states of the 192-atom box propagated by full velocity Verlet with a fresh `H2O-Phase-Diagram-model_compressed.pb` call at every new position (`E` from −1007.907 to −1006.458 eV, max |F| from 1.22 to 3.30 eV Å⁻¹, numerically zero net force). The DPA4C case uses `product/data/dpa4c_water_box_trajectory.npz`, produced the same way with `DPA4C-Neo-OMat24-v20260819.pt` (deepmd-kit 3.2.0, Bohrium job 20808156; `E` from −906.019 to −905.323 eV). Atomic energies, forces and virials are model outputs; accelerations are `F/m`. See `docs/04_nnmd_end_to_end.md`.
- MatterVis provenance is retained for every atomistic structure. All 3D scenes use the fixed asymmetric orthographic direction `[1.55, -1.0, 0.62]`; no 111 view is used.

## Render

The retained scientific data allow the four outputs to be regenerated independently:

```bash
python scripts/md_visuals/render_velocity_verlet.py
python scripts/md_visuals/render_classical_lj.py
python scripts/md_visuals/render_aimd_scf.py
python scripts/md_visuals/render_nnmd_end_to_end.py --model deepmd
python scripts/md_visuals/render_nnmd_end_to_end.py --model dpa4c
python scripts/md_visuals/render_metadynamics.py
```

Use `--static-only` to regenerate only the A4 PNG and SVG for a part. `render_aimd_scf.py --preview-only` and `render_nnmd_end_to_end.py --preview-only` write phase-boundary keyframes (and a contact sheet) before the full movie; `render_nnmd_end_to_end.py` needs the Python 3.12 interpreter that has `mat_viewer` installed.

## QA

Each part has a figure contract, manifest, source-panel checks, MatterVis provenance, five-problem self-review, strict A4 report, representative video frames, and a compact every-frame report under `product/qa/<part>/`.

Publication gates require:

- 3508 × 2480 static PNG at 300 dpi-equivalent A4 width;
- minimum 10 pt text in static figures;
- 1920 × 600 (exact 16:5), 24 fps, H.264/yuv420p independent videos;
- 10–16 pt text in stills; Arial 16–18 pt text in movies;
- zero frame-level clipping, text overlap, boundary, whitespace, or semantic-colour errors;
- exact neighbor selection and real energy/force provenance where numerical values are shown;
- native MatterVis provenance for atoms, bonds, periodic cells, density overlays, and world-space vectors.
- the metadynamics demo has a strict static manifest and a complete 384-frame report under `product/qa/05_metadynamics/`.
- the DeepMD and DPA4C stories follow the AIMD layout (loop | box + magnifier | energy | E→∇→F→a→Δt chain with a return arrow); every quantity is drawn on the real atoms inside r_c, arrow scales and the ε colour range are data-driven and recorded in `product/qa/04_deep_potential_md/asset_manifest.json` and `product/qa/04_4c_dpa4c/asset_manifest.json`.
- the rigid-water symmetry proof uses PBC-unwrapped source geometry, 240 real MatterVis frames, complete O/H₁/H₂ Cartesian rows, and a DeepMD-style 2×4 environment matrix (R_i=[s,sx/r,sy/r,sz/r]); self-checks for bond lengths, angle, and matrix invariance are in `product/qa/06_symmetry_invariance/descriptor_provenance.json`.
- the DP descriptor stage follows the PPT sketch: a square water box, an in-place local circle, a magnified local circle with real green neighbour links, and the matrix/descriptor logic in the right rail.
- the rigid-water invariance demo uses minimum-image distances and records `max |ΔD|` plus all source coordinates in `product/qa/06_symmetry_invariance/descriptor_provenance.json`.

