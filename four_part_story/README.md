# Four-part molecular-dynamics story

This directory contains four independent figures and four independent 16:9 videos. Each movie is rendered from the same scientific snapshot chain as its still, but is composed as its own visual explanation; no movie is a stitched montage or a screenshot of a still.

## Canonical outputs

| Part | Static figure | Video | Concrete case |
|---|---|---|---|
| 01 Velocity Verlet | `figures/01_velocity_verlet.png` / `.svg` | `videos/01_velocity_verlet.mp4` | one exact H₂O integration step |
| 02 Classical potential | `figures/02_classical_lj.png` / `.svg` | `videos/02_classical_lj.mp4` | TIP3P water dimer, O···O Lennard–Jones subterm (σ = 3.15061 Å, ε = 0.00659568 eV) |
| 03 Ab initio MD | `figures/03_aimd_scf.png` / `.svg` | `videos/03_aimd_scf.mp4` | H₂O dimer, RHF/STO-3G SCF density on a fixed molecular-plane grid, seven ionic geometries |
| 04 Deep Potential MD | `figures/04_deep_potential_md.png` / `.svg` | `videos/04_deep_potential_md.mp4` | 64-water periodic box, center atom O126, 83 atomic minimum-image neighbors inside 6.0 Å |

## Scientific evidence

- Velocity Verlet uses the exact three-stage form: update position with `a_n`, evaluate the new acceleration from the potential, then update velocity with the average acceleration.
- The classical-potential case uses the analytic 12–6 Lennard–Jones O–O term for the stored water-dimer separation. The displayed force is deliberately only this term; TIP3P also has electrostatics, which are not included in this panel.
- The AIMD case retains the complete per-ionic-step RHF/STO-3G density history on one fixed molecular-plane grid, the corresponding SCF residuals, seven nuclear geometries, and central-difference nuclear forces. The contour layer is therefore a labelled 2-D density slice, not a 3-D isosurface or a production DFT calculation.
- The Deep Potential case was evaluated with `H2O-Phase-Diagram-model_compressed.pb` and DeepMD-kit 3.1.3 in a time-limited Bohrium sandbox. The retained 192-atom result gives `E = -1007.9069226533904 eV`, `max |F| = 1.2246452541839372 eV Å⁻¹`, and numerically zero net force. The displayed propagation is a reproducible frozen-force Velocity–Verlet step from that exact snapshot (the second force call is not claimed); the approximation is recorded in `_qa/04_dpmd_native/vv_metadata.json`.
- MatterVis provenance is retained for every atomistic structure. All 3D scenes use the fixed asymmetric orthographic direction `[1.55, -1.0, 0.62]`; no 111 view is used.

## Render

The retained scientific data allow the four outputs to be regenerated independently:

```bash
python four_part_story/render_velocity_verlet.py
python four_part_story/render_classical_lj.py
python four_part_story/render_aimd_scf.py
python four_part_story/render_dpmd_native.py
```

Use `--static-only` to regenerate only the A4 PNG and SVG for a part. `render_aimd_scf.py --preview-only` writes phase-boundary keyframes before the full movie.

## QA

Each part has a figure contract, manifest, source-panel checks, MatterVis provenance, five-problem self-review, strict A4 report, representative video frames, and a compact every-frame report under `_qa/<part>/`.

Publication gates require:

- 3508 × 2480 static PNG at 300 dpi-equivalent A4 width;
- minimum 10 pt text in static figures;
- 1920 × 1080, 24 fps, H.264/yuv420p independent videos;
- 10–16 pt text in both stills and movies (the same compact type scale);
- zero frame-level clipping, text overlap, boundary, whitespace, or semantic-colour errors;
- exact neighbor selection and real energy/force provenance where numerical values are shown;
- native MatterVis provenance for atoms, bonds, periodic cells, density overlays, and world-space vectors.
