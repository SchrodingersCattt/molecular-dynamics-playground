# Four-part molecular-dynamics story

This directory contains the current figure and video set. Each topic is an independent static figure plus an independent video; no video is a stitched montage or a screenshot of the static figure.

## Canonical outputs

| Part | Static figure | Video | Concrete case |
|---|---|---|---|
| 01 Velocity Verlet | `figures/01_velocity_verlet.png` / `.svg` | `videos/01_velocity_verlet.mp4` | one exact H₂O integration step |
| 02 Classical potential | `figures/02_classical_lj.png` / `.svg` | `videos/02_classical_lj.mp4` | 12–6 Lennard-Jones Ar pair, σ = 3.405 Å and ε = 0.0103 eV |
| 03 Ab initio MD | `figures/03_aimd_scf.png` / `.svg` | `videos/03_aimd_scf.mp4` | H₂O dimer, 19-iteration RHF/STO-3G SCF history, true 3D density isosurface, finite-difference forces |
| 04 Deep Potential MD | `figures/04_deep_potential_md.png` / `.svg` | `videos/04_deep_potential_md.mp4` | 64-water periodic box, source atom 126, 83 exact minimum-image neighbors inside 6.0 Å |

## Scientific evidence

- Velocity Verlet uses the exact three-stage form: update position with `a_n`, evaluate the new acceleration from the potential, then update velocity with the average acceleration.
- The classical-potential case uses the analytic 12–6 Lennard-Jones energy and force for the stored Ar separation.
- The AIMD case retains 19 real SCF energies, six three-dimensional AO-density volumes, and central finite-difference nuclear forces. It is a pedagogical RHF/STO-3G calculation, not production DFT.
- The Deep Potential case was evaluated with `H2O-Phase-Diagram-model_compressed.pb` and DeepMD-kit 3.1.3 in a time-limited Bohrium sandbox. The retained 192-atom result gives `E = -1007.9069226533904 eV`, `max |F| = 1.2246452541839372 eV Å⁻¹`, and numerically zero net force. The sandbox was destroyed after the outputs were downloaded and hashed.
- MatterVis provenance is retained for every atomistic structure. All 3D scenes use the fixed asymmetric orthographic direction `[1.55, -1.0, 0.62]`; no 111 view is used.

## Render

The retained scientific data allow the four outputs to be regenerated independently:

```bash
python four_part_story/render_velocity_verlet.py
python four_part_story/render_classical_lj.py
python four_part_story/render_aimd_scf.py
python four_part_story/render_dpmd.py
```

Use `--static-only` to regenerate only the A4 PNG and SVG for a part.

## QA

Each part has a figure contract, manifest, source-panel checks, MatterVis provenance, five-problem self-review, strict A4 report, representative video frames, and a compact every-frame report under `_qa/<part>/`.

Publication gates require:

- 3508 × 2480 static PNG at 300 dpi-equivalent A4 width;
- minimum 10 pt text in static figures;
- 1920 × 1080, 24 fps, H.264/yuv420p independent videos;
- minimum 18 pt video text;
- zero frame-level clipping, text overlap, boundary, whitespace, or semantic-colour errors;
- exact neighbor selection and real energy/force provenance where numerical values are shown.
