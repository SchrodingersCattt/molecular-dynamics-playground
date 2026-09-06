# Three-slide molecular-dynamics story

Three white-background figures and three independent, PowerPoint-ready videos:

1. Velocity Verlet around a real H2O step;
2. AIMD force evaluation for an H2O dimer;
3. Deep-Potential force evaluation in a periodic water box.

The visual grammar is fixed across all three outputs: light-grey structure,
black active nodes, and sparse navy/crimson/green physical accents.

## Final outputs

| Story | Static figure | Independent video | Real case |
|---|---|---|---|
| Velocity Verlet | `figures/01_velocity_verlet.png` / `.svg` | `videos/01_velocity_verlet.mp4` | one H2O integration step |
| AIMD / SCF | `figures/02_aimd_scf.png` / `.svg` | `videos/02_aimd_scf.mp4` | generalized RHF/STO-3G H2O dimer calculation |
| Deep Potential MD | `figures/03_deep_potential_md.png` / `.svg` | `videos/03_deep_potential_md.mp4` | 64-H2O periodic box and exact 6.0 Å local environment |

All raster figures and videos are 1920 × 1080. Videos use H.264/yuv420p
at 24 fps and are separate files rather than a stitched montage.

## Render

```bash
python three_slide_story/render_slide1_velocity_verlet.py
python three_slide_story/render_slide2_aimd_scf.py
python three_slide_story/render_slide3_deep_potential.py
```

Use `--static-only` with any renderer to regenerate only PNG and SVG.

Data-generation scripts are kept beside the renderers:

```bash
python three_slide_story/compute_h2o_dimer_scf.py
python three_slide_story/generate_water_box.py
```

## Scientific scope

- Slide 1 reads the real H2O state stored in `md_workflows/classical_data.npz`.
- Slide 2 uses an actual converged 19-iteration SCF history and finite-difference
  dimer forces; the calculation is pedagogical RHF/STO-3G, not production DFT.
- Slide 3 uses stable atom IDs and exact three-dimensional minimum-image
  distances. The 64-water box is a deterministic visualization structure at
  0.997 g cm^-3, not a claimed equilibrated trajectory frame.
- A Bohrium DeePMD inference was configured and passed local dry-run validation,
  but the Windows uploader failed before job creation. Therefore slide 3 shows
  symbolic `E` and `F = -grad E` only—no fabricated numerical model forces.

Each slide has a figure contract, provenance, extracted video frames, a
five-problem visual review, and strict machine-readable QA under `_qa/`.
