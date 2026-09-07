# Molecular Dynamics Visualization

The canonical deliverables are organized under `product/`: A4 figures, independent PowerPoint-ready videos, scientific source data, and QA records. Rendering and data-generation entry points live under `scripts/`.

1. Velocity Verlet: exact position → acceleration → velocity update.
2. Classical potential: the 12–6 Lennard-Jones potential for a real Ar pair.
3. Ab initio MD: one H₂O-dimer force query containing a repeated SCF loop.
4. Deep Potential MD: a real 64-water periodic box, exact 6 Å neighborhood, learned atomic energies, and retained DeepMD forces.

The visual grammar uses grey construction lines with sparse crimson, green, and navy accents. Static figures are 3508 × 2480 px at A4 landscape width with a 10 pt minimum font. Videos are 1920 × 600 (exactly 16:5), 24 fps, H.264/yuv420p, with Arial text at 16–18 pt.

Older stories, workflows, media, and plans are preserved under `_archive/legacy/` for provenance; they are not the current presentation set.

## Active entry points

- `scripts/md_visuals/`: render figures, videos, MatterVis scenes, and QA.
- `scripts/build_box/`: build the reproducible H₂O, LJ, AIMD, and water-box source cases.
- `scripts/run_md/`: local MD/RHF engines and evaluation runs.
- `scripts/submit_calculation/`: Bohrium/DeepMD submission and worker scripts.
- `docs/`: visual documentation and review records.
- `report/`: the Chinese MD reading notes and references.
