# Molecular Dynamics Visualization

The canonical deliverables are organized under `product/`: A4 figures, independent PowerPoint-ready videos, scientific source data, and QA records. Rendering and data-generation entry points live under `scripts/`.

1. Velocity Verlet: exact position → acceleration → velocity update.
2. Classical potential: the 12–6 Lennard-Jones potential for a real Ar pair.
3. Ab initio MD: one H₂O-dimer force query containing a repeated SCF loop.
3b. UKS reactive AIMD: a kick-started TNT C–NO₂ homolysis story with α/β spin-density separation; real PBE0/def2-SVP UKS data are required for the scientific deliverable, while the analytic path remains a renderer-only fixture.
4. Deep Potential MD: a real 64-water periodic box, exact 6 Å neighborhood, learned atomic energies, DeepMD forces, and the full velocity-Verlet feedback (E → ∇ → F → a → updated v, r) over five real MD steps.
5. DPA4C MD (`04_4c`): the same box, velocities and time step with the force provider swapped for DPA4C-Neo-OMat24, evaluated live on Bohrium; the layout is identical so the pluggable module is obvious.

The visual grammar uses grey construction lines with sparse crimson, green, and navy accents. Static figures are 3508 × 2480 px at A4 landscape width. All compositions use Arial and the shared 12/14/16/18/24 pt typography scale. Videos are 1920 × 600 (exactly 16:5), 24 fps, H.264/yuv420p.

Older stories, workflows, media, and plans are preserved under `_archive/legacy/` for provenance; they are not the current presentation set.

## Active entry points

- `scripts/md_visuals/`: render figures, videos, MatterVis scenes, and QA.
- `scripts/build_box/`: build the reproducible H₂O, LJ, AIMD, and water-box source cases.
- `scripts/run_md/`: local MD/RHF engines and evaluation runs.
- `scripts/submit_calculation/`: Bohrium/DeepMD submission and worker scripts.
- `docs/`: visual documentation and review records.
- `report/`: the Chinese MD reading notes and references.
