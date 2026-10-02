# 04 — Neural-network MD, end to end (DeepMD and DPA4C)

Two independent 30 s videos and two A4 stills share one renderer,
`scripts/md_visuals/render_nnmd_end_to_end.py`, and one layout. Only the
force-provider box changes between them:

| Stem | Force provider | Trajectory | Video | Still |
|---|---|---|---|---|
| `04_deep_potential_md` | DeepMD · DeepPot-SE (`H2O-Phase-Diagram-model_compressed.pb`) | `product/data/dpmd_water_box_trajectory.npz` | `product/videos/04_deep_potential_md.mp4` | `product/figures/04_deep_potential_md.png/.svg` |
| `04_4c_dpa4c` | DPA4C · equivariant descriptor (`DPA4C-Neo-OMat24-v20260819.pt`) | `product/data/dpa4c_water_box_trajectory.npz` | `product/videos/04_4c_dpa4c.mp4` | `product/figures/04_4c_dpa4c.png/.svg` |

Both trajectories start from the same prepared 64-water box
(`product/data/water_box_64.npz`, 192 atoms, L = 12.4296 Å), the same Maxwell
velocities (300 K, seed 260906) and the same 0.5 fs step. Each trajectory holds
six states and five velocity-Verlet updates with a fresh model call at every new
position; energies, atomic energies, forces and virials are model outputs, not
fits. Accelerations are derived as `a = F / m` with eV Å⁻¹ amu⁻¹ → Å fs⁻².

## Layout (mirrors 03 AIMD)

```
┌──────────┬─────────────────────────────────────┬────────────────────┐
│ A        │ B  stage title                       │ C  E(step) plot    │
│ VV loop  │    64-H2O box ──guide── magnified    ├────────────────────┤
│ r → a → v│    with r_c      O126 environment    │ D  E → ∇ → F → a → Δt │
│          │    descriptor rows   legend          │    (real O126 numbers) │
├──────────┴─────────────────────────────────────┴────────────────────┤
│ ◄──────── updated v and r return to the integrator ─────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

* **A** is the shared velocity-Verlet loop (`draw_vv_loop`). The active node
  follows the stage; the `a` node is where the pluggable force provider sits.
* **B** shows the real system twice: the whole periodic box with the 6 Å
  cutoff circle drawn in place around O126, and a magnifier of the same
  snapshot. Every quantity is drawn on the atoms themselves:
  * neighbours: navy edges from O126 to its 83 minimum-image neighbours,
    atoms outside `r_c` faded;
  * descriptor: for the three nearest neighbours the real `r`, and either the
    DeepPot-SE row `[s(r), s·x/r, s·y/r, s·z/r]` (DeepMD) or the unit vector
    `û` that feeds the l ≤ 2 equivariant channels (DPA4C);
  * network → energy: atom colour = `ε_j − mean ε(species)`, navy–grey–crimson,
    saturating at the largest deviation seen inside `r_c` (0.96 eV for DeepMD,
    0.26 eV for DPA4C);
  * force, acceleration, half-step velocity, displacement: world-space arrows
    on every atom inside `r_c`, scaled so the longest arrow of the whole
    story is 1.9 Å; the legend quotes the real magnitude of that arrow.
    Acceleration arrows make the 16× O/H mass ratio visible.
* **C** is the real total energy against MD step, revealed as the run
  proceeds.
* **D** animates the operator flow for O126 rather than listing formulas.
  The top row runs left to right: a heatmap of the first 12 real `R_i` rows
  (DPA4C: columns `s, û`) fills in during the descriptor stage, the schematic
  network lights up layer by layer, and bars for `ε_i − mean ε(species)` of
  O126 and j₁–j₃ grow. Bar heights are relative to the largest of the four
  bars, and colours use the magnifier's absolute scale. Σ then feeds the real
  total energy. The bottom row runs right to left: pulses travel along the
  dashed `−∂E/∂r` arrow to a force glyph. The glyph's direction is the real
  O126 force projected into the magnifier camera. The flow then passes
  through `÷m` to the acceleration glyph and exits along the bottom back into
  the loop's `a` node.
* Neighbour edges and the j₁/j₂/j₃ labels appear together in one stage.

## Timeline (30 s, 24 fps)

* Steps 1 and 2: 9 s detailed blocks — positions, neighbours, descriptor,
  network, energy, force, acceleration, velocity, move.
* Steps 3–5: 1.5 s rapid cycles through the same stages.

## Reproduce

```bash
PY=/c/Users/gmy72/AppData/Local/Programs/Python/Python312/python   # has mat_viewer
$PY scripts/md_visuals/render_nnmd_end_to_end.py --model deepmd --preview-only
$PY scripts/md_visuals/render_nnmd_end_to_end.py --model dpa4c --preview-only
$PY scripts/md_visuals/render_nnmd_end_to_end.py            # both models, stills + videos
```

MatterVis renders are cached under `product/qa/<stem>/source/mattervis_v1/`
with JSON sidecars; `asset_manifest.json` records the camera, the data-driven
arrow scales and the colour range; `story_provenance.json` records the labelled
neighbours, the energies and the central-atom force/acceleration per state.

### Regenerating the trajectories

`scripts/run_md/run_water_box_nnmd.py` is the shared runner (numpy + deepmd):

```bash
python scripts/run_md/run_water_box_nnmd.py --model <model.pb|model.pt> \
  --input product/data/water_box_64.npz \
  --output product/data/<label>_water_box_trajectory.npz \
  --metadata product/data/<label>_water_box_trajectory.json \
  --label <label> --steps 5 --dt 0.5 --temperature 300 --seed 260906
```

The DPA4C run needs a deepmd-kit that ships the `dpa4c` descriptor
(3.2.0 release). The Bohrium job that produced the retained trajectory, its
`run.sh` (wheel-based upgrade inside the `dpmd-cu126-pt:v20260701-pt` image)
and the full stdout are kept under `product/qa/04_4c/bohr_live_water_v2/`
(job 20808156). See `docs/04_4c_dpa4c.md` for the details and the pitfalls hit
along the way.

## QA

`render_nnmd_end_to_end.py` validates every keyframe and every video frame
with the house `LayoutRegistry` (Arial 16–18 pt in video, ≥ 10 pt in stills,
edge pads, text overlaps) and with the `visualize_data` pixel checks
(whitespace bands, clipping, semantic colours). Reports live in
`product/qa/<stem>/_qa/`.
