# 04 — Neural-network MD, end to end (DeepMD and DPA4C)

Two independent 25 s videos and two A4 stills share one renderer,
`scripts/md_visuals/render_nnmd_end_to_end.py`, and one layout. Only the
force provider changes between them:

| Stem | Force provider | Trajectory | Forward-pass dump | Video |
|---|---|---|---|---|
| `04_deep_potential_md` | DeepMD · DeepPot-SE, retrained uncompressed (`dpse_h2o_phase_diagram_retrain.pb`) | `product/data/dpmd_water_box_trajectory.npz` | `product/data/dpmd_internals.npz` | `product/videos/04_deep_potential_md.mp4` |
| `04_4c_dpa4c` | DPA4C · equivariant descriptor (`DPA4C-Neo-OMat24-v20260819.pt`) | `product/data/dpa4c_water_box_trajectory.npz` | `product/data/dpa4c_internals.npz` | `product/videos/04_4c_dpa4c.mp4` |

Both trajectories start from the same prepared 64-water box
(`product/data/water_box_64.npz`, 192 atoms, L = 12.4296 Å), the same Maxwell
velocities (300 K, seed 260906) and the same 0.5 fs step. Each trajectory holds
six states and five velocity-Verlet updates with a fresh model call at every new
position; energies, atomic energies, forces and virials are model outputs.

## Layout

```
┌──────────┬──────────────────────────────┬──────────────────────────────────┐
│ A        │ B  stage title               │ D  rows of O126's neighbours  │ Σ_j │
│ VV loop  │    box ── magnifier (O126)   │    matrix → embedding layers  │ D_i │
│ r → a → v│    j1..j3 labels   legend    │    (one row per neighbour)    │ fit │
│          │         ●●● fly into rows ──►│                               │ ε, E│
│          │                              │ ◄──── F = −∂E/∂r back through │     │
├──────────┴──────────────────────────────┴──────────────────────────────────┤
│ ◄──────── F enters the integrator: a = F/m ───────────────────────────────┘
└────────────────────────────────────────────────────────────────────────────┘
```

* **A** is the shared velocity-Verlet loop. The `a` node is where the
  pluggable force provider plugs in.
* **B** shows the real system: the periodic box with the 6 Å cutoff circle
  around O126, and a magnifier of the same snapshot with neighbour edges, ε
  colours, and force/acceleration/velocity/displacement arrows on every atom
  inside `r_c`. The circle, its guide lines and the magnified cutoff sphere
  are visible from the first frame. The magnifier keeps whole water
  molecules; chemical views draw bonds, atoms outside `r_c` are faded (a bond
  takes the fainter end), and the neighbour views show bare atoms. The step
  label reads `Simulation step NN (t fs)` at the bottom left; a colour key for
  the active arrow sits at the bottom right.
* **D** is the operator pipeline of O126, drawn from the model's own forward
  pass for the current state:
  * **gather** — all neighbours in the magnifier fly into their rows at the
    same time, so the matrix is assembled in one step. Rows follow the
    model's order: DeepMD sorts by species, then distance (O block, H block;
    the padded slots up to `sel = 200 O + 400 H` are not drawn). DPA4C rows
    are sorted by distance.
  * **embed** — every following block is the model's activation for the same
    rows. DeepMD: `R̃` (4) → `G¹` (25) → `G²` (50) → `G` (100), with
    separate O←O / O←H nets for the two blocks. DPA4C: `Y_lm` (9) and
    `e(r)` (16) → radial MLP hidden `h` (176) → `g` (64) → pair FiLM and
    envelope → amplitude `φ` (64).
  * **contract** — the neighbour sum (navy) and the neighbour sum
    builds up in real partial sums: DeepMD `T = R̃ᵀG/N` (4 × 100; the
    padded slots close the sum), then `D = TᵀT<` (100 × 12). DPA4C moments
    `X⁽⁰⁾` (64), `X⁽¹⁾` (3 × 8), `X⁽²⁾` (5 × 4), then the 208 invariants.
  * **fit** — the three hidden layers of the fitting net light up (teal),
    then the real `ε_O126`, then `E = Σ ε_i`; both lines are left-aligned.
  * **force** — every block frame turns orange: `F = −∂E/∂r` is the gradient
    through the same blocks. One solid path leaves the middle of the E line,
    runs down, left along the bottom strip, up the A|B gutter and into the
    right side of the loop's `a` node; it grows in orange during the force
    stage and is grey otherwise (`a = F/m` is not repeated in panel D).
* Colours: structure and descriptor navy, fitting/ε/E teal, force and
  acceleration orange, velocity purple, displacement blue — the same r/v/a
  colours as the loop and the other movies. Panel D carries only block
  symbols, the model name and the ε/E values (no tensor dimensions).
* Heatmaps are signed (navy negative, crimson positive). Each block saturates
  at its own 98th-percentile |value|; the partial-sum blocks use the largest
  |value| of their final sum.

## Timeline (25 s, 24 fps)

* Slow display: 12.9 s — positions, neighbours, gather, embed, contract, fit,
  energy, force, acceleration, velocity, move, followed by a short completed
  state hold.
* Fast scan: four update-capable states use direct stage changes at about half
  the former rapid-cycle duration; the final saved state is held as a clean
  positions frame. No slow-pass blending, neighbour-flight animation, or
  pulse animation is used in this phase.

## Model data

### DeepMD: retrained uncompressed DeepPot-SE

The published `H2O-Phase-Diagram-model_compressed.pb` (AIS Square, Zhang et
al., PRL 126, 236001) is only available compressed. Compression replaces the
embedding net with a polynomial table, so its 25- and 50-wide hidden layers do
not exist in the file. For this story the model was retrained **uncompressed**
with the same architecture and the same data:

* input: the training script stored in the published graph, unchanged
  (`se_e2_a`, `sel = [200, 400]`, `rcut = 6.0`, `rcut_smth = 0.5`,
  embedding `[25, 50, 100]`, `axis_neuron = 12`, fitting `[240, 240, 240]`
  with ResNet, tanh, float64, same seeds and loss prefactors), except
  `numb_steps = 600 000` (published: 16 000 000) and `decay_steps = 3000`
  (keeps the published 200 learning-rate decays);
* data: the full AIS Square `H2O-Phase-Diagram` dataset (324 systems);
* trained on one A800 with DeePMD-kit 2.2.8 (TensorFlow);
  600k steps, 8082 s wall time;
* `dp test` on 41 systems (every 8th, 20 frames each, list in
  `product/data/dpse_retrain/test_systems.txt`):
  retrained model energy RMSE 2.65 meV/atom, force RMSE 0.130 eV/Å;
  published compressed model on the same frames 2.17 meV/atom, 0.128 eV/Å.

The trajectory was then rerun with this model.

### Forward-pass dumps

* `scripts/run_md/dump_dpse_internals.py` (DeePMD-kit 2.x) reads the
  embedding and fitting weights from the frozen graph and recomputes O126's
  forward pass layer by layer in NumPy. It is accepted only if it reproduces
  the model's own descriptor and atomic energy; the errors are about 1e-15
  (`dpmd_internals.json`).
* `scripts/run_md/dump_dpa4c_internals.py` (DeePMD-kit 3.2, `pt_expt`)
  wraps the `DescrptDPA4C` stages and records the tensors the model itself
  produced (radial basis, radial MLP, pair FiLM, amplitudes, harmonics,
  moments, readout, descriptor), then runs the fitting layers on the recorded
  descriptor. Checks: amplitudes and moments recomputed from the recorded
  edges agree to ≤ 2e-6 (float32), and `ε` agrees with the model to ≤ 1e-6 eV
  (`dpa4c_internals.json`).

The renderer checks that every state's neighbour set and `ε_O126` agree with
the trajectory before it draws anything.

## Reproduce

```bash
PY=/c/Users/gmy72/AppData/Local/Programs/Python/Python312/python   # has mat_viewer
$PY scripts/md_visuals/render_nnmd_end_to_end.py --model deepmd --preview-only
$PY scripts/md_visuals/render_nnmd_end_to_end.py --model dpa4c --preview-only
$PY scripts/md_visuals/render_nnmd_end_to_end.py            # both models, stills + videos
```

Model-side runs (on the A800 node, `/aisi-nas/guomingyu/personal/mlip-playground/261002_dpse_internals`):

```bash
TF=/aisi-nas/guomingyu/conda_env/deepmd-v2.2.9/bin     # DeePMD-kit 2.2.8 (TF)
PT=/aisi-nas/guomingyu/conda_env/deepmd-dpa4-t211/bin  # DeePMD-kit 3.2.0 (dpa4c)
cd train && $TF/dp train input.json && $TF/dp freeze -o ../dpse_h2o_phase_diagram_retrain.pb && cd ..
$TF/python run_water_box_nnmd.py --model dpse_h2o_phase_diagram_retrain.pb --input water_box_64.npz \
  --output dpmd_water_box_trajectory.npz --metadata dpmd_water_box_trajectory.json --label dpmd
$TF/python dump_dpse_internals.py --model dpse_h2o_phase_diagram_retrain.pb \
  --trajectory dpmd_water_box_trajectory.npz --output dpmd_internals.npz
$PT/python dump_dpa4c_internals.py --model DPA4C-Neo-OMat24-v20260819.pt \
  --trajectory dpa4c_water_box_trajectory.npz --output dpa4c_internals.npz
```

The training input is kept as `product/data/dpse_retrain/input.json` (with
`lcurve.out`). The DPA4C trajectory itself comes from Bohrium job 20808156;
see `docs/04_4c_dpa4c.md`.

## QA

Every keyframe and every video frame is validated with the house
`LayoutRegistry` (Arial with the shared 12/14/16/18/24 pt scale, edge pads,
text overlaps) and with the `visualize_data` pixel checks. Reports live in
`product/qa/<stem>/_qa/`.
