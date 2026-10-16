# 05 — Neural-network MD on a TNT crystal box (DPA1 and DPA4C), plus the 05a training step

The 05 series uses the same layout as the 04 series, with a TNT crystal box in place
of the 64-H2O box, so the series stays chemically consistent with the TNT
reaction scenes (02b/03b).

| Stem | Mirrors | Force provider | Renderer |
|---|---|---|---|
| `05_tnt_deep_potential_md` | `04_deep_potential_md` | DeepMD · DPA1 `se_atten_v2` (`deepems-25.pb`, TF, compressed) | `scripts/md_visuals/render_05_tnt_nnmd.py --model deepmd` |
| `05_4c_tnt_dpa4c` | `04_4c_dpa4c` | DPA4C-Neo, 05e EMs model (`model.ckpt-1000000.pt`, seed 42) | `scripts/md_visuals/render_05_tnt_nnmd.py --model dpa4c` |
| `05a_tnt_training` | `04a_training` | schematic training step (teaching fixture) | `scripts/md_visuals/render_05a_tnt_training.py` |

Videos: `product/videos/<stem>.mp4` (1920 × 600, 24 fps); stills: `product/figures/<stem>.png/.svg`.

## System

* TNT, CSD ZZZMUC01 (orthorhombic Pca2₁, 295 K X-ray), 1×2×1 supercell:
  14.991 × 12.154 × 20.017 Å, 336 atoms, 16 molecules (C112H80N48O96).
* Equilibrated with DPA4C-Neo (05e): Langevin NVT 300 K, dt 0.5 fs, 1 ps + 4 ps
  (a800 GPU 0). The last production frame (positions + velocities, 318 K
  instantaneous) is the starting state of both 05 stories.
* Centre atom: the nitro N closest to the box centre (index 203); it is shown as
  "the central N" — no atom-index labels. 88 neighbours inside r_c = 6 Å at every state.
* Each story holds six states / five velocity-Verlet updates (0.5 fs) with a fresh
  model call per state, as in 04.

## Forward-pass dumps

* DPA1 (`dump_dpa1_internals.py`, DeePMD-kit 2.2.8 TF): the shipped model is
  compressed, so the embedding hidden layers (G¹, G²) do not exist in the graph.
  Panel D therefore shows R̃ → radial embedding G_s(s) → pair type embedding τ_ij →
  G = G_s(1 + τ_ij) (checked to 1e-15), recovered by re-running the graph's own
  `TabulateFusionSeAtten` op with one-hot environment rows. The neighbour sum over
  all 250 slots reproduces the graph contraction (2e-14), `D = TᵀT<` reproduces
  `o_descriptor` (6e-17), and ε_N equals `o_atom_energy`. DPA1 rows are mixed-type and
  distance-sorted, so there are no species blocks.
* DPA4C (`dump_dpa4c_internals.py`, `pt_expt`): the tensors the model produced;
  moments recomputed from the dumped amplitudes/harmonics agree to ≤ 1.2e-6 and
  ε_N to ≤ 7e-7 eV. This checkpoint has 32 channels and 192-wide fitting layers; the
  renderer reads the angular index layout from the dump.

## Data (not versioned)

All trajectory data live in `data/05_tnt_dpa4c/` (git-ignored; copied by scp from
`/aisi-nas/guomingyu/personal/mlip-playground/05_tnt_dpa4c/`):
`tnt_nvt_300K_prod.extxyz` (4 ps, 801 frames), `nnmd05/*_tnt_box_trajectory.npz/.json`,
`nnmd05/*_tnt_internals.npz/.json`. Copy them back before re-rendering.

## Reproduce (a800, one GPU)

```bash
cd /aisi-nas/guomingyu/personal/mlip-playground/05_tnt_dpa4c
PT=/aisi-nas/guomingyu/conda_env/deepmd-dpa4-t211-260908/bin; TF=/aisi-nas/guomingyu/conda_env/deepmd-v2.2.9/bin
export CUDA_VISIBLE_DEVICES=0
$PT/python run_md.py                                  # 300 K NVT, 5 ps
$PT/python scripts/build_start.py                     # -> tnt_box_336.npz
cd nnmd05
$PT/python ../scripts/run_tnt_nnmd.py --model ../model/model.ckpt-1000000.pt --input ../tnt_box_336.npz --output dpa4c_tnt_box_trajectory.npz --metadata dpa4c_tnt_box_trajectory.json --label dpa4c
$PT/python ../scripts/dump_dpa4c_internals.py --model ../model/model.ckpt-1000000.pt --trajectory dpa4c_tnt_box_trajectory.npz --output dpa4c_tnt_internals.npz
$TF/python ../scripts/run_tnt_nnmd.py --model ../model_dpa1/deepems-25.pb --input ../tnt_box_336.npz --output dpa1_tnt_box_trajectory.npz --metadata dpa1_tnt_box_trajectory.json --label dpa1
$TF/python ../scripts/dump_dpa1_internals.py --model ../model_dpa1/deepems-25.pb --trajectory dpa1_tnt_box_trajectory.npz --output dpa1_tnt_internals.npz
```

Local (Scat): `python scripts/md_visuals/render_05_tnt_nnmd.py` then
`python scripts/md_visuals/render_05a_tnt_training.py`.

## 05a

Same layout, timeline and revised box wording as 04a ("DFT reference",
"Neural-network prediction", "Error"). The per-atom numbers in
`product/data/05a_tnt_training_demo.json` (central nitro N and its two O) are
a schematic teaching fixture, labelled as such; they are neither DFT nor model output.
