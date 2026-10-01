# 04_4c — DPA4C local environment to MD force

The 04_4c deliverable (`product/videos/04_4c_dpa4c.mp4`,
`product/figures/04_4c_dpa4c.png/.svg`) is rendered by
`scripts/md_visuals/render_nnmd_end_to_end.py --model dpa4c`. Layout, timeline
and QA are shared with the DeepMD story and documented in
`docs/04_nnmd_end_to_end.md`; this note records what is specific to DPA4C.

## Data

* Model: `DPA4C-Neo-OMat24-v20260819.pt` (AIS Square resource 433, CC BY-NC
  4.0), `descriptor.type = dpa4c`, r_c = 6.0 Å, 64 channels, l_max = 2, full
  periodic-table type map. Files and hashes:
  `product/data/dpa4c/omat24_neo/`.
* Trajectory: `product/data/dpa4c_water_box_trajectory.npz` + `.json`
  (schema `nnmd_periodic_trajectory/v2`), six states of the 64-water box,
  five velocity-Verlet updates with a fresh DPA4C call at every new position.
  Total energy −906.019 → −905.323 eV, max |F| 1.56 → 3.25 eV Å⁻¹, atomic
  energies sum to the total, net force ≈ 1e-16 eV Å⁻¹.
* The same prepared box, seed and time step were used for the DeepMD run, so
  the two stories differ only in the force provider.

## What the video shows for the descriptor stage

DPA4C builds its features from the smooth-cutoff neighbourhood and the unit
vectors `û_ij = r_ij / |r_ij|` contracted into l ≤ 2 equivariant channels. The
video therefore labels the three nearest neighbours with their real `r` and
`û`, draws the 83 O126 edges inside `r_c`, and then colours the atoms by their
model atomic energies. The internal 64-channel tensors are not plotted; the
stage title names the operation and the atoms show its inputs and outputs.

## How the trajectory was produced (Bohrium job 20808156)

`product/qa/04_4c/bohr_live_water_v2/` holds `job.json`, `input/run.sh`, the
runner copy, every submit response and every job log.

* Image `registry.dp.tech/dptech/dp/native/prod-4392433/dpmd-cu126-pt:v20260701-pt`
  on `c8_m32_1 * NVIDIA V100`. Its `/opt/mamba/envs/dpmd` python ships
  deepmd 3.2.0b1.dev87 with torch 2.12.1+cu126, which predates the `dpa4c`
  descriptor (`Unknown descriptor type: dpa4c`).
* Compute nodes have no PyPI access (the mirror returned 403), so `run.sh`
  installs `deepmd-kit==3.2.0` and `mpich` from wheels shipped in
  `input/wheels/` (not versioned; see `.gitignore`).
* The PyPI wheel's compiled ops were built against torch 2.11.0; loading them
  under torch 2.12.1 fails. `run.sh` therefore keeps the image's compiled
  `deepmd/lib` and only takes the Python sources from the 3.2.0 wheel — the
  `dpa4c` descriptor is pure Python.
* The run then evaluates on the GPU (`cuda True`), with a CPU fallback in the
  script in case the device is reported busy.

Earlier attempts (sandbox images without `dpa4c`, nonexistent machine types,
`ModuleNotFoundError: deepmd` for the default python, mpich metadata) are in
`docs/04_4c_bohrium_infra_notes.md` and the retained logs.
