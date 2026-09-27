# 04_4c — DPA4C local environment to MD force

The independent 04_4c deliverable is rendered by
`scripts/md_visuals/render_dpa4c.py`. It uses the retained 64-water periodic
case and makes the DPA4C local calculation explicit: a smooth cutoff
neighbourhood, radial and directional features, low-order equivariant channels,
rotational contractions, a shared fitting readout, and the energy-to-force
derivative.

## Current status

The checked-in video uses a real returned LAMMPS trajectory from the
DPA4C-Neo OMat24 four-model ensemble (`1008` atoms, `0.5 fs` timestep, first
`48` frames retained for the 16 s slow-motion render). The source trajectory,
source hash, pair style, ensemble count, checkpoint and configuration are all
recorded under `product/data/dpa4c/omat24_neo/` and `product/qa/04_4c/`.

The returned dump contains positions and velocities but no per-frame energy or
force columns. The video therefore does not invent numerical force values: it
shows the real DPA4C trajectory and the descriptor-to-energy-to-force
calculation path, with that limitation written in the manifest.

## Reproduce

```powershell
python scripts/md_visuals/render_dpa4c.py --preview-only
python scripts/md_visuals/render_dpa4c.py
python -m pip install git+https://github.com/SchrodingersCattt/aissq-explorer.git
python scripts/submit_calculation/fetch_dpa4c_omol.py --resource-name DPA4C-OMat24
python scripts/submit_calculation/run_dpa4c_md.py --model product/data/dpa4c/<downloaded-model>
```

`fetch_dpa4c_omol.py` uses the public `AissqClient` flow from
`SchrodingersCattt/aissq-explorer`: search the AIS Square model catalogue,
resolve the exact resource, then stream every file returned by the model
detail endpoint. If no exact name is supplied it searches `DPA4C`, then
`DPA`, and records the selected resource in `model_manifest.json`.

After a successful target-host run, replace the preview input with the
generated DPA4C trajectory and rerun the renderer; the output paths remain
independent of 04.
