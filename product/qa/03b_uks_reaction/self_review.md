# 03b Self-Review

## Current checked-in render

- Figure: `product/figures/03b_uks_reaction.png`
- Video: `product/videos/03b_uks_reaction.mp4`
- Data backend: `pyscf_uks` (Bohrium job 20835773, PBE0/def2-SVP)
- Full video QA: 720/720 frames passed.
- Initial geometry check: neutral C7H5N3O6 TNT, 21 atoms, selected C2–N2 = 1.475 Å, minimum O–O distance = 2.152 Å; mat-vis preflight passed.
- MatterVis arrow scales: force = 150, half-step velocity = 34, position drift = 4.0; native vector metadata was inspected after rendering.

## Findings and actions

0. The earlier centre asset was a 2-D aromatic-plane slice of the 3-D field;
   nuclei outside that plane therefore had no contour at their coordinates.
   The density unit conversion in the NPZ path was also corrected and checked
   by per-frame electron-count integration. The centre asset now uses line-only
   alpha/beta contours from a fixed-grid 3-D maximum projection along the
   camera normal, with the same world-space coordinates as MatterVis nuclei;
   the existing panel layout and timeline are unchanged.

1. The first composition used Unicode subscripts and Greek glyphs that were missing from Arial. They were replaced with mathtext or ASCII-safe labels.
2. The narrow left rail clipped the velocity node label. A compact custom Verlet loop was added for 03b.
3. The first spin contour filled the whole plane near zero density. Positive and negative fields are now thresholded separately, leaving localized red/blue lobes.
4. The initial frame has almost no physical spin density. Persistent alpha/beta legend marks keep the semantic colours readable without fabricating a spin field.
5. The first 03b video used an independent reaction-animation timeline. It was replaced by a direct adaptation of `render_aimd_scf.py`: two detailed ion steps, nested SCF iterations, convergence pause, force, velocity, position, then rapid ion-step cycling with native MatterVis assets.
6. Real UKS data were generated in Bohrium because the local Windows environment has no compatible PySCF installation. The job image, machine type, and output hashes are recorded in the data manifest.
7. The rapid section uses 19 real ionic snapshots, so all 16 rapid blocks advance monotonically through the dissociation trajectory. There is no modulo replay and no terminal freeze.

8. A no-user-timeout Bohrium resubmission is running as job 20864421 with the
   prebuilt PySCF/geometric image; the request and status are recorded in
   `bohrium_resubmission_20864421.json`.

## Stop condition

The TNT visual/data fixture is ready for review; the real UKS backend and full-frame QA both pass for the recovered 101-frame dataset.

