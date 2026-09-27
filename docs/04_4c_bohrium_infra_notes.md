# 04_4c Bohrium / CLI notes

This note records the reproducibility boundary for the live DPA4C water-box
run and the CLI issues observed on 2026-09-24.

## Intended run

- Project: `27666`
- Input: this project’s `product/data/water_box_64.npz`
- Public checkpoint: `DPA4C-Neo-OMat24-v20260819.pt`
- Runner: `scripts/submit_calculation/run_dpa4c_water_ase.py`
- Target sandbox: `c16_m64_1 * NVIDIA 4090`, finite 1800 s timeout
- Output: live DPA4C energy, atomic energy, forces, and 48 water-box states

## CLI pitfalls

1. `bohr job submit` rejects nonexistent image names with `imageName not found`.
   The public-looking `registry.dp.tech/dptech/deepmd-kit:3.2.0-cuda12.8`
   tag was not present in this project’s catalog.
2. `bohr doctor` and job submission require network access to
   `https://open.bohrium.com`; local sandbox policy initially blocked that
   socket. `require_escalated` was needed for the explicitly authorized job.
3. `bohr job describe` uses the **Bohr ID** (`bohrJobId`), not the platform
   `jobId`. For this run they were different.
4. Queued jobs cannot be terminated with `job terminate`; use
   `bohr job +cancel <bohr_id>`.
5. `bohr job log` uses `--out <directory>`, not the global `-o` output-format
   flag. A job can be running with no log files until the command starts.
6. `bohr sandbox exec` is sensitive to PowerShell quoting. The reliable form
   is `bohr sandbox exec <sandbox_id> -- bash /home/user/script.sh`; upload a
   script with `sandbox files write` instead of embedding shell operators.
7. The default `pytorch20-scicomp:1.0.6` sandbox had Torch but no DeePMD. A
   pip-installed DeePMD 3.2 wheel then failed because its CXX11 ABI (1) did
   not match the preinstalled Torch ABI (0). NumPy 2 also produced a Torch
   ABI warning, and the DeePMD PT backend required MPICH package metadata.
8. A project-local `dpmd-cu126-pt:v20260701-pt` runtime is the intended
   solution, but sandbox image preparation must finish before it can be used.

## Current state

No external unpublished structure is used. The 04_4c renderer refuses to
render until a live 0314 water-box trajectory is present, so it cannot silently
fall back to vanilla DPMD or another project’s structure.
