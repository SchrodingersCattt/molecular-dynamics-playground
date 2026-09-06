# Bohrium inference attempt

- Intended calculation: evaluate the 64-water periodic box with the supplied DeePMD model in `registry.dp.tech/dptech/dpmd:2.2.8-cuda12.0`.
- Project/machine: project `17142`, one `c2_m4_cpu` node, 20-minute limit.
- Configuration dry run: passed.
- Submission result: no Bohrium job was created. The Windows client failed while synchronizing the local input bundle, reporting `Access is denied` for its temporary input directory. A second attempt with a workspace-local temporary directory failed at the same pre-submission step.
- Scientific consequence: the figure shows only the exact structural neighbor selection and symbolic relations `E = sum_i epsilon_i` and `F = -grad_R E`. It contains no claimed DeePMD energy and no quantitative force arrows.
- Reproducibility: `job.json`, `run.sh`, and `evaluate_water_box_deepmd.py` preserve the intended calculation without retaining duplicate model or input payloads.
