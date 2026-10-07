# Bohrium TNT UKS job

This bundle runs the real TNT C2–N2 PBE0/def2-SVP geometry optimization and
100-step UKS trajectory. It is intentionally separate from the local demo
path: `backend=analytic_demo` is never used for the scientific result.

Submit from the 0314 project root with the Bohrium CLI, for example:

```bash
bohr job submit \
  -m registry.dp.tech/dptech/ubuntu:20.04-py3.10 \
  -t c16_m32_cpu \
  -c "bash run_tnt_uks.sh" \
  -p scripts/submit_calculation/tnt_uks_job \
  --project_id "$BOHRIUM_PROJECT_ID" \
  -n 0314-tnt-uks-c2-no2 \
  --max_run_time 360 \
  --max_reschedule_times 2
```

Download `results/` and the job log into the project QA directory after the
job reaches `Finished`. Preserve the job ID, image, resolved package versions,
stdout/stderr, and SHA256 hashes in the provenance manifest.
