"""
compute_deepmd_md.py — Bohrium orchestrator for DeePMD-kit MD
==============================================================
Submits worker_deepmd.py + the H₂O DeePMD model to Bohrium via dpdispatcher,
waits for completion, downloads results.npz, and repacks it as
deepmd_data.npz (ready for render_deepmd_md.py to render).

Model: H2O-Phase-Diagram-model_compressed.pb  (54 MB, pre-downloaded locally)
  Source: https://store.aissquare.com/models/4560428e-db9c-11ee-9b22-506b4b2349d8/H2O-Phase-Diagram-model_compressed.pb
  Must be present at scripts/submit_calculation/H2O-Phase-Diagram-model_compressed.pb before running.
  (The Bohrium cluster may not have internet access; the file is uploaded via
   dpdispatcher forward_files.)

Usage:
    python scripts/submit_calculation/compute_deepmd_md.py           # submit to Bohrium
    python scripts/submit_calculation/compute_deepmd_md.py --force   # re-submit even if npz exists

Auth:
    Copy scripts/submit_calculation/.env.template → scripts/submit_calculation/.env and fill in BOHR_TICKET
    (or BOHR_EMAIL + BOHR_PASSWORD) and BOHR_PROJECT_ID.
"""

import argparse
import os
import sys
import shutil
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_FILENAME = "H2O-Phase-Diagram-model_compressed.pb"
BOX_FILENAME = "water_box_64.npz"

# ── Output paths ───────────────────────────────────────────────────────────────
NPZ_OUT      = os.path.join(REPO_ROOT, "product", "data", "deepmd_data.npz")
PERIODIC_NPZ_OUT = os.path.join(REPO_ROOT, "product", "data", "dpmd_water_box_trajectory.npz")
WORK_DIR     = os.path.join(HERE, "bohrium_work")
MODEL_PATH   = os.path.join(HERE, MODEL_FILENAME)
WORKER_PATH  = os.path.join(HERE, "worker_deepmd.py")
BOX_PATH    = os.path.join(REPO_ROOT, "product", "data", BOX_FILENAME)
PERIODIC_RUNNER = os.path.join(REPO_ROOT, "scripts", "run_md", "run_water_box_dpmd.py")

# ── Bohrium job parameters ─────────────────────────────────────────────────────
DOCKER_IMAGE = "registry.dp.tech/dptech/dpmd:2.2.8-cuda12.0"
MACHINE_TYPE = "c2_m8_cpu"
N_STEPS      = 100    # enough steps for visible dynamics in the animation
DT_FS        = 0.5
PERIODIC_STEPS = 5
DEFAULT_BOHRIUM_PROJECT_ID = 17142


# ══════════════════════════════════════════════════════════════════════════════
# NPZ save helper
# ══════════════════════════════════════════════════════════════════════════════

def _save_npz(steps, time_fs, positions, velocities, forces,
              E_total, per_atom_e, KE, TE,
              r_ij, s_ij, n_nbr, label="DeePMD-kit"):
    np.savez(
        NPZ_OUT,
        steps      = steps,
        time_fs    = time_fs,
        positions  = positions,
        velocities = velocities,
        forces     = forces,
        E_total    = E_total,
        per_atom_e = per_atom_e,
        KE         = KE,
        TE         = TE,
        r_ij       = r_ij,
        s_ij       = s_ij,
        n_nbr      = n_nbr,
        label      = np.array([label]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Bohrium submission via dpdispatcher
# ══════════════════════════════════════════════════════════════════════════════

def run_bohrium():
    """Submit worker_deepmd.py to Bohrium and download results."""
    # ── Load .env ──────────────────────────────────────────────────────────────
    env_path = os.path.join(HERE, ".env")
    if not os.path.exists(env_path) and not os.environ.get("BOHR_ACCESS_KEY"):
        sys.exit(
            f"[ERROR] {env_path} not found.\n"
            f"Copy scripts/submit_calculation/.env.template → scripts/submit_calculation/.env and fill in credentials."
        )
    if os.environ.get("BOHR_ACCESS_KEY") and not os.environ.get("BOHR_TICKET"):
        os.environ["BOHR_TICKET"] = os.environ["BOHR_ACCESS_KEY"]
    try:
        from dotenv import load_dotenv
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        # Manual parse if python-dotenv not installed
        if os.path.exists(env_path):
            with open(env_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

    ticket     = os.environ.get("BOHR_TICKET", "").strip()
    email      = os.environ.get("BOHR_EMAIL", "").strip()
    password   = os.environ.get("BOHR_PASSWORD", "").strip()
    project_id = int(os.environ.get("BOHR_PROJECT_ID", str(DEFAULT_BOHRIUM_PROJECT_ID)))

    if not ticket and not (email and password):
        sys.exit(
            "[ERROR] No Bohrium credentials found.\n"
            "Set BOHR_TICKET (or BOHR_EMAIL + BOHR_PASSWORD) in scripts/submit_calculation/.env"
        )

    # ── Check model file ───────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"[ERROR] Model file not found: {MODEL_PATH}\n"
            f"Download it from:\n"
            f"  https://store.aissquare.com/models/4560428e-db9c-11ee-9b22-506b4b2349d8/H2O-Phase-Diagram-model_compressed.pb\n"
            f"and place it at: {MODEL_PATH}"
        )
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"[Model] {MODEL_FILENAME}  ({size_mb:.1f} MB)", flush=True)

    # ── Check worker file ──────────────────────────────────────────────────────
    if not os.path.exists(WORKER_PATH):
        sys.exit(
            f"[ERROR] Worker file not found: {WORKER_PATH}\n"
            "Expected: scripts/submit_calculation/worker_deepmd.py"
        )
    if not os.path.exists(BOX_PATH):
        sys.exit(f"[ERROR] Periodic input not found: {BOX_PATH}")
    if not os.path.exists(PERIODIC_RUNNER):
        sys.exit(f"[ERROR] Periodic runner not found: {PERIODIC_RUNNER}")

    # ── Import dpdispatcher ────────────────────────────────────────────────────
    try:
        from dpdispatcher import Machine, Resources, Task, Submission
        from dpdispatcher.contexts.dp_cloud_server_context import BohriumContext
    except ImportError:
        sys.exit(
            "[ERROR] dpdispatcher not installed.\n"
            "Run:  pip install dpdispatcher"
        )

    # ── Prepare local work directory ───────────────────────────────────────────
    # Remove stale dpdispatcher state files so the job is re-submitted fresh
    # (old .sub / .sub.run / submission.json cause dpdispatcher to skip upload)
    import glob
    for stale in glob.glob(os.path.join(WORK_DIR, "*.sub")) + \
                 glob.glob(os.path.join(WORK_DIR, "*.sub.run")) + \
                 glob.glob(os.path.join(WORK_DIR, "*_flag_if_job_task_fail")) + \
                 glob.glob(os.path.join(WORK_DIR, "*_task_tag_finished")) + \
                 glob.glob(os.path.join(WORK_DIR, "*_job_tag_finished")):
        try:
            os.remove(stale)
            print(f"[setup] Removed stale file: {os.path.basename(stale)}", flush=True)
        except OSError:
            pass
    submission_home = os.path.join(os.path.expanduser("~"), ".dpdispatcher", "submission")
    if os.path.isdir(submission_home):
        for stale in glob.glob(os.path.join(submission_home, "*.json")):
            try:
                with open(stale, encoding="utf-8") as handle:
                    content = handle.read()
                if WORK_DIR.replace("\\", "/") in content.replace("\\", "/"):
                    os.remove(stale)
                    print(f"[setup] Removed stale submission record: {os.path.basename(stale)}", flush=True)
            except OSError:
                pass
    # Also clear the dpdispatcher home-dir submission JSON for this work_base
    import glob as _glob
    home_dp = os.path.join(os.path.expanduser("~"), ".dpdispatcher", "dp_cloud_server")
    if os.path.isdir(home_dp):
        for jf in _glob.glob(os.path.join(home_dp, "*.json")):
            try:
                with open(jf) as fh:
                    content = fh.read()
                if WORK_DIR.replace("\\", "/") in content.replace("\\", "/"):
                    os.remove(jf)
                    print(f"[setup] Removed stale submission JSON: {os.path.basename(jf)}", flush=True)
            except OSError:
                pass

    os.makedirs(WORK_DIR, exist_ok=True)
    shutil.copy(WORKER_PATH, os.path.join(WORK_DIR, "worker_deepmd.py"))
    shutil.copy(MODEL_PATH,  os.path.join(WORK_DIR, MODEL_FILENAME))
    shutil.copy(BOX_PATH, os.path.join(WORK_DIR, BOX_FILENAME))
    shutil.copy(PERIODIC_RUNNER, os.path.join(WORK_DIR, "run_water_box_dpmd.py"))

    # ── Build remote_profile ───────────────────────────────────────────────────
    # Auth option A: BOHR_TICKET — set env var; BohriumContext reads it via
    #   os.environ.get("BOHR_TICKET") in its __init__.
    # Auth option B: email + password — passed into remote_profile dict;
    #   BohriumContext reads remote_profile["email"] / ["password"].
    remote_profile = {
        "program_id": project_id,
        "input_data": {
            "job_type":     "container",
            "image_name":   DOCKER_IMAGE,
            "machine_type": MACHINE_TYPE,
            # Print remote stdout/stderr locally so failures are visible
            "output_log":   True,
            # Pre-populate backward_files with forward-slash paths.
            # dpdispatcher's _gen_backward_files_list uses os.path.join which
            # produces backslashes on Windows (e.g. ".\\results.npz"), causing
            # Bohrium's Linux container to fail to find the files and skip
            # packaging out.zip.  By setting this here, the if-not check in
            # do_submit() skips _gen_backward_files_list entirely.
            "backward_files": [
                "periodic_results.npz",
                "periodic_metadata.json",
                "worker.log",
                "worker.err",
                "worker_crash.log",
            ], 
        },
    }
    if ticket:
        os.environ["BOHR_TICKET"] = ticket
        print("[Bohrium] Auth: BOHR_TICKET (Access Token)", flush=True)
    else:
        remote_profile["email"]    = email
        remote_profile["password"] = password
        print(f"[Bohrium] Auth: email ({email})", flush=True)

    # ── dpdispatcher 1.0.x API: create context first, then machine ─────────────
    context = BohriumContext(
        local_root    = WORK_DIR,
        remote_root   = ".",
        remote_profile= remote_profile,
    )

    machine = Machine.subclasses_dict["Bohrium"](context)

    resources = Resources(
        number_node   = 1,
        cpu_per_node  = 2,
        gpu_per_node  = 0,
        queue_name    = "normal",
        group_size    = 1,
    )

    task = Task(
        command       = (
            f"python run_water_box_dpmd.py "
            f"--model {MODEL_FILENAME} "
            f"--input {BOX_FILENAME} "
            f"--output periodic_results.npz "
            f"--metadata periodic_metadata.json "
            f"--steps {PERIODIC_STEPS} "
            f"--dt {DT_FS} "
        ),
        task_work_path= ".",
        forward_files = ["run_water_box_dpmd.py", MODEL_FILENAME, BOX_FILENAME],
        # Download results + diagnostic logs regardless of exit code
        backward_files= ["periodic_results.npz", "periodic_metadata.json", "worker.log", "worker.err", "worker_crash.log"],
        outlog        = "worker.log",
        errlog        = "worker.err",
    )

    submission = Submission(
        work_base  = WORK_DIR,
        machine    = machine,
        resources  = resources,
        task_list  = [task],
    )

    print("[Bohrium] Submitting job...", flush=True)
    submission.run_submission()
    print("[Bohrium] Job completed.", flush=True)

    # ── Print diagnostic logs if present ──────────────────────────────────────
    for logname in ("worker.log", "worker.err", "worker_crash.log"):
        logpath = os.path.join(WORK_DIR, logname)
        if os.path.exists(logpath):
            size = os.path.getsize(logpath)
            print(f"\n{'='*60}", flush=True)
            print(f"[{logname}]  ({size} bytes)", flush=True)
            print('='*60, flush=True)
            try:
                with open(logpath) as fh:
                    print(fh.read(), flush=True)
            except Exception as e:
                print(f"  (could not read: {e})", flush=True)

    # ── Repack results.npz → deepmd_data.npz ──────────────────────────────────
    results_path = os.path.join(WORK_DIR, "periodic_results.npz")
    if not os.path.exists(results_path):
        # Print crash log if available before exiting
        crash_log = os.path.join(WORK_DIR, "worker_crash.log")
        if os.path.exists(crash_log):
            print("\n[ERROR] worker_crash.log found — worker failed:", flush=True)
            with open(crash_log) as fh:
                print(fh.read(), flush=True)
        sys.exit(f"[ERROR] results.npz not found in {WORK_DIR} after job completion.")

    metadata_result = os.path.join(WORK_DIR, "periodic_metadata.json")
    os.makedirs(os.path.dirname(PERIODIC_NPZ_OUT), exist_ok=True)
    shutil.copy(results_path, PERIODIC_NPZ_OUT)
    if os.path.exists(metadata_result):
        shutil.copy(
            metadata_result,
            os.path.join(os.path.dirname(PERIODIC_NPZ_OUT), "dpmd_water_box_trajectory.json"),
        )
    print(f"[Bohrium] Repacked → {PERIODIC_NPZ_OUT}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Submit DeePMD-kit MD to Bohrium and save deepmd_data.npz"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-submit even if deepmd_data.npz already exists"
    )
    args = parser.parse_args()

    if os.path.exists(NPZ_OUT) and not args.force:
        print(f"[INFO] {NPZ_OUT} already exists.  Use --force to re-run.")
        return

    run_bohrium()


if __name__ == "__main__":
    main()
