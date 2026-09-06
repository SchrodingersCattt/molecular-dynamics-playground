#!/usr/bin/env bash
set -euo pipefail

# Reproducible direct Bohr CLI route for the 192-atom periodic DPMD run.
# This script deliberately does not call dpdispatcher.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="$REPO_ROOT/scripts/submit_calculation/H2O-Phase-Diagram-model_compressed.pb"
INPUT_NPZ="$REPO_ROOT/product/data/water_box_64.npz"
RUN_ROOT="${BOHR_DPMD_RUN_ROOT:-$REPO_ROOT/product/qa/04_dpmd_bohr_cli}"
INPUT_DIR="$RUN_ROOT/input"
OUTPUT_DIR="$RUN_ROOT/output"
SUBMIT_JSON="$RUN_ROOT/submit.json"
JOURNAL="$RUN_ROOT/journal.json"
SANDBOX_ID="${BOHR_SANDBOX_ID:-}"

# These values are the direct Bohr CLI inputs. Override only when a different
# project/resource is intentionally selected; no credentials are stored here.
PROJECT_ID="${BOHR_PROJECT_ID:-17142}"
IMAGE="${BOHR_IMAGE:-registry.dp.tech/dptech/dpmd:2.2.8-cuda12.0}"
MACHINE="${BOHR_MACHINE_TYPE:-c2_m4_cpu}"
MAX_RUN_TIME="${BOHR_MAX_RUN_TIME:-20}"
STEPS="${BOHR_DPMD_STEPS:-5}"
DT_FS="${BOHR_DPMD_DT_FS:-0.5}"

command -v bohr >/dev/null 2>&1 || {
    echo "bohr CLI is required" >&2
    exit 1
}

for required in "$MODEL" "$INPUT_NPZ" "$REPO_ROOT/scripts/run_md/run_water_box_dpmd.py"; do
    [[ -f "$required" ]] || {
        echo "missing input: $required" >&2
        exit 1
    }
done

rm -rf "$INPUT_DIR" "$OUTPUT_DIR"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
cp "$MODEL" "$INPUT_DIR/H2O-Phase-Diagram-model_compressed.pb"
cp "$INPUT_NPZ" "$INPUT_DIR/water_box_64.npz"
cp "$REPO_ROOT/scripts/run_md/run_water_box_dpmd.py" "$INPUT_DIR/run_water_box_dpmd.py"
cp "$REPO_ROOT/scripts/submit_calculation/bohr_worker_periodic_dpmd.py" "$INPUT_DIR/bohr_worker_periodic_dpmd.py"

COMMAND="python bohr_worker_periodic_dpmd.py --model H2O-Phase-Diagram-model_compressed.pb --input water_box_64.npz --output periodic_results.npz --metadata periodic_metadata.json --steps ${STEPS} --dt ${DT_FS}"

echo "[bohr] project=$PROJECT_ID machine=$MACHINE image=$IMAGE steps=$STEPS dt_fs=$DT_FS"
echo "[bohr] command=$COMMAND"

# --wait makes the result state deterministic for the next download command.
bohr job submit \
    --project_id "$PROJECT_ID" \
    --job_name "md-water-box-dpmd-direct-cli" \
    --image_address "$IMAGE" \
    --machine_type "$MACHINE" \
    --max_run_time "$MAX_RUN_TIME" \
    --max_reschedule_times 1 \
    --nnode 1 \
    --log_file worker.log \
    --command "$COMMAND" \
    --input_directory "$(cygpath -w "$INPUT_DIR" 2>/dev/null || printf '%s' "$INPUT_DIR")" \
    --wait \
    --no-interactive \
    --output json | tee "$SUBMIT_JSON"

BOHR_ID="$(python - "$SUBMIT_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
data = payload.get("data", payload)
if isinstance(data, dict):
    value = data.get("bohrId") or data.get("bohr_id") or data.get("id")
else:
    value = None
if value is None:
    raise SystemExit("Could not find bohrId in submit response")
print(value)
PY
)"

echo "[bohr] bohrId=$BOHR_ID"
bohr job download "$BOHR_ID" --out "$OUTPUT_DIR" --no-interactive --output json

if [[ -f "$OUTPUT_DIR/periodic_results.npz" ]]; then
    cp "$OUTPUT_DIR/periodic_results.npz" "$REPO_ROOT/product/data/dpmd_water_box_trajectory.npz"
fi
if [[ -f "$OUTPUT_DIR/periodic_metadata.json" ]]; then
    cp "$OUTPUT_DIR/periodic_metadata.json" "$REPO_ROOT/product/data/dpmd_water_box_trajectory.json"
fi

python - "$SUBMIT_JSON" "$JOURNAL" "$REPO_ROOT" "$BOHR_ID" "$COMMAND" "$PROJECT_ID" "$IMAGE" "$MACHINE" "$STEPS" "$DT_FS" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

submit_path, journal_path, root, bohr_id, command, project_id, image, machine, steps, dt = sys.argv[1:]
root_path = pathlib.Path(root)
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

outputs = {}
for name in ("dpmd_water_box_trajectory.npz", "dpmd_water_box_trajectory.json"):
    path = root_path / "product" / "data" / name
    if path.exists():
        outputs[name] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
payload = {
    "schema": "bohr_direct_cli_journal/v1",
    "bohr_id": int(bohr_id),
    "submit_response": json.load(open(submit_path, encoding="utf-8")),
    "command": command,
    "cli": "bohr job submit + bohr job download; no dpdispatcher",
    "project_id": int(project_id),
    "image": image,
    "machine_type": machine,
    "steps": int(steps),
    "dt_fs": float(dt),
    "inputs": {
        name: {"path": str(root_path / "scripts" / "submit_calculation" / name) if name.endswith(".pb") else str(root_path / "product" / "data" / name), "sha256": sha(root_path / "scripts" / "submit_calculation" / name) if name.endswith(".pb") else sha(root_path / "product" / "data" / name)}
        for name in ("H2O-Phase-Diagram-model_compressed.pb", "water_box_64.npz")
    },
    "outputs": outputs,
}
pathlib.Path(journal_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "[bohr] results downloaded to $OUTPUT_DIR"