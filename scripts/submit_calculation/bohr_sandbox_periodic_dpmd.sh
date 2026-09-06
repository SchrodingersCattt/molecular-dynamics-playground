#!/usr/bin/env bash
set -euo pipefail

# Direct Bohr sandbox route for the periodic 64-water DPMD trajectory.
# This is the canonical reproducible route; it deliberately does not use
# dpdispatcher and avoids the Windows bohr job --input_directory sync path.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="$REPO_ROOT/scripts/submit_calculation/H2O-Phase-Diagram-model_compressed.pb"
INPUT_NPZ="$REPO_ROOT/product/data/water_box_64.npz"
RUN_ROOT="${BOHR_DPMD_SANDBOX_ROOT:-$REPO_ROOT/product/qa/04_dpmd_bohr_sandbox}"
INPUT_DIR="$RUN_ROOT/input"
OUTPUT_DIR="$RUN_ROOT/output"
JOURNAL="$RUN_ROOT/journal.json"
PROJECT_ID="${BOHR_PROJECT_ID:-17142}"
TEMPLATE="${BOHR_SANDBOX_TEMPLATE:-ch4-deepmd}"
STEPS="${BOHR_DPMD_STEPS:-5}"
DT_FS="${BOHR_DPMD_DT_FS:-0.5}"
REMOTE="C:/Program Files/Git/root/dpmd"
REMOTE_LINUX="/home/user/C:/Program Files/Git/root/dpmd"

command -v bohr >/dev/null 2>&1 || { echo "bohr CLI is required" >&2; exit 1; }
for required in "$MODEL" "$INPUT_NPZ" "$REPO_ROOT/scripts/run_md/run_water_box_dpmd.py" "$REPO_ROOT/scripts/submit_calculation/bohr_worker_periodic_dpmd.py"; do
    [[ -f "$required" ]] || { echo "missing input: $required" >&2; exit 1; }
done

rm -rf "$INPUT_DIR" "$OUTPUT_DIR"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
cp "$MODEL" "$INPUT_DIR/H2O-Phase-Diagram-model_compressed.pb"
cp "$INPUT_NPZ" "$INPUT_DIR/water_box_64.npz"
cp "$REPO_ROOT/scripts/run_md/run_water_box_dpmd.py" "$INPUT_DIR/run_water_box_dpmd.py"
cp "$REPO_ROOT/scripts/submit_calculation/bohr_worker_periodic_dpmd.py" "$INPUT_DIR/bohr_worker_periodic_dpmd.py"

CREATE_JSON="$RUN_ROOT/create.json"
bohr sandbox create --template "$TEMPLATE" --project-id "$PROJECT_ID" --timeout 1800 --reserve-failed-sandbox --output json > "$CREATE_JSON"
SANDBOX_ID="$(python - "$CREATE_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["data"]["sandboxID"])
PY
)"

echo "[bohr sandbox] sandbox=$SANDBOX_ID project=$PROJECT_ID template=$TEMPLATE"
SOURCE_WIN="$(cygpath -w "$INPUT_DIR" 2>/dev/null || printf '%s' "$INPUT_DIR")"
bohr sandbox files write "$SANDBOX_ID" "$REMOTE" --source "$SOURCE_WIN" --timeout 600 --output json > "$RUN_ROOT/upload.json"

COMMAND="python '$REMOTE_LINUX/bohr_worker_periodic_dpmd.py' --model '$REMOTE_LINUX/H2O-Phase-Diagram-model_compressed.pb' --input '$REMOTE_LINUX/water_box_64.npz' --output '$REMOTE_LINUX/periodic_results.npz' --metadata '$REMOTE_LINUX/periodic_metadata.json' --steps $STEPS --dt $DT_FS"
echo "[bohr sandbox] command=$COMMAND"
bohr sandbox exec "$SANDBOX_ID" --command "$COMMAND" --timeout 1200 --output json > "$RUN_ROOT/exec.json"

# The CLI accepts the returned C:/... path when it is passed without the
# /home/user prefix. This avoids the Windows/MSYS double-prefix conversion.
bohr sandbox files read "$SANDBOX_ID" "$REMOTE/periodic_results.npz" --destination "$OUTPUT_DIR/periodic_results.npz" --timeout 600 --output json > "$RUN_ROOT/read_results.json"
bohr sandbox files read "$SANDBOX_ID" "$REMOTE/periodic_metadata.json" --destination "$OUTPUT_DIR/periodic_metadata.json" --timeout 600 --output json > "$RUN_ROOT/read_metadata.json"
cp "$OUTPUT_DIR/periodic_results.npz" "$REPO_ROOT/product/data/dpmd_water_box_trajectory.npz"
cp "$OUTPUT_DIR/periodic_metadata.json" "$REPO_ROOT/product/data/dpmd_water_box_trajectory.json"

python - "$JOURNAL" "$REPO_ROOT" "$SANDBOX_ID" "$PROJECT_ID" "$TEMPLATE" "$COMMAND" "$STEPS" "$DT_FS" <<'PY'
import hashlib, json, pathlib, sys
journal, root, sandbox, project, template, command, steps, dt = sys.argv[1:]
root = pathlib.Path(root)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
inputs = {
    "model": root / "scripts/submit_calculation/H2O-Phase-Diagram-model_compressed.pb",
    "water_box": root / "product/data/water_box_64.npz",
}
outputs = {
    "trajectory": root / "product/data/dpmd_water_box_trajectory.npz",
    "metadata": root / "product/data/dpmd_water_box_trajectory.json",
}
payload = {
    "schema": "bohr_direct_sandbox_journal/v1",
    "route": "bohr sandbox create + files write/read + sandbox exec",
    "dispatcher": False,
    "sandbox_id": sandbox,
    "project_id": int(project),
    "template": template,
    "command": command,
    "steps": int(steps),
    "dt_fs": float(dt),
    "inputs": {key: {"path": str(path), "sha256": sha(path)} for key, path in inputs.items()},
    "outputs": {key: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for key, path in outputs.items()},
    "reproduction_script": "scripts/submit_calculation/bohr_sandbox_periodic_dpmd.sh",
}
pathlib.Path(journal).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "[bohr sandbox] trajectory written to product/data and journaled at $JOURNAL"
