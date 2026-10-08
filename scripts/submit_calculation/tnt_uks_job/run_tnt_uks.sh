#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"

python - <<'PY'
import pyscf
import geometric
import ase
print("PySCF image import OK", pyscf.__version__)
print("geometric", getattr(geometric, "__version__", "unknown"))
print("ASE", ase.__version__)
PY

mkdir -p results
PYTHONPATH="$PWD/scripts/run_md:$PWD/scripts/build_box" \
  python scripts/build_box/generate_uks_tnt.py --steps 100 --speed 0.10 --geometry tnt_optimized.xyz --force

cp product/data/uks_tnt_reaction.npz results/
cp product/data/uks_tnt_reaction.json results/
cp product/data/uks_tnt_reaction_density3d.npz results/
cp product/qa/03b_uks_reaction/source/tnt_optimized.xyz results/
python - <<'PY'
import json
import platform
import sys
from pathlib import Path

payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "cwd": str(Path.cwd()),
}
Path("results/runtime.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
