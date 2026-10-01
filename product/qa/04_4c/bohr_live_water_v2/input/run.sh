#!/usr/bin/env bash
# Live DPA4C-Neo-OMat24 Velocity-Verlet run for the 0314 64-water box.
set -uo pipefail
nvidia-smi --query-gpu=name --format=csv || true
echo "--- locating a python with deepmd ---"
which dp python python3 2>/dev/null || true
ls /opt 2>/dev/null || true
conda env list 2>/dev/null || true
CANDIDATES=()
if command -v dp >/dev/null 2>&1; then
    DP_SHEBANG="$(head -1 "$(command -v dp)" | sed 's/^#!//')"
    CANDIDATES+=("$DP_SHEBANG")
fi
CANDIDATES+=(python python3 /opt/conda/bin/python /opt/mamba/bin/python /opt/miniconda3/bin/python /opt/deepmd-kit/bin/python /root/miniconda3/bin/python)
for candidate in /opt/conda/envs/*/bin/python /opt/mamba/envs/*/bin/python /root/miniconda3/envs/*/bin/python; do
    [[ -x "$candidate" ]] && CANDIDATES+=("$candidate")
done
PY=""
for candidate in "${CANDIDATES[@]}"; do
    [[ -z "$candidate" ]] && continue
    if "$candidate" -c "import deepmd" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [[ -z "$PY" ]]; then
    echo "no python with deepmd found; searching filesystem" >&2
    find / -maxdepth 7 -type d -name deepmd -path "*site-packages*" 2>/dev/null | head -5
    for found in $(find / -maxdepth 7 -type d -name deepmd -path "*site-packages*" 2>/dev/null | head -5); do
        prefix="${found%%/lib/*}"
        for candidate in "$prefix/bin/python" "$prefix/bin/python3"; do
            if [[ -x "$candidate" ]] && "$candidate" -c "import deepmd" >/dev/null 2>&1; then
                PY="$candidate"
                break 2
            fi
        done
    done
fi
if [[ -z "$PY" ]]; then
    echo "FATAL: deepmd not available in this image" >&2
    exit 3
fi
echo "using $PY"
"$PY" -c "import deepmd, torch; print('deepmd', deepmd.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" -c "import numpy; print('numpy', numpy.__version__)"
# The image's deepmd (3.2.0b1.dev) predates the dpa4c descriptor.  Upgrade to
# the released deepmd-kit that ships DPA4C, keeping the image's torch build.
DPA4C_CHECK="from deepmd.dpmodel.descriptor.base_descriptor import BaseDescriptor; BaseDescriptor.get_class_by_type('dpa4c'); print('dpa4c descriptor available')"
if ! "$PY" -c "$DPA4C_CHECK" >/dev/null 2>&1; then
    echo "--- dpa4c descriptor missing; installing deepmd-kit 3.2.0 from bundled wheels ---"
    # Compute nodes have no PyPI access; wheels are shipped in ./wheels.
    # The cibuildwheel deepmd wheel loads libmpi from the `mpich` pip package.
    # The PyPI wheel's compiled ops were built against torch 2.11.0 while the
    # image ships torch 2.12.1.  The dpa4c descriptor itself is pure Python,
    # so keep the image's compiled deepmd/lib (built for this torch) and only
    # take the Python sources from the 3.2.0 wheel.
    DEEPMD_DIR="$("$PY" -c 'import deepmd, os; print(os.path.dirname(deepmd.__file__))')"
    cp -a "$DEEPMD_DIR/lib" /tmp/deepmd_lib_image
    "$PY" -m pip install --no-index --find-links wheels --upgrade "deepmd-kit==3.2.0" mpich 2>&1 | tail -8 || true
    rm -rf "$DEEPMD_DIR/lib" && cp -a /tmp/deepmd_lib_image "$DEEPMD_DIR/lib"
    ls "$DEEPMD_DIR/lib"
    "$PY" -c "import deepmd, torch; print('deepmd', deepmd.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    "$PY" -c "$DPA4C_CHECK" || true
fi
# The V100 node reported "CUDA-capable device busy or unavailable" once; the
# 192-atom box is cheap enough to evaluate on CPU, so fall back when needed.
if ! "$PY" -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; then
    echo "CUDA unusable; running on CPU"
    export CUDA_VISIBLE_DEVICES=""
fi
set -e
"$PY" run_water_box_nnmd.py \
  --model DPA4C-Neo-OMat24-v20260819.pt \
  --input water_box_64.npz \
  --output dpa4c_water_box_trajectory.npz \
  --metadata dpa4c_water_box_trajectory.json \
  --label dpa4c --steps 5 --dt 0.5 --temperature 300 --seed 260906
