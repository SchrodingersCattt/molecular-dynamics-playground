#!/usr/bin/env bash
set -euo pipefail
python evaluate_water_box_deepmd.py > eval.log 2>&1
