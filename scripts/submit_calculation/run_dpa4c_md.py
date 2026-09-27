"""Run DPA4C MD after installing DeePMD-kit >=3.2.

This wrapper keeps the exact input/output contract needed by 04_4c.  It does
not silently fall back to the vanilla ``.pb`` model: missing DPA4C support is
reported as an actionable error.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--system", type=Path, default=Path("product/data/water_box_64.npz"))
    ap.add_argument("--output", type=Path, default=Path("product/data/04_4c_dpa4c_trajectory.npz"))
    args = ap.parse_args()
    if not args.model.exists(): raise SystemExit(f"Missing DPA4C checkpoint: {args.model}")
    try:
        import deepmd  # noqa: F401
    except ImportError as exc:
        raise SystemExit("DPA4C requires DeePMD-kit >=3.2 with the PyTorch Exportable backend; install it before running this command") from exc
    raise SystemExit("DPA4C trajectory runner is intentionally gated: connect this wrapper to dp --pt-expt/ASE on the target CUDA host; no vanilla fallback is permitted")

if __name__ == "__main__": main()
