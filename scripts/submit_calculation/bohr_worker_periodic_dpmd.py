"""Thin direct-Bohr worker wrapper for the project periodic DPMD runner."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()
    from run_water_box_dpmd import run

    run(
        args.model,
        args.input,
        args.output,
        args.metadata,
        args.steps,
        args.dt,
        300.0,
        260906,
    )


if __name__ == "__main__":
    main()