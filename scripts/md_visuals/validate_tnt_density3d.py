"""Validate fixed-grid TNT alpha/beta density and Cube atom alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("density_npz", type=Path)
    parser.add_argument("cube_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.density_npz, allow_pickle=False) as archive:
        alpha = np.asarray(archive["rho_alpha_3d"])
        beta = np.asarray(archive["rho_beta_3d"])
        positions = np.asarray(archive["positions_ang"])
        elements = np.asarray(archive["elements"]).astype(str)
        grid = [np.asarray(archive[key], dtype=float) for key in ("grid_x_ang", "grid_y_ang", "grid_z_ang")]
    if alpha.shape != beta.shape or alpha.ndim != 4:
        raise SystemExit(f"alpha/beta shape mismatch: {alpha.shape} vs {beta.shape}")
    if alpha.shape[0] != positions.shape[0] or positions.shape[1] != len(elements):
        raise SystemExit("density frame/atom dimensions do not match positions/elements")
    spacing = np.asarray([np.diff(axis).mean() for axis in grid])
    for axis, values in zip(("x", "y", "z"), grid):
        if not np.allclose(np.diff(values), np.diff(values)[0], atol=1.0e-8):
            raise SystemExit(f"non-uniform {axis} grid spacing")
    cube_checks = []
    for frame in (0, positions.shape[0] // 2, positions.shape[0] - 1):
        for channel in ("alpha", "beta"):
            path = args.cube_dir / f"frame_{frame:04d}_{channel}.cube"
            if not path.exists():
                raise SystemExit(f"missing representative Cube: {path}")
            from mat_viewer.cube.io import read_cube

            cube = read_cube(path)
            coords = np.asarray([atom.coord for atom in cube.atoms], dtype=float)
            max_error = float(np.max(np.linalg.norm(coords - positions[frame], axis=1)))
            if max_error > 1.0e-6:
                raise SystemExit(f"Cube atom mismatch at frame {frame}: {max_error} A")
            cube_checks.append({"frame": frame, "channel": channel, "path": str(path), "max_atom_error_angstrom": max_error})
    payload = {
        "density_npz": str(args.density_npz),
        "frames": int(alpha.shape[0]),
        "shape": list(alpha.shape[1:]),
        "spacing_angstrom": spacing.tolist(),
        "elements": elements.tolist(),
        "cube_checks": cube_checks,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
