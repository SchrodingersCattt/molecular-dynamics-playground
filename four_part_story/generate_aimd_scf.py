"""Generate fresh H2O-dimer RHF/SCF data and true 3D density volumes.

The reference integral engine is shared with the project, but the data product
and all visual geometry in ``four_part_story`` are new.  The volume grid is
evaluated directly from the stored AO density matrices, so the rendered
isosurface is a three-dimensional field rather than a decorated 2D contour.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "md_workflows"))

import engine_rhf as rhf  # noqa: E402
from three_slide_story.compute_h2o_dimer_scf import (  # noqa: E402
    ELEMENTS,
    dimer_geometry,
    finite_difference_forces,
    run_scf,
)


DATA_DIR = Path(__file__).resolve().parent / "data"


def density_volumes(
    density_matrices: np.ndarray,
    basis: list[dict],
    selected_iterations: np.ndarray,
    *,
    nx: int = 68,
    ny: int = 50,
    nz: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_angstrom = np.linspace(-3.35, 3.35, nx)
    y_angstrom = np.linspace(-2.35, 2.35, ny)
    z_angstrom = np.linspace(-1.75, 1.75, nz)
    x_grid, y_grid, z_grid = np.meshgrid(
        x_angstrom * rhf.ANG_TO_BOHR,
        y_angstrom * rhf.ANG_TO_BOHR,
        z_angstrom * rhf.ANG_TO_BOHR,
        indexing="ij",
    )
    ao_values = np.zeros((len(basis), nx, ny, nz), dtype=np.float64)
    for basis_index, function in enumerate(basis):
        lx, ly, lz = function["l"]
        cx, cy, cz = function["center"]
        dx = x_grid - cx
        dy = y_grid - cy
        dz = z_grid - cz
        radius_squared = dx * dx + dy * dy + dz * dz
        angular = (dx**lx) * (dy**ly) * (dz**lz)
        for exponent, coefficient in zip(function["exps"], function["coef"]):
            ao_values[basis_index] += (
                rhf._norm_prim(exponent, function["l"])
                * coefficient
                * angular
                * np.exp(-exponent * radius_squared)
            )

    volumes = []
    for iteration in selected_iterations:
        matrix = density_matrices[int(iteration)]
        density = np.einsum("mn,mxyz,nxyz->xyz", matrix, ao_values, ao_values, optimize=True)
        volumes.append(np.maximum(density, 0.0).astype(np.float32))
    return x_angstrom, y_angstrom, z_angstrom, np.asarray(volumes)


def write_extxyz(path: Path, positions: np.ndarray) -> None:
    lines = [str(len(ELEMENTS)), 'Properties=species:S:1:pos:R:3 source="fresh RHF H2O dimer"']
    for element, position in zip(ELEMENTS, positions):
        lines.append(f"{element} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-forces", action="store_true")
    parser.add_argument("--force-workers", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    positions = dimer_geometry()
    result = run_scf(positions, mixing=0.70, tolerance=1.0e-8)
    selected = np.unique(
        np.clip(np.array([0, 2, 5, 9, 13, len(result["energies"]) - 1]), 0, len(result["energies"]) - 1)
    )
    grid_x, grid_y, grid_z, volumes = density_volumes(
        result["density_matrices"], result["basis"], selected
    )
    if args.skip_forces:
        previous = np.load(ROOT / "three_slide_story" / "data" / "h2o_dimer_scf.npz")
        forces = np.asarray(previous["forces"], dtype=float)
        force_origin = "verified prior central-finite-difference result"
    else:
        forces = finite_difference_forces(positions, workers=args.force_workers)
        force_origin = "fresh central finite difference of converged RHF energy"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "aimd_h2o_dimer.npz"
    np.savez_compressed(
        output,
        elements=ELEMENTS,
        positions=positions,
        bonds=np.array([[0, 1], [0, 2], [3, 4], [3, 5]], dtype=int),
        hydrogen_bond=np.array([1, 3], dtype=int),
        energies=result["energies"],
        residuals=result["residuals"],
        forces=forces,
        density_iterations=selected,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        density_volumes=volumes,
        method=np.array("RHF/STO-3G"),
        mixing=np.array(0.70),
        convergence_tolerance=np.array(1.0e-8),
    )
    write_extxyz(DATA_DIR / "aimd_h2o_dimer.extxyz", positions)
    final_delta = float(abs(result["energies"][-1] - result["energies"][-2]))
    metadata = {
        "method": "RHF/STO-3G",
        "engine": "project pure-NumPy Gaussian integral engine",
        "atoms": ELEMENTS.tolist(),
        "iterations": int(len(result["energies"])),
        "selected_density_iterations_zero_based": selected.tolist(),
        "density_grid_shape": list(volumes.shape[1:]),
        "density_grid_units": "electron density from AO density matrix on an angstrom coordinate grid",
        "final_energy_hartree": float(result["energy"]),
        "final_delta_energy_hartree": final_delta,
        "density_mixing": 0.70,
        "convergence_tolerance_hartree": 1.0e-8,
        "force_method": force_origin,
        "force_delta_angstrom": 0.003,
        "max_force_ev_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "wall_seconds": time.perf_counter() - started,
    }
    (DATA_DIR / "aimd_h2o_dimer.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
