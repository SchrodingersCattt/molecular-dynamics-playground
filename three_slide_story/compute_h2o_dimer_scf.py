"""Generate real RHF/STO-3G SCF data for a hydrogen-bonded H2O dimer.

The integral evaluator is the project's pure-NumPy reference implementation.
This script generalizes its monomer basis construction to six atoms and records
the actual SCF energy and density-matrix history used by slide 02.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "md_workflows"))
import engine_rhf as rhf  # noqa: E402


ELEMENTS = np.array(["O", "H", "H", "O", "H", "H"])
CHARGES = np.array([8.0, 1.0, 1.0, 8.0, 1.0, 1.0])


def dimer_geometry() -> np.ndarray:
    """Return a compact hydrogen-bonded water dimer geometry in angstrom."""

    bond = 0.957
    angle = np.deg2rad(104.5)
    half = np.deg2rad(52.25)
    o1 = np.array([-1.45, 0.0, 0.0])
    o2 = np.array([1.45, 0.0, 0.0])
    return np.array(
        [
            o1,
            o1 + bond * np.array([1.0, 0.0, 0.0]),
            o1 + bond * np.array([np.cos(angle), np.sin(angle), 0.0]),
            o2,
            o2 + bond * np.array([np.cos(half), np.sin(half), 0.0]),
            o2 + bond * np.array([np.cos(half), -np.sin(half), 0.0]),
        ],
        dtype=float,
    )


def build_dimer_basis(positions: np.ndarray) -> list[dict]:
    positions_bohr = positions * rhf.ANG_TO_BOHR
    basis: list[dict] = []
    oxygen_shells = [
        rhf._STO3G_O_1s,
        rhf._STO3G_O_2s,
        rhf._STO3G_O_2px,
        rhf._STO3G_O_2py,
        rhf._STO3G_O_2pz,
    ]
    for element, center in zip(ELEMENTS, positions_bohr):
        shells = oxygen_shells if element == "O" else [rhf._STO3G_H_1s]
        for exponents, coefficients, angular_momentum in shells:
            basis.append(
                {
                    "center": center.copy(),
                    "exps": exponents,
                    "coef": coefficients,
                    "l": angular_momentum,
                }
            )
    return basis


def run_scf(
    positions: np.ndarray,
    *,
    mixing: float = 0.70,
    tolerance: float = 1.0e-8,
    max_iterations: int = 80,
    keep_density_history: bool = True,
) -> dict:
    basis = build_dimer_basis(positions)
    positions_bohr = positions * rhf.ANG_TO_BOHR
    overlap, kinetic, attraction, eri = rhf._build_integrals(
        basis, positions_bohr, CHARGES
    )
    h_core = kinetic + attraction

    nuclear_energy = 0.0
    for i in range(len(CHARGES)):
        for j in range(i + 1, len(CHARGES)):
            distance = np.linalg.norm(positions_bohr[i] - positions_bohr[j])
            nuclear_energy += CHARGES[i] * CHARGES[j] / distance

    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    keep = eigenvalues > 1.0e-8
    orthogonalizer = eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))
    fock_orthogonal = orthogonalizer.T @ h_core @ orthogonalizer
    _, coeff_orthogonal = np.linalg.eigh(fock_orthogonal)
    coefficients = orthogonalizer @ coeff_orthogonal
    occupied = 10
    density = 2.0 * coefficients[:, :occupied] @ coefficients[:, :occupied].T

    energies: list[float] = []
    residuals: list[float] = []
    density_history: list[np.ndarray] = []
    previous_energy: float | None = None

    for _ in range(max_iterations):
        coulomb = np.einsum("kl,ijkl->ij", density, eri, optimize=True)
        exchange = np.einsum("kl,ikjl->ij", density, eri, optimize=True)
        fock = h_core + coulomb - 0.5 * exchange
        electronic_energy = 0.5 * np.sum(density * (h_core + fock))
        total_energy = float(electronic_energy + nuclear_energy)

        fock_orthogonal = orthogonalizer.T @ fock @ orthogonalizer
        orbital_energies, coeff_orthogonal = np.linalg.eigh(fock_orthogonal)
        coefficients = orthogonalizer @ coeff_orthogonal
        diagonalized_density = (
            2.0 * coefficients[:, :occupied] @ coefficients[:, :occupied].T
        )
        residual = float(np.linalg.norm(diagonalized_density - density) / np.sqrt(density.size))

        energies.append(total_energy)
        residuals.append(residual)
        if keep_density_history:
            density_history.append(density.copy())

        if previous_energy is not None and abs(total_energy - previous_energy) < tolerance:
            density = diagonalized_density
            break
        previous_energy = total_energy
        density = mixing * diagonalized_density + (1.0 - mixing) * density

    return {
        "energy": total_energy,
        "orbital_energies": orbital_energies,
        "energies": np.asarray(energies),
        "residuals": np.asarray(residuals),
        "density_matrices": np.asarray(density_history),
        "basis": basis,
        "coefficients": coefficients,
    }


def density_grid(
    density_matrices: np.ndarray,
    basis: list[dict],
    *,
    nx: int = 128,
    ny: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ang = np.linspace(-3.2, 3.2, nx)
    y_ang = np.linspace(-2.0, 2.0, ny)
    x_bohr, y_bohr = np.meshgrid(x_ang * rhf.ANG_TO_BOHR, y_ang * rhf.ANG_TO_BOHR)
    values = np.zeros((len(basis), ny, nx), dtype=float)

    for index, function in enumerate(basis):
        lx, ly, lz = function["l"]
        cx, cy, cz = function["center"]
        dx = x_bohr - cx
        dy = y_bohr - cy
        dz = -cz
        radius_squared = dx * dx + dy * dy + dz * dz
        angular = (dx**lx) * (dy**ly) * (dz**lz)
        for exponent, coefficient in zip(function["exps"], function["coef"]):
            values[index] += (
                rhf._norm_prim(exponent, function["l"])
                * coefficient
                * angular
                * np.exp(-exponent * radius_squared)
            )

    grids = np.asarray(
        [
            np.maximum(
                np.einsum("mn,mxy,nxy->xy", matrix, values, values, optimize=True),
                0.0,
            )
            for matrix in density_matrices
        ],
        dtype=np.float32,
    )
    return x_ang, y_ang, grids


def _shifted_energy(task: tuple[np.ndarray, int, int, float]) -> tuple[int, int, float, float]:
    positions, atom_index, axis, delta = task
    plus = positions.copy()
    minus = positions.copy()
    plus[atom_index, axis] += delta
    minus[atom_index, axis] -= delta
    plus_energy = run_scf(
        plus,
        mixing=1.0,
        tolerance=1.0e-9,
        keep_density_history=False,
    )["energy"]
    minus_energy = run_scf(
        minus,
        mixing=1.0,
        tolerance=1.0e-9,
        keep_density_history=False,
    )["energy"]
    return atom_index, axis, float(plus_energy), float(minus_energy)


def finite_difference_forces(
    positions: np.ndarray,
    *,
    delta: float = 0.003,
    workers: int = 4,
) -> np.ndarray:
    """Return in-plane forces in eV/angstrom from central energy differences."""

    tasks = [
        (positions, atom_index, axis, delta)
        for atom_index in range(len(positions))
        for axis in (0, 1)
    ]
    forces = np.zeros_like(positions)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_shifted_energy, task) for task in tasks]
        for future in as_completed(futures):
            atom_index, axis, plus_energy, minus_energy = future.result()
            derivative_hartree_per_angstrom = (plus_energy - minus_energy) / (2.0 * delta)
            forces[atom_index, axis] = -derivative_hartree_per_angstrom * rhf.HARTREE_TO_EV
    return forces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("three_slide_story/data/h2o_dimer_scf.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("three_slide_story/data/h2o_dimer_scf.json"))
    parser.add_argument("--mixing", type=float, default=0.70)
    parser.add_argument("--skip-forces", action="store_true")
    parser.add_argument("--force-workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    started = time.perf_counter()
    positions = dimer_geometry()
    result = run_scf(positions, mixing=args.mixing)
    x_grid, y_grid, density = density_grid(result["density_matrices"], result["basis"])
    forces = (
        np.zeros_like(positions)
        if args.skip_forces
        else finite_difference_forces(positions, workers=args.force_workers)
    )
    elapsed = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        elements=ELEMENTS,
        positions=positions,
        energies=result["energies"],
        residuals=result["residuals"],
        orbital_energies=result["orbital_energies"],
        grid_x=x_grid,
        grid_y=y_grid,
        density=density,
        forces=forces,
        method=np.array("RHF/STO-3G"),
        mixing=np.array(args.mixing),
        convergence_tolerance=np.array(1.0e-8),
    )
    metadata = {
        "method": "RHF/STO-3G",
        "engine": "md_workflows/engine_rhf.py generalized to the H2O dimer",
        "atoms": ELEMENTS.tolist(),
        "iterations": int(len(result["energies"])),
        "final_energy_hartree": float(result["energy"]),
        "final_delta_energy_hartree": float(abs(result["energies"][-1] - result["energies"][-2])),
        "density_mixing": args.mixing,
        "convergence_tolerance_hartree": 1.0e-8,
        "force_method": "central finite difference of converged RHF energy",
        "force_delta_angstrom": 0.003,
        "max_force_ev_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "wall_seconds": elapsed,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
