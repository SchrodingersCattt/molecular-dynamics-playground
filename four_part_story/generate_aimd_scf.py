"""Generate the complete 19-iteration RHF density story for AIMD.

Every displayed density starts from the AO density matrix saved at that exact
SCF iteration. A residual-driven Gaussian blur is stored separately as a
visual convergence encoding; the unmodified density is retained alongside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "md_workflows"))

import engine_rhf as rhf  # noqa: E402
from three_slide_story.compute_h2o_dimer_scf import (  # noqa: E402
    ELEMENTS,
    run_scf,
)


DATA_DIR = ROOT / "data"
CUBE_DIR = ROOT / "_qa" / "03_aimd_scf" / "source" / "cubes"
ATOMIC_NUMBERS = {"H": 1, "O": 8}
TIME_STEP_FS = 0.50
ACCELERATION_FACTOR = 0.00964853322  # (eV/A)/amu -> A/fs^2
DISPLAY_MOTION_SCALE = 80.0
MOTION_FRAMES = 25
MAX_BLUR_SIGMA_VOXELS = 3.0
FORCE_DELTA_ANGSTROM = 0.003


def presentation_dimer_geometry() -> np.ndarray:
    """Return an asymmetric, non-planar water dimer for readable true forces."""

    bond = 0.990
    angle = np.deg2rad(102.5)
    oxygen_1 = np.array([-1.45, -0.10, -0.10])
    oxygen_2 = np.array([1.42, 0.18, 0.20])

    def unit(vector: np.ndarray | list[float]) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        return value / np.linalg.norm(value)

    donor_direction = unit([0.90, -0.35, 0.25])
    donor_plane = np.array([0.05, 0.80, 0.58])
    donor_perpendicular = unit(
        donor_plane - np.dot(donor_plane, donor_direction) * donor_direction
    )
    donor_other = (
        np.cos(angle) * donor_direction
        + np.sin(angle) * donor_perpendicular
    )

    acceptor_bisector = unit([0.80, -0.22, 0.56])
    acceptor_plane = np.array([0.15, 0.96, 0.16])
    acceptor_perpendicular = unit(
        acceptor_plane
        - np.dot(acceptor_plane, acceptor_bisector) * acceptor_bisector
    )
    half_angle = angle / 2.0
    acceptor_first = (
        np.cos(half_angle) * acceptor_bisector
        + np.sin(half_angle) * acceptor_perpendicular
    )
    acceptor_second = (
        np.cos(half_angle) * acceptor_bisector
        - np.sin(half_angle) * acceptor_perpendicular
    )
    return np.array(
        [
            oxygen_1,
            oxygen_1 + bond * donor_direction,
            oxygen_1 + bond * donor_other,
            oxygen_2,
            oxygen_2 + bond * acceptor_first,
            oxygen_2 + bond * acceptor_second,
        ],
        dtype=float,
    )


def _shifted_energy_3d(
    task: tuple[np.ndarray, int, int, float],
) -> tuple[int, int, float, float]:
    positions, atom_index, axis, delta = task
    plus = positions.copy()
    minus = positions.copy()
    plus[atom_index, axis] += delta
    minus[atom_index, axis] -= delta
    plus_energy = run_scf(
        plus, mixing=1.0, tolerance=1.0e-9, keep_density_history=False
    )["energy"]
    minus_energy = run_scf(
        minus, mixing=1.0, tolerance=1.0e-9, keep_density_history=False
    )["energy"]
    return atom_index, axis, float(plus_energy), float(minus_energy)


def finite_difference_forces_3d(
    positions: np.ndarray,
    *,
    delta: float = FORCE_DELTA_ANGSTROM,
    workers: int = 4,
) -> np.ndarray:
    """Return all Cartesian nuclear forces from converged RHF energies."""

    tasks = [
        (positions, atom_index, axis, delta)
        for atom_index in range(len(positions))
        for axis in range(3)
    ]
    forces = np.zeros_like(positions)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_shifted_energy_3d, task) for task in tasks]
        for future in as_completed(futures):
            atom_index, axis, plus_energy, minus_energy = future.result()
            derivative_hartree_per_angstrom = (
                plus_energy - minus_energy
            ) / (2.0 * delta)
            forces[atom_index, axis] = (
                -derivative_hartree_per_angstrom * rhf.HARTREE_TO_EV
            )
    return forces


def density_volumes(
    density_matrices: np.ndarray,
    basis: list[dict],
    selected_iterations: np.ndarray,
    *,
    nx: int = 68,
    ny: int = 50,
    nz: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate each AO density matrix on one fixed Cartesian grid."""
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

    volumes: list[np.ndarray] = []
    for iteration in selected_iterations:
        matrix = density_matrices[int(iteration)]
        density = np.einsum(
            "mn,mxyz,nxyz->xyz", matrix, ao_values, ao_values, optimize=True
        )
        volumes.append(np.maximum(density, 0.0).astype(np.float32))
    return x_angstrom, y_angstrom, z_angstrom, np.asarray(volumes)


def blur_sigmas(residuals: np.ndarray) -> np.ndarray:
    """Map the real SCF residual envelope monotonically from coarse to sharp."""
    residuals = np.maximum(np.asarray(residuals, dtype=float), 1.0e-30)
    logs = np.log10(residuals)
    denominator = max(float(logs[0] - logs[-1]), 1.0e-12)
    fraction = np.clip((logs - logs[-1]) / denominator, 0.0, 1.0)
    sigmas = MAX_BLUR_SIGMA_VOXELS * fraction
    sigmas = np.maximum.accumulate(sigmas[::-1])[::-1]
    sigmas[-1] = 0.0
    return sigmas


def write_cube(
    path: Path,
    positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    volume: np.ndarray,
    *,
    comment: str,
) -> None:
    """Write one standards-compliant Gaussian Cube in Bohr coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    origin_bohr = np.array([grid_x[0], grid_y[0], grid_z[0]]) * rhf.ANG_TO_BOHR
    steps_bohr = np.array(
        [np.diff(grid_x).mean(), np.diff(grid_y).mean(), np.diff(grid_z).mean()]
    ) * rhf.ANG_TO_BOHR
    lines = [
        "RHF/STO-3G H2O dimer electron density",
        comment,
        f"{len(ELEMENTS):5d} {origin_bohr[0]:13.6f} {origin_bohr[1]:13.6f} {origin_bohr[2]:13.6f}",
        f"{len(grid_x):5d} {steps_bohr[0]:13.6f} {0.0:13.6f} {0.0:13.6f}",
        f"{len(grid_y):5d} {0.0:13.6f} {steps_bohr[1]:13.6f} {0.0:13.6f}",
        f"{len(grid_z):5d} {0.0:13.6f} {0.0:13.6f} {steps_bohr[2]:13.6f}",
    ]
    for element, position in zip(ELEMENTS, positions):
        atomic_number = ATOMIC_NUMBERS[str(element)]
        xyz_bohr = np.asarray(position) * rhf.ANG_TO_BOHR
        lines.append(
            f"{atomic_number:5d} {float(atomic_number):13.6f} "
            f"{xyz_bohr[0]:13.6f} {xyz_bohr[1]:13.6f} {xyz_bohr[2]:13.6f}"
        )
    flat = np.asarray(volume, dtype=float).ravel(order="C")
    for start in range(0, len(flat), 6):
        lines.append(" ".join(f"{value:13.5e}" for value in flat[start : start + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_extxyz_trajectory(path: Path, frames: np.ndarray, *, source: str) -> None:
    lines: list[str] = []
    for frame_index, positions in enumerate(frames):
        lines.extend(
            [
                str(len(ELEMENTS)),
                f'Properties=species:S:1:pos:R:3 frame={frame_index} source="{source}" pbc="F F F"',
            ]
        )
        for element, position in zip(ELEMENTS, positions):
            lines.append(
                f"{element} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-forces", action="store_true")
    parser.add_argument("--force-workers", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    positions = presentation_dimer_geometry()
    result = run_scf(positions, mixing=0.70, tolerance=1.0e-8)
    if len(result["energies"]) != 19:
        raise RuntimeError(
            f"Expected 19 actual SCF iterations for the story, got {len(result['energies'])}"
        )
    selected = np.arange(len(result["energies"]), dtype=int)
    grid_x, grid_y, grid_z, raw_volumes = density_volumes(
        result["density_matrices"], result["basis"], selected
    )
    sigmas = blur_sigmas(result["residuals"])
    display_volumes = np.asarray(
        [
            gaussian_filter(volume, sigma=float(sigma), mode="nearest").astype(
                np.float32
            )
            if sigma > 1.0e-9
            else volume.copy()
            for volume, sigma in zip(raw_volumes, sigmas)
        ]
    )

    if args.skip_forces:
        previous = np.load(DATA_DIR / "aimd_h2o_dimer.npz")
        if not np.allclose(previous["positions"], positions, atol=1.0e-12):
            raise RuntimeError("Cached forces do not belong to the current dimer geometry")
        forces = np.asarray(previous["forces"], dtype=float)
        force_origin = "verified cached 3D central finite difference of converged RHF energy"
    else:
        forces = finite_difference_forces_3d(
            positions, workers=args.force_workers
        )
        force_origin = "fresh 3D central finite difference of converged RHF energy"

    masses = np.array([15.999 if element == "O" else 1.008 for element in ELEMENTS])
    accelerations = forces / masses[:, None] * ACCELERATION_FACTOR
    vv_displacement = 0.5 * accelerations * TIME_STEP_FS**2
    fractions = np.linspace(0.0, 1.0, MOTION_FRAMES) ** 2
    physical_motion = np.asarray(
        [positions + fraction * vv_displacement for fraction in fractions]
    )
    display_motion = np.asarray(
        [
            positions + DISPLAY_MOTION_SCALE * fraction * vv_displacement
            for fraction in fractions
        ]
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    cube_records: list[dict] = []
    for iteration, (raw, display, residual, sigma) in enumerate(
        zip(raw_volumes, display_volumes, result["residuals"], sigmas)
    ):
        raw_path = CUBE_DIR / f"rho_{iteration + 1:02d}_raw.cube"
        display_path = CUBE_DIR / f"rho_{iteration + 1:02d}_display.cube"
        write_cube(
            raw_path,
            positions,
            grid_x,
            grid_y,
            grid_z,
            raw,
            comment=f"SCF iteration {iteration + 1}/19; unmodified rho^k(r)",
        )
        write_cube(
            display_path,
            positions,
            grid_x,
            grid_y,
            grid_z,
            display,
            comment=(
                f"SCF iteration {iteration + 1}/19; residual-driven blur "
                f"sigma={sigma:.6f} voxels"
            ),
        )
        cube_records.append(
            {
                "iteration_one_based": iteration + 1,
                "residual": float(residual),
                "blur_sigma_voxels": float(sigma),
                "raw_cube": str(raw_path.relative_to(ROOT)),
                "display_cube": str(display_path.relative_to(ROOT)),
                "raw_sha256": sha256(raw_path),
                "display_sha256": sha256(display_path),
            }
        )

    np.savez_compressed(
        DATA_DIR / "aimd_h2o_dimer.npz",
        elements=ELEMENTS,
        positions=positions,
        bonds=np.array([[0, 1], [0, 2], [3, 4], [3, 5]], dtype=int),
        hydrogen_bond=np.array([1, 3], dtype=int),
        energies=result["energies"],
        residuals=result["residuals"],
        forces=forces,
        accelerations=accelerations,
        vv_displacement=vv_displacement,
        physical_motion_positions=physical_motion,
        display_motion_positions=display_motion,
        display_motion_scale=np.array(DISPLAY_MOTION_SCALE),
        time_step_fs=np.array(TIME_STEP_FS),
        density_iterations=selected,
        density_blur_sigma_voxels=sigmas,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        density_volumes_raw=raw_volumes,
        density_volumes_display=display_volumes,
        method=np.array("RHF/STO-3G"),
        mixing=np.array(0.70),
        convergence_tolerance=np.array(1.0e-8),
    )
    write_extxyz_trajectory(
        DATA_DIR / "aimd_h2o_dimer_motion_true.extxyz",
        physical_motion,
        source="RHF force, zero initial velocity, true 0.5 fs position update",
    )
    write_extxyz_trajectory(
        DATA_DIR / "aimd_h2o_dimer_motion_display.extxyz",
        display_motion,
        source="RHF force VV displacement with documented visual amplification",
    )

    final_delta = float(abs(result["energies"][-1] - result["energies"][-2]))
    metadata = {
        "method": "RHF/STO-3G",
        "engine": "project pure-NumPy Gaussian integral engine",
        "geometry": "asymmetric non-planar presentation dimer; forces remain unmodified RHF finite differences",
        "atoms": ELEMENTS.tolist(),
        "iterations": int(len(result["energies"])),
        "saved_density_iterations_zero_based": selected.tolist(),
        "density_grid_shape": list(raw_volumes.shape[1:]),
        "density_grid_units": "electron density from AO density matrix on an angstrom coordinate grid",
        "density_visual_encoding": (
            "Each displayed field is the real rho^k(r); Gaussian blur sigma is a "
            "monotonic function of the saved SCF density residual."
        ),
        "density_blur_sigma_voxels": sigmas.tolist(),
        "cube_records": cube_records,
        "final_energy_hartree": float(result["energy"]),
        "final_delta_energy_hartree": final_delta,
        "density_mixing": 0.70,
        "convergence_tolerance_hartree": 1.0e-8,
        "force_method": force_origin,
        "force_delta_angstrom": FORCE_DELTA_ANGSTROM,
        "max_force_ev_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "time_step_fs": TIME_STEP_FS,
        "display_motion_scale": DISPLAY_MOTION_SCALE,
        "max_true_vv_displacement_angstrom": float(
            np.linalg.norm(vv_displacement, axis=1).max()
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    (DATA_DIR / "aimd_h2o_dimer.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
