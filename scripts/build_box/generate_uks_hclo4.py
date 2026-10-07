"""Generate the 03b UKS reactive-AIMD source data.

The preferred path uses a real PySCF UKS/PBE0 calculation.  The repository's
visual review can also be regenerated on systems without a compiler-backed
PySCF installation by passing ``--demo``.  The latter is an explicitly marked
analytic surrogate: it supplies a reproducible reaction-shaped scene for layout
and renderer QA, but it is never labelled as an electronic-structure result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "product" / "data"
QA_DIR = REPO_ROOT / "product" / "qa" / "03b_uks_reaction"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run_md"))

from engine_md import CONV_ACCEL  # noqa: E402
from engine_uks import (  # noqa: E402
    PySCFUnavailable,
    UKSEngine,
    UKSResult,
    electron_count,
    gaussian_density,
    pyscf_available,
)


STEM = "03b_uks_reaction"
ELEMENTS = np.asarray(["Cl", "O", "H", "O", "O", "O"])
CHARGE = 0
SPIN = 0
OH_ATOMS = np.asarray([1, 2], dtype=int)
CLO3_ATOMS = np.asarray([0, 3, 4, 5], dtype=int)
MASSES = np.asarray([35.45, 15.999, 1.008, 15.999, 15.999, 15.999], dtype=float)
DT_FS = 0.10
N_STEPS = 100
RELATIVE_SPEEDS = (0.04, 0.05, 0.06)
GRID_X = np.linspace(-3.5, 9.5, 128)
GRID_Y = np.linspace(-4.0, 4.0, 88)


def initial_geometry() -> np.ndarray:
    """Gas-phase-like distorted-tetrahedral HClO4 reference geometry.

    The OH arm is deliberately bent: experimental/theoretical gas-phase
    structures place Cl--OH near 1.64 Å, terminal Cl--O near 1.41 Å, and
    Cl--O--H near 105 degrees.  Keeping the bend explicit is essential for the
    03b MatterVis view and for a meaningful Cl--O--H angle check.
    """

    cl_oh = 1.64
    cl_o = 1.41
    oh = 0.98
    oh_angle = np.deg2rad(105.0)
    # At O, the Cl->O bond points toward -x.  Place H at 105 degrees from
    # O->Cl, in the xy plane, so the hydroxyl group is visibly bent.
    h_direction = np.asarray([np.cos(np.deg2rad(75.0)), np.sin(np.deg2rad(75.0)), 0.0])
    terminal_x = np.cos(oh_angle) * cl_o
    terminal_rho = np.sin(oh_angle) * cl_o
    return np.asarray(
        [
            [0.00, 0.00, 0.00],  # Cl
            [cl_oh, 0.00, 0.00],  # acidic O
            [cl_oh, 0.00, 0.00] + oh * h_direction,  # acidic H, bent OH arm
            [terminal_x, terminal_rho, 0.00],
            [terminal_x, -0.5 * terminal_rho, np.sqrt(3.0) * 0.5 * terminal_rho],
            [terminal_x, -0.5 * terminal_rho, -np.sqrt(3.0) * 0.5 * terminal_rho],
        ],
        dtype=float,
    )


def reaction_metrics(positions: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    clo = float(np.linalg.norm(positions[1] - positions[0]))
    oh = float(np.linalg.norm(positions[2] - positions[1]))
    oh_centre = positions[OH_ATOMS].mean(axis=0)
    clo3_centre = positions[CLO3_ATOMS].mean(axis=0)
    direction = oh_centre - clo3_centre
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        direction = np.asarray([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    return clo, oh, oh_centre, direction


def kick_velocities(positions: np.ndarray, relative_speed: float) -> np.ndarray:
    """Give OH and ClO3 opposite mass-weighted translational velocities."""

    _, _, _, direction = reaction_metrics(positions)
    m_oh = float(MASSES[OH_ATOMS].sum())
    m_clo3 = float(MASSES[CLO3_ATOMS].sum())
    total = m_oh + m_clo3
    velocities = np.zeros_like(positions, dtype=float)
    velocities[OH_ATOMS] = direction * float(relative_speed) * m_clo3 / total
    velocities[CLO3_ATOMS] = -direction * float(relative_speed) * m_oh / total
    # Numerical cleanup makes the invariant explicit in the saved manifest.
    velocities -= (MASSES[:, None] * velocities).sum(axis=0) / MASSES.sum()
    return velocities


def _bond_harmonic(
    positions: np.ndarray,
    i: int,
    j: int,
    k: float,
    r0: float,
) -> tuple[float, np.ndarray]:
    delta = positions[j] - positions[i]
    distance = float(np.linalg.norm(delta))
    unit = delta / max(distance, 1.0e-12)
    dr = distance - r0
    force_on_i = k * dr * unit
    forces = np.zeros_like(positions)
    forces[i] += force_on_i
    forces[j] -= force_on_i
    return 0.5 * k * dr * dr, forces


def _bond_morse(
    positions: np.ndarray,
    i: int,
    j: int,
    depth: float,
    width: float,
    r0: float,
) -> tuple[float, np.ndarray]:
    delta = positions[j] - positions[i]
    distance = float(np.linalg.norm(delta))
    unit = delta / max(distance, 1.0e-12)
    exponent = np.exp(-width * (distance - r0))
    value = depth * (1.0 - exponent) ** 2 - depth
    # F_j = -dV/dr * rhat; dV/dr = 2 D a exp(-a dr)(1-exp(-a dr)).
    d_v_dr = 2.0 * depth * width * exponent * (1.0 - exponent)
    force_on_j = -d_v_dr * unit
    forces = np.zeros_like(positions)
    forces[j] += force_on_j
    forces[i] -= force_on_j
    return float(value), forces


def analytic_demo_evaluate(positions: np.ndarray, *, step: int, branch_id: int = 0) -> UKSResult:
    """Explicitly labelled reaction-shaped surrogate for renderer QA."""

    positions = np.asarray(positions, dtype=float)
    r_clo, r_oh, oh_centre, _ = reaction_metrics(positions)
    energy = -100.0
    forces = np.zeros_like(positions)
    # The fallback is a visual surrogate rather than a physical HClO4 PES.  A
    # shallow Morse well keeps the requested initial-kick story visible in a
    # short review movie; the real UKS path uses the electronic gradient.
    r_clo0 = float(np.linalg.norm(initial_geometry()[1] - initial_geometry()[0]))
    value, contribution = _bond_morse(positions, 0, 1, depth=0.65, width=1.8, r0=r_clo0)
    energy += value
    forces += contribution
    for i, j in ((1, 2), (0, 3), (0, 4), (0, 5)):
        r0 = float(np.linalg.norm(initial_geometry()[j] - initial_geometry()[i]))
        value, contribution = _bond_harmonic(positions, i, j, k=18.0 if (i, j) == (1, 2) else 7.5, r0=r0)
        energy += value
        forces += contribution

    x = GRID_X
    y = GRID_Y
    clo3_centre = positions[CLO3_ATOMS].mean(axis=0)
    separation_progress = float(np.clip((r_clo - r_clo0) / 2.4, 0.0, 1.0))
    spin_amplitude = 0.62 * separation_progress**0.8
    rho_alpha = gaussian_density(
        x,
        y,
        positions[:, :2],
        np.asarray([0.45, 0.38, 0.30, 0.40, 0.40, 0.40]),
        np.asarray([0.30, 0.50, 0.22, 0.42, 0.42, 0.42]),
    )
    rho_beta = rho_alpha.copy()
    spin_oh = gaussian_density(x, y, oh_centre[None, :2], np.asarray([0.58]), np.asarray([spin_amplitude]))
    spin_clo3 = gaussian_density(x, y, clo3_centre[None, :2], np.asarray([0.70]), np.asarray([spin_amplitude]))
    # Opposite local spin signs are the BS-singlet visual cue: the OH fragment
    # carries positive m(r), while the ClO3 fragment carries negative m(r).
    signed_spin = spin_oh - spin_clo3
    rho_alpha = rho_alpha + 0.5 * signed_spin
    rho_beta = rho_beta - 0.5 * signed_spin
    iterations = 10 + int(5 * separation_progress) + int(step % 3 == 0)
    residuals = np.geomspace(1.0e-2, 1.0e-9, iterations)
    scf_energies = np.linspace(energy + 0.02, energy, iterations)
    return UKSResult(
        energy_ev=float(energy),
        forces_ev_ang=forces,
        dm_alpha=np.zeros((1, 1)),
        dm_beta=np.zeros((1, 1)),
        scf_energies_ev=scf_energies,
        scf_residuals=residuals,
        spin_square=float(separation_progress**2),
        spin_multiplicity=float(1.0 + separation_progress),
        spin_density_x=x,
        spin_density_y=y,
        rho_alpha=rho_alpha,
        rho_beta=rho_beta,
        converged=True,
        iterations=iterations,
        branch_id=int(branch_id),
    )


def _serialize_frame(result: UKSResult, positions: np.ndarray, velocities: np.ndarray, step: int) -> dict[str, object]:
    clo, oh, _, _ = reaction_metrics(positions)
    return {
        "step": int(step),
        "positions": np.asarray(positions, dtype=float),
        "velocities": np.asarray(velocities, dtype=float),
        "forces": np.asarray(result.forces_ev_ang, dtype=float),
        "energy_ev": float(result.energy_ev),
        "r_clo": float(clo),
        "r_oh": float(oh),
        "scf_energies_ev": np.asarray(result.scf_energies_ev, dtype=float),
        "scf_residuals": np.asarray(result.scf_residuals, dtype=float),
        "spin_square": float(result.spin_square),
        "spin_multiplicity": float(result.spin_multiplicity),
        "spin_density_x": np.asarray(result.spin_density_x, dtype=float),
        "spin_density_y": np.asarray(result.spin_density_y, dtype=float),
        "rho_alpha": np.asarray(result.rho_alpha, dtype=float),
        "rho_beta": np.asarray(result.rho_beta, dtype=float),
        "spin_density_metric": float(result.spin_density_metric),
        "scf_iterations": int(result.iterations),
        "converged": bool(result.converged),
        "branch_id": int(result.branch_id),
    }


def run_trajectory(
    evaluator: Callable[[np.ndarray, int, int], UKSResult],
    *,
    relative_speed: float,
    n_steps: int = N_STEPS,
    dt_fs: float = DT_FS,
) -> list[dict[str, object]]:
    positions = initial_geometry()
    velocities = kick_velocities(positions, relative_speed)
    branch_id = 0
    result = evaluator(positions, 0, branch_id)
    frames = [_serialize_frame(result, positions, velocities, 0)]
    for step in range(1, int(n_steps) + 1):
        acceleration = result.forces_ev_ang * CONV_ACCEL / MASSES[:, None]
        velocity_half = velocities + 0.5 * acceleration * dt_fs
        next_positions = positions + velocity_half * dt_fs
        try:
            result_next = evaluator(next_positions, step, branch_id)
        except RuntimeError:
            branch_id += 1
            result_next = evaluator(next_positions, step, branch_id)
        acceleration_next = result_next.forces_ev_ang * CONV_ACCEL / MASSES[:, None]
        next_velocities = velocity_half + 0.5 * acceleration_next * dt_fs
        frames.append(_serialize_frame(result_next, next_positions, next_velocities, step))
        positions = next_positions
        velocities = next_velocities
        result = result_next
    return frames


def _pad_history(values: list[np.ndarray], fill: float = np.nan) -> np.ndarray:
    width = max(len(value) for value in values)
    output = np.full((len(values), width), fill, dtype=float)
    for row, value in enumerate(values):
        output[row, : len(value)] = value
    return output


def save_dataset(frames: list[dict[str, object]], *, backend: str, relative_speed: float) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.asarray([frame["positions"] for frame in frames], dtype=float)
    velocities = np.asarray([frame["velocities"] for frame in frames], dtype=float)
    forces = np.asarray([frame["forces"] for frame in frames], dtype=float)
    energies = np.asarray([frame["energy_ev"] for frame in frames], dtype=float)
    r_clo = np.asarray([frame["r_clo"] for frame in frames], dtype=float)
    r_oh = np.asarray([frame["r_oh"] for frame in frames], dtype=float)
    spin_square = np.asarray([frame["spin_square"] for frame in frames], dtype=float)
    spin_multiplicity = np.asarray([frame["spin_multiplicity"] for frame in frames], dtype=float)
    spin_metric = np.asarray([frame["spin_density_metric"] for frame in frames], dtype=float)
    iterations = np.asarray([frame["scf_iterations"] for frame in frames], dtype=int)
    converged = np.asarray([frame["converged"] for frame in frames], dtype=bool)
    branch_ids = np.asarray([frame["branch_id"] for frame in frames], dtype=int)
    rho_alpha = np.asarray([frame["rho_alpha"] for frame in frames], dtype=float)
    rho_beta = np.asarray([frame["rho_beta"] for frame in frames], dtype=float)
    spin_x = np.asarray(frames[0]["spin_density_x"], dtype=float)
    spin_y = np.asarray(frames[0]["spin_density_y"], dtype=float)
    scf_energies = _pad_history([np.asarray(frame["scf_energies_ev"]) for frame in frames])
    scf_residuals = _pad_history([np.asarray(frame["scf_residuals"]) for frame in frames])
    output = DATA_DIR / "uks_hclo4_reaction.npz"
    np.savez_compressed(
        output,
        elements=ELEMENTS,
        positions=positions,
        velocities=velocities,
        forces=forces,
        energy_ev=energies,
        r_clo=r_clo,
        r_oh=r_oh,
        spin_square=spin_square,
        spin_multiplicity=spin_multiplicity,
        spin_density_metric=spin_metric,
        scf_iterations=iterations,
        converged=converged,
        branch_ids=branch_ids,
        spin_density_x=spin_x,
        spin_density_y=spin_y,
        rho_alpha=rho_alpha,
        rho_beta=rho_beta,
        scf_energies_ev=scf_energies,
        scf_residuals=scf_residuals,
        dt_fs=np.asarray(DT_FS),
        relative_speed=np.asarray(relative_speed),
        charge=np.asarray(CHARGE),
        spin=np.asarray(SPIN),
    )
    payload = {
        "schema_version": 1,
        "stem": STEM,
        "backend": backend,
        "method": "UKS/PBE0/def2-SVP" if backend == "pyscf_uks" else "analytic surrogate for visual QA",
        "elements": ELEMENTS.tolist(),
        "charge": CHARGE,
        "spin": SPIN,
        "electron_count": electron_count(ELEMENTS, CHARGE),
        "reaction": "HClO4 -> OH radical + ClO3 radical (Cl-O(H) homolytic dissociation)",
        "trajectory": "kick-started, non-equilibrium, qualitative BOMD",
        "geometry_source": "fixed hand-tuned tetrahedral reference geometry; no geometry optimizer invoked",
        "dt_fs": DT_FS,
        "n_steps": len(frames) - 1,
        "relative_speed_ang_fs": float(relative_speed),
        "mass_weighted_com_velocity_initial_ang_fs": (
            (MASSES[:, None] * velocities[0]).sum(axis=0) / MASSES.sum()
        ).tolist(),
        "mass_weighted_com_velocity_max_ang_fs": (
            np.max(
                np.linalg.norm(
                    (MASSES[None, :, None] * velocities).sum(axis=1) / MASSES.sum(),
                    axis=1,
                )
            )
        ).item(),
        "oh_atoms": OH_ATOMS.tolist(),
        "clo3_atoms": CLO3_ATOMS.tolist(),
        "npz": str(output),
        "source_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "notes": [
            "The analytic backend is for layout and renderer QA only; it is not UKS data.",
            "A real run requires a platform-supported PySCF installation.",
        ] if backend != "pyscf_uks" else [
            "UKS density branch is state-followed from the previous converged alpha/beta density.",
            "Energetics are qualitative and are not a production barrier or rate calculation.",
        ],
    }
    manifest = DATA_DIR / "uks_hclo4_reaction.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def choose_speed(*, demo: bool, requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    # The fallback is deterministic and uses the middle pilot value.
    return 0.05 if demo else 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 03b HClO4 UKS reactive-AIMD dataset")
    parser.add_argument("--demo", action="store_true", help="use the labelled analytic surrogate when PySCF is unavailable")
    parser.add_argument("--force", action="store_true", help="overwrite the existing dataset")
    parser.add_argument("--speed", type=float, default=None, help="OH/ClO3 relative speed in Angstrom/fs")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    args = parser.parse_args()
    output = DATA_DIR / "uks_hclo4_reaction.npz"
    if output.exists() and not args.force:
        print(f"[INFO] {output} already exists; use --force to regenerate")
        return
    demo = bool(args.demo)
    if not demo and not pyscf_available():
        raise PySCFUnavailable(
            "PySCF is not installed. Use --demo for a labelled visual surrogate, "
            "or install a platform-supported PySCF build for real UKS data."
        )
    speed = choose_speed(demo=demo, requested=args.speed)
    if demo:
        def evaluator(positions: np.ndarray, step: int, branch: int) -> UKSResult:
            return analytic_demo_evaluate(positions, step=step, branch_id=branch)

        backend = "analytic_demo"
    else:
        engine = UKSEngine(ELEMENTS, charge=CHARGE, spin=SPIN, basis="def2-svp", xc="pbe0")
        state: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"dm": None}

        def evaluator(positions: np.ndarray, step: int, branch: int) -> UKSResult:
            result = engine.evaluate(positions, dm0=state["dm"], branch_id=branch, seed_if_missing=state["dm"] is None)
            state["dm"] = result.density_matrix
            return result

        backend = "pyscf_uks"
    frames = run_trajectory(evaluator, relative_speed=speed, n_steps=args.steps)
    save_dataset(frames, backend=backend, relative_speed=speed)
    print(f"Saved {len(frames)} frames ({backend}) → {output}")


if __name__ == "__main__":
    main()

