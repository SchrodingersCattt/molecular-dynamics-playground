"""Generate a real TIP3P O--O Lennard-Jones Velocity Verlet step.

The water molecules are treated as rigid translating bodies for this focused
demonstration. The highlighted interaction is only the TIP3P oxygen LJ term;
production water models also include electrostatics and constraints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "product"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run_md"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "build_box"))

from engine_md import CONV_ACCEL, velocity_verlet_step  # noqa: E402
from compute_h2o_dimer_scf import dimer_geometry  # noqa: E402


ELEMENTS = np.array(["O", "H", "H", "O", "H", "H"])
MOLECULES = (np.array([0, 1, 2]), np.array([3, 4, 5]))
OXYGENS = np.array([0, 3])
WATER_MASS_AMU = 15.9994 + 2.0 * 1.008
MOLECULE_MASSES = np.array([WATER_MASS_AMU, WATER_MASS_AMU])

# TIP3P oxygen Lennard--Jones parameters.
SIGMA_ANGSTROM = 3.15061
EPSILON_KCAL_PER_MOL = 0.1521
KCAL_PER_MOL_TO_EV = 0.0433641153087705
EPSILON_EV = EPSILON_KCAL_PER_MOL * KCAL_PER_MOL_TO_EV

DT_FS = 2.0
TANGENTIAL_SPEED_ANGSTROM_PER_FS = 0.015
MOTION_FRAMES = 25
DISPLAY_FORCE_SCALE = 4.0
DISPLAY_DISPLACEMENT_SCALE = 24.0
DISPLAY_VELOCITY_SCALE = 55.0


def potential(separation: np.ndarray | float) -> np.ndarray | float:
    ratio = SIGMA_ANGSTROM / np.asarray(separation)
    return 4.0 * EPSILON_EV * (ratio**12 - ratio**6)


def radial_force_on_right(separation: np.ndarray | float) -> np.ndarray | float:
    ratio = SIGMA_ANGSTROM / np.asarray(separation)
    return 24.0 * EPSILON_EV / np.asarray(separation) * (2.0 * ratio**12 - ratio**6)


def generalized_forces(oxygen_positions: np.ndarray) -> np.ndarray:
    delta = oxygen_positions[1] - oxygen_positions[0]
    separation = float(np.linalg.norm(delta))
    axis = delta / separation
    magnitude = float(radial_force_on_right(separation))
    return np.vstack((-magnitude * axis, magnitude * axis))


def apply_rigid_translations(
    reference: np.ndarray,
    oxygen_reference: np.ndarray,
    oxygen_coordinates: np.ndarray,
) -> np.ndarray:
    translated = reference.copy()
    for indices, old_oxygen, new_oxygen in zip(
        MOLECULES, oxygen_reference, oxygen_coordinates
    ):
        translated[indices] += new_oxygen - old_oxygen
    return translated


def write_extxyz(path: Path, frames: np.ndarray) -> None:
    lines: list[str] = []
    for frame_index, frame in enumerate(frames):
        lines.extend(
            [
                str(len(ELEMENTS)),
                f'Properties=species:S:1:pos:R:3 frame={frame_index} source="TIP3P O-O LJ rigid-water VV" pbc="F F F"',
            ]
        )
        for element, xyz in zip(ELEMENTS, frame):
            lines.append(
                f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_reference = dimer_geometry()
    q0 = atomic_reference[OXYGENS].copy()
    axis0 = (q0[1] - q0[0]) / np.linalg.norm(q0[1] - q0[0])
    # Give the two rigid waters equal and opposite transverse velocities.  The
    # out-of-plane component keeps the native 3-D arrows clear of both O--H
    # bonds in the fixed MatterVis camera while remaining perpendicular to the
    # O...O interaction coordinate.
    tangent0 = np.array([0.0, -1.0, 3.0], dtype=float)
    tangent0 -= np.dot(tangent0, axis0) * axis0
    tangent0 /= np.linalg.norm(tangent0)
    v0 = np.vstack(
        (
            TANGENTIAL_SPEED_ANGSTROM_PER_FS * tangent0,
            -TANGENTIAL_SPEED_ANGSTROM_PER_FS * tangent0,
        )
    )
    f0 = generalized_forces(q0)
    a0 = f0 * CONV_ACCEL / MOLECULE_MASSES[:, None]
    q1 = q0 + v0 * DT_FS + 0.5 * a0 * DT_FS**2
    f1 = generalized_forces(q1)
    a1 = f1 * CONV_ACCEL / MOLECULE_MASSES[:, None]
    v1 = v0 + 0.5 * (a0 + a1) * DT_FS

    times = np.linspace(0.0, DT_FS, MOTION_FRAMES)
    q_motion = np.asarray(
        [q0 + v0 * time + 0.5 * a0 * time**2 for time in times]
    )
    atomic_motion = np.asarray(
        [
            apply_rigid_translations(atomic_reference, q0, coordinates)
            for coordinates in q_motion
        ]
    )
    atomic_positions = np.stack((atomic_motion[0], atomic_motion[-1]))

    if not np.allclose(q_motion[-1], q1, atol=1.0e-12):
        raise RuntimeError("Rigid-water interpolation does not end at q[n+1]")
    reference_distances = [
        np.linalg.norm(
            atomic_reference[indices][:, None, :]
            - atomic_reference[indices][None, :, :],
            axis=-1,
        )
        for indices in MOLECULES
    ]
    for frame in atomic_motion:
        for indices, expected in zip(MOLECULES, reference_distances):
            observed = np.linalg.norm(
                frame[indices][:, None, :] - frame[indices][None, :, :], axis=-1
            )
            if not np.allclose(expected, observed, atol=1.0e-12):
                raise RuntimeError("Rigid-water internal geometry changed")

    separations = np.linalg.norm(
        np.stack((q0[1] - q0[0], q1[1] - q1[0])), axis=1
    )
    energies = np.asarray(potential(separations), dtype=float)
    finite_difference_step = 1.0e-5
    finite_difference = -(
        potential(separations + finite_difference_step)
        - potential(separations - finite_difference_step)
    ) / (2.0 * finite_difference_step)
    analytic = np.asarray(radial_force_on_right(separations), dtype=float)
    derivative_error = float(np.max(np.abs(finite_difference - analytic)))
    if derivative_error > 1.0e-8:
        raise RuntimeError(
            f"TIP3P LJ analytic-force validation failed: {derivative_error}"
        )

    # Six reproducible generalized-coordinate steps for the detailed/rapid
    # animation.  Step zero is the existing saved state.
    trajectory_oxygen_positions = [q0.copy()]
    trajectory_velocities = [v0.copy()]
    trajectory_forces = []
    trajectory_accelerations = []
    position, velocity = q0.copy(), v0.copy()
    for _ in range(6):
        force = generalized_forces(position)
        acceleration = force * CONV_ACCEL / MOLECULE_MASSES[:, None]
        delta = position[1] - position[0]
        separation = float(np.linalg.norm(delta))
        axis = delta / separation
        magnitude = float(radial_force_on_right(separation))
        def force_fn(q):
            d = q[1] - q[0]
            r = float(np.linalg.norm(d))
            f = float(radial_force_on_right(r)) * d / r
            return np.vstack((-f, f))
        position, velocity, _ = velocity_verlet_step(
            position, velocity, force_fn, MOLECULE_MASSES, DT_FS
        )
        trajectory_forces.append(force)
        trajectory_accelerations.append(acceleration)
        trajectory_oxygen_positions.append(position.copy())
        trajectory_velocities.append(velocity.copy())
    trajectory_forces.append(generalized_forces(position))
    trajectory_accelerations.append(
        trajectory_forces[-1] * CONV_ACCEL / MOLECULE_MASSES[:, None]
    )
    trajectory_atomic_positions = np.asarray([
        apply_rigid_translations(atomic_reference, q0, coordinates)
        for coordinates in trajectory_oxygen_positions
    ])

    kinetic0 = 0.5 * np.sum(MOLECULE_MASSES[:, None] * v0**2) / CONV_ACCEL
    kinetic1 = 0.5 * np.sum(MOLECULE_MASSES[:, None] * v1**2) / CONV_ACCEL
    total_energies = np.array([energies[0] + kinetic0, energies[1] + kinetic1])

    np.savez_compressed(
        output_dir / "classical_lj.npz",
        elements=ELEMENTS,
        molecule_indices=np.stack(MOLECULES),
        oxygen_indices=OXYGENS,
        molecule_masses_amu=MOLECULE_MASSES,
        sigma_angstrom=SIGMA_ANGSTROM,
        epsilon_ev=EPSILON_EV,
        epsilon_kcal_per_mol=EPSILON_KCAL_PER_MOL,
        dt_fs=DT_FS,
        oxygen_positions=np.stack((q0, q1)),
        molecule_velocities=np.stack((v0, v1)),
        molecule_accelerations=np.stack((a0, a1)),
        molecule_forces=np.stack((f0, f1)),
        atomic_positions=atomic_positions,
        motion_times_fs=times,
        motion_oxygen_positions=q_motion,
        motion_atomic_positions=atomic_motion,
        oo_separations_angstrom=separations,
        lj_energies_ev=energies,
        total_energies_ev=total_energies,
        display_displacement_scale=DISPLAY_DISPLACEMENT_SCALE,
        display_force_scale=DISPLAY_FORCE_SCALE,
        display_velocity_scale=DISPLAY_VELOCITY_SCALE,
        trajectory_oxygen_positions=np.asarray(trajectory_oxygen_positions),
        trajectory_velocities=np.asarray(trajectory_velocities),
        trajectory_forces=np.asarray(trajectory_forces),
        trajectory_accelerations=np.asarray(trajectory_accelerations),
        trajectory_atomic_positions=trajectory_atomic_positions,
        trajectory_times_fs=np.arange(7, dtype=float) * DT_FS,
    )
    metadata = {
        "case": "TIP3P oxygen-oxygen Lennard-Jones term on a real H2O dimer",
        "scope_note": "The highlighted LJ term is not the complete water model; TIP3P also contains electrostatics.",
        "rigid_body_note": "Each water is translated as one rigid generalized coordinate for the focused VV demonstration.",
        "sigma_angstrom": SIGMA_ANGSTROM,
        "epsilon_kcal_per_mol": EPSILON_KCAL_PER_MOL,
        "epsilon_ev": EPSILON_EV,
        "dt_fs": DT_FS,
        "oo_separation_angstrom": {
            "before": float(separations[0]),
            "after": float(separations[1]),
        },
        "lj_energy_ev": {
            "before": float(energies[0]),
            "after": float(energies[1]),
        },
        "radial_force_on_right_ev_per_angstrom": {
            "before": float(analytic[0]),
            "after": float(analytic[1]),
        },
        "analytic_vs_finite_difference_max_error": derivative_error,
        "maximum_rigid_translation_angstrom": float(
            np.linalg.norm(atomic_positions[1] - atomic_positions[0], axis=1).max()
        ),
        "one_step_total_energy_drift_ev": float(
            total_energies[1] - total_energies[0]
        ),
        "display_scales": {
            "displacement": DISPLAY_DISPLACEMENT_SCALE,
            "force": DISPLAY_FORCE_SCALE,
            "velocity": DISPLAY_VELOCITY_SCALE,
        },
        "camera": {
            "projection": "orthographic",
            "direction": [1.55, -1.0, 0.62],
            "up": [0.0, 0.0, 1.0],
        },
    }
    (output_dir / "classical_lj.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_extxyz(output_dir / "classical_lj.extxyz", atomic_positions)
    write_extxyz(output_dir / "classical_lj_motion.extxyz", atomic_motion)
    write_extxyz(output_dir / "classical_lj_trajectory.extxyz", trajectory_atomic_positions)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
