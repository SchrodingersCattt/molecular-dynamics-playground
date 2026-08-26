from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "md_workflows"))

from engine_md import CONV_ACCEL, maxwell_boltzmann_velocities, velocity_verlet_step  # noqa: E402
from engine_pes import WATER_EQ, forces, potential_energy  # noqa: E402


MASS = np.array([15.9994, 1.008, 1.008], dtype=float)
ELEMENTS = np.array(["O", "H", "H"])
BONDS = np.array([[0, 1], [0, 2]], dtype=int)
DT_FS = 0.5
DISPLAY_SCALES = {"displacement": 12.0, "acceleration": 25.0, "velocity": 5.0}


def prepared_state() -> tuple[np.ndarray, np.ndarray]:
    positions = WATER_EQ.copy()
    first = positions[1] - positions[0]
    second = positions[2] - positions[0]
    positions[1] += 0.05 * first / np.linalg.norm(first)
    positions[2] -= 0.03 * second / np.linalg.norm(second)
    velocities = maxwell_boltzmann_velocities(MASS, 1200.0, seed=20260827)
    return positions, velocities


def write_extxyz(path: Path, positions: np.ndarray) -> None:
    lines = []
    for frame_index, frame in enumerate(positions):
        lines.extend(
            [
                "3",
                f'Properties=species:S:1:pos:R:3 frame={frame_index} pbc="F F F"',
            ]
        )
        for element, xyz in zip(ELEMENTS, frame):
            lines.append(f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    r0, v0 = prepared_state()
    f0 = forces(r0)
    a0 = f0 * CONV_ACCEL / MASS[:, None]
    r1 = r0 + v0 * DT_FS + 0.5 * a0 * DT_FS**2
    f1 = forces(r1)
    a1 = f1 * CONV_ACCEL / MASS[:, None]
    v1 = v0 + 0.5 * (a0 + a1) * DT_FS

    r_engine, v_engine, f_engine = velocity_verlet_step(r0, v0, forces, MASS, DT_FS)
    residuals = {
        "position_max_abs_angstrom": float(np.max(np.abs(r1 - r_engine))),
        "velocity_max_abs_angstrom_per_fs": float(np.max(np.abs(v1 - v_engine))),
        "force_max_abs_ev_per_angstrom": float(np.max(np.abs(f1 - f_engine))),
    }
    if max(residuals.values()) > 1.0e-12:
        raise RuntimeError(f"Velocity Verlet residual check failed: {residuals}")

    energy0 = potential_energy(r0) + 0.5 * np.sum(MASS[:, None] * v0**2) / CONV_ACCEL
    energy1 = potential_energy(r1) + 0.5 * np.sum(MASS[:, None] * v1**2) / CONV_ACCEL
    np.savez(
        output_dir / "vv_h2o_step.npz",
        elements=ELEMENTS,
        source_ids=np.array(["O0", "H1", "H2"]),
        bonds=BONDS,
        masses=MASS,
        dt_fs=DT_FS,
        positions=np.stack((r0, r1)),
        velocities=np.stack((v0, v1)),
        accelerations=np.stack((a0, a1)),
        forces=np.stack((f0, f1)),
        display_displacement_scale=DISPLAY_SCALES["displacement"],
        display_acceleration_scale=DISPLAY_SCALES["acceleration"],
        display_velocity_scale=DISPLAY_SCALES["velocity"],
    )
    metadata = {
        "case": "one deterministic H2O Velocity Verlet step",
        "dt_fs": DT_FS,
        "initial_temperature_sampling_K": 1200.0,
        "seed": 20260827,
        "display_scales": DISPLAY_SCALES,
        "maximum_physical_displacement_angstrom": float(np.linalg.norm(r1 - r0, axis=1).max()),
        "maximum_acceleration_angstrom_per_fs2": float(np.linalg.norm(a1, axis=1).max()),
        "maximum_velocity_angstrom_per_fs": float(np.linalg.norm(v1, axis=1).max()),
        "energy_eV": {"before": float(energy0), "after": float(energy1), "one_step_drift": float(energy1 - energy0)},
        "equation_residuals": residuals,
        "camera": {"projection": "orthographic", "direction": [1.55, -1.0, 0.62], "up": [0.0, 0.0, 1.0]},
    }
    (output_dir / "vv_h2o_step.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_extxyz(output_dir / "vv_h2o_step.extxyz", np.stack((r0, r1)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
