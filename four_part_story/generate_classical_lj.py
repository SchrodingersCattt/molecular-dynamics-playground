from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SIGMA_ANGSTROM = 3.405
EPSILON_EV = 0.0103
R_MIN_ANGSTROM = 2.0 ** (1.0 / 6.0) * SIGMA_ANGSTROM
PAIR_AXIS = np.array([1.0, 0.35, 0.20], dtype=float)
PAIR_AXIS /= np.linalg.norm(PAIR_AXIS)


def potential(r: np.ndarray | float) -> np.ndarray | float:
    ratio = SIGMA_ANGSTROM / np.asarray(r)
    return 4.0 * EPSILON_EV * (ratio**12 - ratio**6)


def radial_force_on_right(r: np.ndarray | float) -> np.ndarray | float:
    ratio = SIGMA_ANGSTROM / np.asarray(r)
    return 24.0 * EPSILON_EV / np.asarray(r) * (2.0 * ratio**12 - ratio**6)


def pair_positions(r: float) -> np.ndarray:
    return np.vstack((-0.5 * r * PAIR_AXIS, 0.5 * r * PAIR_AXIS))


def write_extxyz(path: Path, separations: list[float]) -> None:
    lines = []
    labels = ["repulsive", "equilibrium", "attractive"]
    for frame_index, (label, r) in enumerate(zip(labels, separations)):
        lines.extend(["2", f'Properties=species:S:1:pos:R:3 frame={frame_index} state={label} pbc="F F F"'])
        for xyz in pair_positions(r):
            lines.append(f"Ar {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    r_curve = np.linspace(0.86 * R_MIN_ANGSTROM, 1.65 * R_MIN_ANGSTROM, 700)
    u_curve = potential(r_curve)
    f_curve = radial_force_on_right(r_curve)
    states_r = np.array([0.88 * R_MIN_ANGSTROM, R_MIN_ANGSTROM, 1.32 * R_MIN_ANGSTROM])
    states_u = potential(states_r)
    states_f = radial_force_on_right(states_r)

    delta = 1.0e-5
    finite_difference = -(potential(states_r + delta) - potential(states_r - delta)) / (2.0 * delta)
    derivative_error = float(np.max(np.abs(finite_difference - states_f)))
    equilibrium_force = float(abs(radial_force_on_right(R_MIN_ANGSTROM)))
    if derivative_error > 1.0e-8 or equilibrium_force > 1.0e-12:
        raise RuntimeError("LJ analytic-force validation failed")

    np.savez(
        output_dir / "classical_lj.npz",
        sigma_angstrom=SIGMA_ANGSTROM,
        epsilon_ev=EPSILON_EV,
        r_min_angstrom=R_MIN_ANGSTROM,
        pair_axis=PAIR_AXIS,
        r_curve=r_curve,
        u_curve=u_curve,
        f_curve=f_curve,
        state_r=states_r,
        state_u=states_u,
        state_f=states_f,
        state_positions=np.stack([pair_positions(float(value)) for value in states_r]),
    )
    metadata = {
        "case": "argon pair with the standard 12-6 Lennard-Jones form",
        "sigma_angstrom": SIGMA_ANGSTROM,
        "epsilon_ev": EPSILON_EV,
        "r_min_angstrom": R_MIN_ANGSTROM,
        "states": [
            {"name": name, "r_angstrom": float(r), "U_eV": float(u), "force_on_right_eV_per_angstrom": float(f)}
            for name, r, u, f in zip(["repulsive", "equilibrium", "attractive"], states_r, states_u, states_f)
        ],
        "analytic_vs_finite_difference_max_error": derivative_error,
        "equilibrium_force_abs": equilibrium_force,
        "camera": {"projection": "orthographic", "direction": [1.55, -1.0, 0.62], "up": [0.0, 0.0, 1.0]},
    }
    (output_dir / "classical_lj.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_extxyz(output_dir / "classical_lj.extxyz", list(states_r))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
