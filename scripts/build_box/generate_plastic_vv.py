"""Generate the 01 plastic-ball Velocity-Verlet dataset.

Nine identical, argon-like spheres interact through the Lennard-Jones 12-6 pair
potential.  The initial velocities follow the same recipe as LAMMPS

    velocity all create T seed dist gaussian mom yes

* every atom draws one independent Gaussian triple (vx, vy, vz);
* the centre-of-mass velocity is subtracted (``mom yes``);
* all velocities are multiplied by one factor so that the instantaneous
  temperature equals the requested T exactly.

The raw draw is expressed in units of sigma_T = sqrt(k_B T / m).  LAMMPS draws
unit Gaussians and divides by sqrt(m); the two differ by one global constant
that the final rescale removes, so the converged velocities are identical.  The
three intermediate states (raw draw, drift removed, rescaled) are stored because
the animation shows each of them.

The step length (80 fs) is chosen for visibility; omega * dt stays below 0.5 for
every contact, so the integrator remains stable and nearly energy conserving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "product" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run_md"))

from engine_md import CONV_ACCEL, KB_EV  # noqa: E402

N_ATOMS = 9
MASS_AMU = 39.948
LJ_SIGMA = 3.40  # Angstrom
LJ_EPSILON = 0.060  # eV, deeper than argon so that the nine-ball cluster stays bound
BALL_RADIUS = 1.65  # Angstrom, display radius only
TEMPERATURE_K = 100.0
DT_FS = 80.0
N_STATES = 8  # r_0 ... r_7 (seven velocity-Verlet updates)
POSITION_SEED = 496
VELOCITY_SEED = 11
DEGREES_OF_FREEDOM = 3 * N_ATOMS - 3


def lj_energy_forces(positions: np.ndarray) -> tuple[float, np.ndarray]:
    energy = 0.0
    forces = np.zeros_like(positions)
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            delta = positions[i] - positions[j]
            r2 = float(delta @ delta)
            s6 = (LJ_SIGMA**2 / r2) ** 3
            energy += 4.0 * LJ_EPSILON * (s6 * s6 - s6)
            # -dE/dr * rhat with rhat = delta / r
            magnitude = 24.0 * LJ_EPSILON * (2.0 * s6 * s6 - s6) / r2
            force = magnitude * delta
            forces[i] += force
            forces[j] -= force
    return energy, forces


def build_pile() -> np.ndarray:
    """Loose random pile of nine balls whose neighbours sit near the LJ force maximum.

    Pair distances stay above 4.0 A (the LJ force maximum lies at 1.24 sigma =
    4.2 A), so contacts are gentle and the 80 fs step conserves energy.
    """
    rng = np.random.default_rng(POSITION_SEED)
    points: list[np.ndarray] = []
    for _ in range(20000):
        if len(points) == N_ATOMS:
            break
        candidate = rng.uniform(-6.2, 6.2, size=3)
        if np.linalg.norm(candidate) > 6.2:
            continue
        if all(np.linalg.norm(candidate - other) > 4.0 for other in points):
            points.append(candidate)
    if len(points) != N_ATOMS:
        raise RuntimeError("could not place nine balls")
    positions = np.asarray(points)
    return positions - positions.mean(axis=0)


def temperature(velocities: np.ndarray) -> float:
    kinetic = 0.5 * MASS_AMU * float((velocities**2).sum()) / CONV_ACCEL
    return 2.0 * kinetic / (DEGREES_OF_FREEDOM * KB_EV)


def lammps_style_velocities(seed: int) -> dict[str, np.ndarray | float]:
    sigma_t = float(np.sqrt(KB_EV * TEMPERATURE_K * CONV_ACCEL / MASS_AMU))
    rng = np.random.default_rng(seed)
    raw = sigma_t * rng.standard_normal((N_ATOMS, 3))
    vcm = raw.mean(axis=0)  # equal masses
    no_drift = raw - vcm
    t_before = temperature(no_drift)
    factor = float(np.sqrt(TEMPERATURE_K / t_before))
    final = factor * no_drift
    return {
        "sigma_t": sigma_t,
        "raw": raw,
        "vcm": vcm,
        "no_drift": no_drift,
        "scale_factor": factor,
        "final": final,
        "temperature_before_rescale": t_before,
    }


def choose_velocity_seed(start: int) -> int:
    """Pick a seed whose drift and rescale are both visible on screen."""
    for seed in range(start, start + 400):
        draw = lammps_style_velocities(seed)
        sigma_t = float(draw["sigma_t"])
        drift = float(np.linalg.norm(draw["vcm"])) / sigma_t
        factor = float(draw["scale_factor"])
        raw = np.asarray(draw["raw"]) / sigma_t
        within = np.abs(np.asarray(draw["final"]) / sigma_t).max() < 2.6 and np.abs(raw).max() < 2.7
        if 0.55 <= drift <= 0.85 and 1.14 <= factor <= 1.28 and within:
            return seed
    raise RuntimeError("no suitable velocity seed found")


def run_trajectory(positions0: np.ndarray, velocities0: np.ndarray) -> dict[str, np.ndarray]:
    positions = [positions0.copy()]
    full_velocities = [velocities0.copy()]
    half_velocities: list[np.ndarray] = []
    forces_list: list[np.ndarray] = []
    potentials: list[float] = []
    energy, forces = lj_energy_forces(positions0)
    potentials.append(energy)
    forces_list.append(forces)
    for _ in range(N_STATES - 1):
        accel = forces_list[-1] * CONV_ACCEL / MASS_AMU
        v_half = full_velocities[-1] + 0.5 * accel * DT_FS
        new_positions = positions[-1] + v_half * DT_FS
        energy, new_forces = lj_energy_forces(new_positions)
        new_accel = new_forces * CONV_ACCEL / MASS_AMU
        new_velocities = v_half + 0.5 * new_accel * DT_FS
        half_velocities.append(v_half)
        positions.append(new_positions)
        full_velocities.append(new_velocities)
        forces_list.append(new_forces)
        potentials.append(energy)
    forces_array = np.asarray(forces_list)
    return {
        "positions": np.asarray(positions),
        "velocities": np.asarray(full_velocities),
        "half_velocities": np.asarray(half_velocities),
        "forces": forces_array,
        "accelerations": forces_array * CONV_ACCEL / MASS_AMU,
        "potential_ev": np.asarray(potentials),
    }


def central_difference_check(positions: np.ndarray, forces: np.ndarray, h: float = 1.0e-5) -> float:
    numeric = np.zeros_like(positions)
    for i in range(len(positions)):
        for axis in range(3):
            shifted = positions.copy()
            shifted[i, axis] += h
            plus, _ = lj_energy_forces(shifted)
            shifted[i, axis] -= 2.0 * h
            minus, _ = lj_energy_forces(shifted)
            numeric[i, axis] = -(plus - minus) / (2.0 * h)
    return float(np.max(np.abs(numeric - forces)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 01 plastic-ball VV data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = DATA_DIR / "vv_plastic_balls.npz"
    if output.exists() and not args.force:
        print(f"[INFO] {output} exists; use --force to regenerate")
        return
    positions0 = build_pile()
    seed = choose_velocity_seed(VELOCITY_SEED)
    draw = lammps_style_velocities(seed)
    trajectory = run_trajectory(positions0, np.asarray(draw["final"]))
    kinetic = 0.5 * MASS_AMU * (trajectory["velocities"] ** 2).sum(axis=(1, 2)) / CONV_ACCEL
    total = kinetic + trajectory["potential_ev"]
    pair_distances = [
        float(np.linalg.norm(positions0[i] - positions0[j]))
        for i in range(N_ATOMS)
        for j in range(i + 1, N_ATOMS)
    ]
    checks = {
        "min_pair_distance_ang": min(pair_distances),
        "ball_diameter_ang": 2.0 * BALL_RADIUS,
        "max_force_central_difference_error_ev_ang": central_difference_check(
            positions0, trajectory["forces"][0]
        ),
        "net_force_max_ev_ang": float(np.abs(trajectory["forces"].sum(axis=1)).max()),
        "initial_temperature_k": temperature(np.asarray(draw["final"])),
        "initial_com_velocity_max": float(np.abs(np.asarray(draw["final"]).mean(axis=0)).max()),
        "total_energy_drift_ev": float(total.max() - total.min()),
        "kinetic_energy_ev": kinetic.tolist(),
    }
    if checks["min_pair_distance_ang"] <= checks["ball_diameter_ang"] + 0.3:
        raise RuntimeError("balls overlap or touch; adjust the pile")
    if checks["max_force_central_difference_error_ev_ang"] > 1.0e-6:
        raise RuntimeError("analytic LJ force disagrees with central differences")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions=trajectory["positions"],
        velocities=trajectory["velocities"],
        half_velocities=trajectory["half_velocities"],
        forces=trajectory["forces"],
        accelerations=trajectory["accelerations"],
        potential_ev=trajectory["potential_ev"],
        velocity_raw=np.asarray(draw["raw"]),
        velocity_no_drift=np.asarray(draw["no_drift"]),
        velocity_com_removed=np.asarray(draw["vcm"]),
        velocity_scale_factor=np.asarray(draw["scale_factor"]),
        sigma_t=np.asarray(draw["sigma_t"]),
        mass_amu=np.asarray(MASS_AMU),
        dt_fs=np.asarray(DT_FS),
        ball_radius=np.asarray(BALL_RADIUS),
        temperature_k=np.asarray(TEMPERATURE_K),
    )
    manifest = {
        "schema_version": 1,
        "stem": "01_velocity_verlet",
        "system": "nine identical spheres, LJ 12-6 pair potential (argon-like sigma and mass, deeper well), free cluster",
        "n_atoms": N_ATOMS,
        "mass_amu": MASS_AMU,
        "lj_sigma_ang": LJ_SIGMA,
        "lj_epsilon_ev": LJ_EPSILON,
        "display_ball_radius_ang": BALL_RADIUS,
        "dt_fs": DT_FS,
        "dt_note": "80 fs is a visibility choice; omega*dt < 0.5 at every LJ contact so velocity Verlet stays stable",
        "temperature_k": TEMPERATURE_K,
        "position_seed": POSITION_SEED,
        "velocity_seed": seed,
        "velocity_recipe": "LAMMPS: velocity all create T seed dist gaussian mom yes (per-atom Gaussian triple, remove COM velocity, rescale to T)",
        "sigma_t_ang_per_fs": float(draw["sigma_t"]),
        "com_velocity_removed_in_sigma_t": float(np.linalg.norm(draw["vcm"]) / float(draw["sigma_t"])),
        "scale_factor": float(draw["scale_factor"]),
        "temperature_before_rescale_k": float(draw["temperature_before_rescale"]),
        "degrees_of_freedom": DEGREES_OF_FREEDOM,
        "checks": checks,
        "npz": str(output),
        "npz_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (DATA_DIR / "vv_plastic_balls.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "checks"}, indent=2))
    print(json.dumps({k: v for k, v in checks.items() if k != "kinetic_energy_ev"}, indent=2))


if __name__ == "__main__":
    main()
