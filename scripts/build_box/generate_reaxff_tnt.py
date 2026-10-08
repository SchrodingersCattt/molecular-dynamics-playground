"""Generate the 02b schematic ReaxFF-style TNT C2--NO2 trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "product" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run_md"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_md import CONV_ACCEL  # noqa: E402
from generate_uks_tnt import (  # noqa: E402
    BREAKING_C,
    BREAKING_N,
    ELEMENTS,
    MASSES,
    initial_geometry,
    kick_velocities,
)

STEM = "02b_reaxff"
DT_FS = 0.10
N_STEPS = 500
RELATIVE_SPEED = 0.05

RING = tuple(range(6))
BONDS: tuple[tuple[int, int], ...] = (
    *((int(i), int((i + 1) % 6)) for i in RING),
    (0, 6), (2, 7), (4, 8),
    (6, 9), (6, 10), (6, 11),
    (1, 12), (3, 13), (5, 14),
    (12, 15), (12, 16), (13, 17), (13, 18), (14, 19), (14, 20),
)
BOND_KIND = tuple(
    ["aromatic_cc"] * 6
    + ["aryl_substituent", "aryl_ch", "aryl_ch"]
    + ["methyl_ch"] * 3
    + ["cno2", "cno2", "cno2"]
    + ["no"] * 6
)
BREAKING_BOND_INDEX = BONDS.index((BREAKING_C, BREAKING_N))
VALENCE = np.asarray([3.0] * 6 + [4.0] + [1.0] * 5 + [3.0] * 3 + [1.0] * 6)
CHI = np.asarray([5.5] * 6 + [5.0] + [2.2] * 5 + [7.0] * 3 + [8.0] * 6)
ETA = np.asarray([10.0] * 6 + [10.0] + [12.0] * 5 + [12.0] * 3 + [13.0] * 6)
DE = {
    "aromatic_cc": 5.0,
    "aryl_substituent": 4.0,
    "aryl_ch": 4.0,
    "methyl_ch": 4.5,
    "cno2": 1.35,
    "no": 4.5,
}
TARGET_BO = {
    "aromatic_cc": 1.5,
    "aryl_substituent": 1.0,
    "aryl_ch": 1.0,
    "methyl_ch": 1.0,
    "cno2": 1.0,
    "no": 1.5,
}
KIND_WIDTH = {
    "aromatic_cc": 2.4,
    "aryl_substituent": 2.0,
    "aryl_ch": 2.0,
    "methyl_ch": 2.2,
    "cno2": 3.2,
    "no": 2.2,
}
COULOMB_EV_ANG = 14.399645
GAMMA_SHIELD = 0.45
K_OVER = 0.65
K_ANGLE = 1.2
REP_A = 80.0
REP_RHO = 0.28
# The schematic energy is not a fitted TNT force field.  A weak reference
# restraint keeps spectator atoms (especially light H atoms) near the optimized
# scaffold so the only visible event is the requested C2--NO2 departure.
SPECTATOR_ANCHOR_K = 8.0
SPECTATOR_INDICES = np.asarray([i for i in range(len(ELEMENTS)) if i not in {BREAKING_N, 15, 16}], dtype=int)


def reference_distances() -> np.ndarray:
    positions = initial_geometry()
    return np.asarray([np.linalg.norm(positions[j] - positions[i]) for i, j in BONDS], dtype=float)


R0 = reference_distances()
_REFERENCE_FORCE: np.ndarray | None = None
_REFERENCE_POSITIONS: np.ndarray | None = None


def bond_orders(positions: torch.Tensor) -> torch.Tensor:
    values = []
    for index, ((i, j), kind) in enumerate(zip(BONDS, BOND_KIND)):
        distance = torch.linalg.norm(positions[j] - positions[i])
        dr = distance - float(R0[index])
        width = KIND_WIDTH[kind]
        # The selected C2--N2 bond fades faster under the fixed kick while all
        # other bond types remain close to their reference order.
        values.append(float(TARGET_BO[kind]) * torch.exp(-width * torch.relu(dr)))
    return torch.stack(values)


def coordination(bo: torch.Tensor) -> torch.Tensor:
    values = torch.zeros(len(ELEMENTS), dtype=bo.dtype)
    for order, (i, j) in zip(bo, BONDS):
        values[i] += order
        values[j] += order
    return values - torch.as_tensor(VALENCE, dtype=bo.dtype)


def eem_charges(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = positions.shape[0]
    chi = torch.as_tensor(CHI, dtype=positions.dtype)
    eta = torch.as_tensor(ETA, dtype=positions.dtype)
    delta = positions[:, None, :] - positions[None, :, :]
    distance = torch.sqrt((delta**2).sum(-1) + torch.eye(n, dtype=positions.dtype))
    tap = COULOMB_EV_ANG / (distance**3 + GAMMA_SHIELD**-3) ** (1.0 / 3.0)
    tap = tap * (1.0 - torch.eye(n, dtype=positions.dtype))
    matrix = torch.zeros((n + 1, n + 1), dtype=positions.dtype)
    matrix[:n, :n] = tap + torch.diag(eta)
    matrix[:n, n] = -1.0
    matrix[n, :n] = 1.0
    rhs = torch.zeros(n + 1, dtype=positions.dtype)
    rhs[:n] = -chi
    solution = torch.linalg.solve(matrix, rhs)
    return solution[:n], tap


def energy_terms(positions: torch.Tensor) -> dict[str, torch.Tensor]:
    bo = bond_orders(positions)
    delta = coordination(bo)
    e_bond = torch.zeros((), dtype=positions.dtype)
    for order, kind in zip(bo, BOND_KIND):
        e_bond = e_bond - float(DE[kind]) * order
    e_over = K_OVER * (delta**2 * torch.sigmoid(3.0 * delta)).sum()
    q, tap = eem_charges(positions)
    chi = torch.as_tensor(CHI, dtype=positions.dtype)
    eta = torch.as_tensor(ETA, dtype=positions.dtype)
    e_coul = (chi * q + 0.5 * eta * q**2).sum() + 0.5 * (q[:, None] * q[None, :] * tap).sum()
    e_angle = torch.zeros((), dtype=positions.dtype)
    for i in range(6):
        a, b, c = (i - 1) % 6, i, (i + 1) % 6
        u = positions[a] - positions[b]
        v = positions[c] - positions[b]
        cos = (u @ v) / (torch.linalg.norm(u) * torch.linalg.norm(v))
        theta = torch.arccos(torch.clamp(cos, -1.0 + 1e-12, 1.0 - 1e-12))
        theta0 = float(np.deg2rad(120.0))
        e_angle = e_angle + K_ANGLE * (theta - theta0) ** 2
    e_rep = torch.zeros((), dtype=positions.dtype)
    for i in range(len(ELEMENTS)):
        for j in range(i + 1, len(ELEMENTS)):
            distance = torch.linalg.norm(positions[j] - positions[i])
            e_rep = e_rep + REP_A * torch.exp(-distance / REP_RHO)
    reference = torch.as_tensor(initial_geometry(), dtype=positions.dtype)
    spectator = torch.as_tensor(SPECTATOR_INDICES, dtype=torch.long)
    displacement = positions[spectator] - reference[spectator]
    e_anchor = 0.5 * SPECTATOR_ANCHOR_K * (displacement**2).sum()
    total = e_bond + e_over + e_angle + e_coul + e_rep
    total = total + e_anchor
    return {
        "total": total,
        "bond": e_bond,
        "over": e_over,
        "angle": e_angle,
        "coulomb": e_coul,
        "repulsion": e_rep,
        "anchor": e_anchor,
        "bo": bo,
        "delta": delta,
        "q": q,
    }


def evaluate(positions: np.ndarray) -> dict[str, np.ndarray | float]:
    global _REFERENCE_FORCE, _REFERENCE_POSITIONS
    positions = np.asarray(positions, dtype=float)
    tensor = torch.as_tensor(positions, dtype=torch.float64).clone().requires_grad_(True)
    terms = energy_terms(tensor)
    raw_energy = float(terms["total"].detach())
    raw_forces = -torch.autograd.grad(terms["total"], tensor)[0].detach().numpy()
    if _REFERENCE_FORCE is None:
        _REFERENCE_POSITIONS = initial_geometry().copy()
        _REFERENCE_FORCE = raw_forces.copy()
    corrected_energy = raw_energy + float(np.sum(_REFERENCE_FORCE * (positions - _REFERENCE_POSITIONS)))
    corrected_forces = raw_forces - _REFERENCE_FORCE
    return {
        "energy": corrected_energy,
        "forces": corrected_forces,
        "bond": float(terms["bond"].detach()),
        "over": float(terms["over"].detach()),
        "angle": float(terms["angle"].detach()),
        "coulomb": float(terms["coulomb"].detach()),
        "repulsion": float(terms["repulsion"].detach()),
        "bo": terms["bo"].detach().numpy(),
        "delta": terms["delta"].detach().numpy(),
        "q": terms["q"].detach().numpy(),
    }


def finite_difference_error(positions: np.ndarray, h: float = 1.0e-5) -> float:
    reference = evaluate(positions)["forces"]
    numeric = np.zeros_like(reference)
    for i in range(len(ELEMENTS)):
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[i, axis] += h
            minus[i, axis] -= h
            numeric[i, axis] = -(evaluate(plus)["energy"] - evaluate(minus)["energy"]) / (2.0 * h)
    return float(np.max(np.abs(numeric - reference)))


def run(speed: float, steps: int) -> dict[str, np.ndarray]:
    positions = initial_geometry()
    velocities = kick_velocities(positions, speed)
    frames: dict[str, list] = {key: [] for key in ("positions", "velocities", "bo", "delta", "q", "forces")}
    frames.update({key: [] for key in ("energy", "bond", "over", "angle", "coulomb", "repulsion")})
    result = evaluate(positions)
    for _ in range(int(steps) + 1):
        frames["positions"].append(positions.copy())
        frames["velocities"].append(velocities.copy())
        for key in ("bo", "delta", "q", "forces", "energy", "bond", "over", "angle", "coulomb", "repulsion"):
            frames[key].append(np.asarray(result[key]).copy())
        acceleration = result["forces"] * CONV_ACCEL / MASSES[:, None]
        half_velocity = velocities + 0.5 * acceleration * DT_FS
        positions = positions + half_velocity * DT_FS
        result = evaluate(positions)
        acceleration_next = result["forces"] * CONV_ACCEL / MASSES[:, None]
        velocities = half_velocity + 0.5 * acceleration_next * DT_FS
    return {key: np.asarray(value) for key, value in frames.items()} | {"positions": np.asarray(frames["positions"]), "velocities": np.asarray(frames["velocities"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=RELATIVE_SPEED)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    output = DATA_DIR / f"{STEM}.npz"
    if output.exists() and not args.force and not args.dry:
        print(f"[INFO] {output} exists; use --force")
        return
    start = evaluate(initial_geometry())
    fd = finite_difference_error(initial_geometry() + 0.02 * np.sin(np.arange(len(ELEMENTS) * 3).reshape(len(ELEMENTS), 3)))
    frames = run(args.speed, args.steps)
    total_energy = frames["energy"] + 0.5 * (MASSES[None, :, None] * frames["velocities"] ** 2).sum(axis=(1, 2)) / CONV_ACCEL
    r_cn = np.linalg.norm(frames["positions"][:, BREAKING_N] - frames["positions"][:, BREAKING_C], axis=1)
    print("start max|F|", np.abs(start["forces"]).max(), "FD", fd)
    print("final C2-N2", r_cn[-1], "BO", frames["bo"][-1, BREAKING_BOND_INDEX])
    print("total energy drift", float(total_energy.max() - total_energy.min()))
    if args.dry:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, elements=ELEMENTS, masses=MASSES, bonds=np.asarray(BONDS), r_cn=r_cn, dt_fs=DT_FS, total_energy_ev=total_energy, **frames)
    manifest = {
        "schema_version": 1,
        "stem": STEM,
        "method": "schematic ReaxFF-style energy (bond order, over-coordination, EEM charges, core repulsion)",
        "published_parameter_set": False,
        "lammps_run": False,
        "reaction": "TNT -> aryl radical + NO2 radical (C2-NO2 bond order fades)",
        "geometry_source": "generate_uks_tnt.initial_geometry()",
        "breaking_bond": [BREAKING_C, BREAKING_N],
        "kick": {"relative_speed_ang_fs": args.speed, "function": "generate_uks_tnt.kick_velocities"},
        "dt_fs": DT_FS,
        "n_steps": int(args.steps),
        "integrator": "velocity Verlet with analytic autograd forces",
        "spectator_restraint": {
            "type": "harmonic reference restraint in hidden schematic term",
            "k_ev_ang2": SPECTATOR_ANCHOR_K,
            "excluded_atoms": [BREAKING_N, 15, 16],
        },
        "checks": {
            "start_max_force_ev_ang": float(np.abs(start["forces"]).max()),
            "force_finite_difference_max_error": fd,
            "total_energy_drift_ev": float(total_energy.max() - total_energy.min()),
            "r_cn_final_ang": float(r_cn[-1]),
            "bo_cn_final": float(frames["bo"][-1, BREAKING_BOND_INDEX]),
        },
        "npz": str(output),
        "source_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (DATA_DIR / f"{STEM}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
