"""Generate the 02b schematic ReaxFF HClO4 -> OH + ClO3 trajectory.

This is a *teaching* implementation of the ReaxFF energy decomposition on the
same six atoms, starting geometry and initial kick as 03b.  The functional
forms follow the published ReaxFF structure:

* bond order from interatomic distance (sigma + pi exponentials),
* bond energy  E = -De BO exp[p_be1 (1 - BO^p_be2)],
* over-coordination penalty from Delta_i = sum_j BO_ij - Val_i,
* valence-angle energy weighted by the two bond orders,
* electronegativity-equalisation (EEM) charges with shielded Coulomb energy,
* an exponential core repulsion for every pair.

The parameter values are chosen for a clear schematic (the initial geometry is
near-stationary (small residual forces) and the weak Cl-OH bond breaks under the 03b kick).  They are
not a published ReaxFF force field and no LAMMPS run is involved.  Forces are
exact gradients (torch autograd, float64) and are verified against central
finite differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "product" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run_md"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_md import CONV_ACCEL  # noqa: E402
from generate_uks_hclo4 import (  # noqa: E402
    CLO3_ATOMS,
    ELEMENTS,
    MASSES,
    OH_ATOMS,
    initial_geometry,
    kick_velocities,
)

STEM = "02b_reaxff"
DT_FS = 0.10
N_STEPS = 500
RELATIVE_SPEED = 0.05
COULOMB_EV_ANG = 14.399645

BONDS = ((0, 1), (0, 3), (0, 4), (0, 5), (1, 2))  # Cl-OH, 3 x Cl=O, O-H
BOND_KIND = ("ClOH", "ClO", "ClO", "ClO", "OH")
ANGLES = ((3, 0, 4), (3, 0, 5), (4, 0, 5), (3, 0, 1), (4, 0, 1), (5, 0, 1), (0, 1, 2))
VALENCE = np.asarray([6.0, 2.0, 1.0, 2.0, 2.0, 2.0])
CHI = np.asarray([7.0, 8.6, 5.6, 8.6, 8.6, 8.6])  # eV
ETA = np.asarray([10.0, 12.5, 13.0, 12.5, 12.5, 12.5])  # eV
GAMMA_SHIELD = 0.45  # 1/Angstrom

P_SIGMA = (-0.12, 8.0)
P_PI = (-0.30, 8.0)
PBE1, PBE2 = 0.35, 1.0
K_OVER, LAMBDA_OVER = 1.2, 4.0
K_ANGLE = 2.4  # eV / rad^2
REP_A, REP_RHO = 160.0, 0.28

# BO targets that fix the reference radii at the starting geometry
TARGET_SIGMA = {"ClOH": 0.95, "ClO": 0.96, "OH": 0.97}
TARGET_PI = {"ClO": 0.70}


def _initial_distances() -> dict[str, float]:
    pos = initial_geometry()
    return {
        kind: float(np.linalg.norm(pos[BONDS[BOND_KIND.index(kind)][1]] - pos[BONDS[BOND_KIND.index(kind)][0]]))
        for kind in ("ClOH", "ClO", "OH")
    }


def reference_radii() -> dict[str, float]:
    """Radii r0 such that exp[p1 (r/r0)^p2] equals the target BO at the start."""
    dist = _initial_distances()
    out = {}
    for kind, target in TARGET_SIGMA.items():
        p1, p2 = P_SIGMA
        out[f"sigma_{kind}"] = dist[kind] / (np.log(target) / p1) ** (1.0 / p2)
    p3, p4 = P_PI
    out["pi_ClO"] = dist["ClO"] / (np.log(TARGET_PI["ClO"]) / p3) ** (1.0 / p4)
    return out


RADII = reference_radii()


def bond_orders(positions: torch.Tensor) -> torch.Tensor:
    values = []
    for (i, j), kind in zip(BONDS, BOND_KIND):
        r = torch.linalg.norm(positions[j] - positions[i])
        bo = torch.exp(P_SIGMA[0] * (r / RADII[f"sigma_{kind}"]) ** P_SIGMA[1])
        if kind == "ClO":
            bo = bo + torch.exp(P_PI[0] * (r / RADII["pi_ClO"]) ** P_PI[1])
        values.append(bo)
    return torch.stack(values)


def coordination(bo: torch.Tensor) -> torch.Tensor:
    total = torch.zeros(6, dtype=bo.dtype)
    for value, (i, j) in zip(bo, BONDS):
        total = total.index_add(0, torch.tensor([i, j]), torch.stack([value, value]))
    return total - torch.as_tensor(VALENCE, dtype=bo.dtype)


def angles_of(positions: torch.Tensor) -> torch.Tensor:
    values = []
    for a, b, c in ANGLES:
        u = positions[a] - positions[b]
        w = positions[c] - positions[b]
        cos = (u @ w) / (torch.linalg.norm(u) * torch.linalg.norm(w))
        values.append(torch.arccos(torch.clamp(cos, -1.0 + 1e-12, 1.0 - 1e-12)))
    return torch.stack(values)


def eem_charges(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = positions.shape[0]
    chi = torch.as_tensor(CHI, dtype=positions.dtype)
    eta = torch.as_tensor(ETA, dtype=positions.dtype)
    diff = positions[:, None, :] - positions[None, :, :]
    r = torch.sqrt((diff**2).sum(-1) + torch.eye(n, dtype=positions.dtype))
    tap = COULOMB_EV_ANG / (r**3 + GAMMA_SHIELD**-3) ** (1.0 / 3.0)
    tap = tap * (1.0 - torch.eye(n, dtype=positions.dtype))
    matrix = torch.zeros(n + 1, n + 1, dtype=positions.dtype)
    matrix[:n, :n] = tap + torch.diag(eta)
    if not torch.is_grad_enabled() or True:
        pass
    matrix[:n, n] = -1.0
    matrix[n, :n] = 1.0
    rhs = torch.zeros(n + 1, dtype=positions.dtype)
    rhs[:n] = -chi
    solution = torch.linalg.solve(matrix, rhs)
    return solution[:n], tap


def energy_terms(positions: torch.Tensor, de: dict[str, float]) -> dict[str, torch.Tensor]:
    bo = bond_orders(positions)
    delta = coordination(bo)
    e_bond = torch.zeros((), dtype=positions.dtype)
    for value, kind in zip(bo, BOND_KIND):
        e_bond = e_bond - de[kind] * value * torch.exp(PBE1 * (1.0 - value**PBE2))
    e_over = (K_OVER * delta**2 * torch.sigmoid(LAMBDA_OVER * delta)).sum()
    theta = angles_of(positions)
    bo_map = {pair: value for pair, value in zip(BONDS, bo)}
    e_angle = torch.zeros((), dtype=positions.dtype)
    for (a, b, c), t in zip(ANGLES, theta):
        f1 = 1.0 - torch.exp(-bo_map[tuple(sorted((a, b)))] ** 3)
        f2 = 1.0 - torch.exp(-bo_map[tuple(sorted((b, c)))] ** 3)
        e_angle = e_angle + K_ANGLE * f1 * f2 * (t - THETA0[(a, b, c)]) ** 2
    q, tap = eem_charges(positions)
    chi = torch.as_tensor(CHI, dtype=positions.dtype)
    eta = torch.as_tensor(ETA, dtype=positions.dtype)
    e_coul = (chi * q + 0.5 * eta * q**2).sum() + 0.5 * (q[:, None] * q[None, :] * tap).sum()
    n = positions.shape[0]
    e_rep = torch.zeros((), dtype=positions.dtype)
    bonded = {pair: kind for pair, kind in zip(BONDS, BOND_KIND)}
    for i in range(n):
        for j in range(i + 1, n):
            amp = de[f"A_{bonded[(i, j)]}"] if (i, j) in bonded else REP_A
            e_rep = e_rep + amp * torch.exp(-torch.linalg.norm(positions[j] - positions[i]) / REP_RHO)
    total = e_bond + e_over + e_angle + e_coul + e_rep
    return {
        "total": total,
        "bond": e_bond,
        "over": e_over,
        "angle": e_angle,
        "coulomb": e_coul,
        "repulsion": e_rep,
        "bo": bo,
        "delta": delta,
        "theta": theta,
        "q": q,
    }


def _angle_np(pos: np.ndarray, a: int, b: int, c: int) -> float:
    u, w = pos[a] - pos[b], pos[c] - pos[b]
    return float(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w)), -1.0, 1.0)))


THETA0 = {abc: _angle_np(initial_geometry(), *abc) for abc in ANGLES}
DE = {"ClOH": 1.4, "ClO": 4.0, "OH": 4.5}  # eV, fixed by design (weak Cl-OH)


def fit_bond_energies() -> dict[str, float]:
    """Fix De per bond type; solve the bonded core-repulsion amplitudes so the
    starting geometry is a stationary point of the schematic energy."""
    pos0 = torch.as_tensor(initial_geometry(), dtype=torch.float64)

    def parameters(amps: np.ndarray) -> dict[str, float]:
        return {**DE, "A_ClOH": amps[0], "A_ClO": amps[1], "A_OH": amps[2]}

    def residual(amps: np.ndarray) -> np.ndarray:
        p = pos0.clone().requires_grad_(True)
        total = energy_terms(p, parameters(amps))["total"]
        return torch.autograd.grad(total, p)[0].detach().numpy().reshape(-1)

    solution = least_squares(
        residual, x0=np.asarray([20.0, 40.0, 40.0]), bounds=(0.0, 5000.0), xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    return parameters(solution.x)


def evaluate(positions: np.ndarray, de: dict[str, float]) -> dict[str, np.ndarray | float]:
    p = torch.as_tensor(positions, dtype=torch.float64).clone().requires_grad_(True)
    terms = energy_terms(p, de)
    grad = torch.autograd.grad(terms["total"], p)[0].detach().numpy()
    return {
        "energy": float(terms["total"].detach()),
        "forces": -grad,
        "bond": float(terms["bond"].detach()),
        "over": float(terms["over"].detach()),
        "angle": float(terms["angle"].detach()),
        "coulomb": float(terms["coulomb"].detach()),
        "repulsion": float(terms["repulsion"].detach()),
        "bo": terms["bo"].detach().numpy(),
        "delta": terms["delta"].detach().numpy(),
        "theta": terms["theta"].detach().numpy(),
        "q": terms["q"].detach().numpy(),
    }


def finite_difference_error(positions: np.ndarray, de: dict[str, float], h: float = 1.0e-5) -> float:
    reference = evaluate(positions, de)["forces"]
    numeric = np.zeros_like(reference)
    for i in range(positions.shape[0]):
        for k in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[i, k] += h
            minus[i, k] -= h
            numeric[i, k] = -(evaluate(plus, de)["energy"] - evaluate(minus, de)["energy"]) / (2.0 * h)
    return float(np.max(np.abs(numeric - reference)))


def run(de: dict[str, float], speed: float, n_steps: int, dt: float) -> dict[str, np.ndarray]:
    positions = initial_geometry()
    velocities = kick_velocities(positions, speed)
    keys = ("bo", "delta", "theta", "q", "forces")
    scalars = ("energy", "bond", "over", "angle", "coulomb", "repulsion")
    frames = {k: [] for k in (*keys, *scalars, "positions", "velocities")}

    def store(pos, vel, res):
        frames["positions"].append(pos.copy())
        frames["velocities"].append(vel.copy())
        for k in keys:
            frames[k].append(np.asarray(res[k]))
        for k in scalars:
            frames[k].append(res[k])

    result = evaluate(positions, de)
    store(positions, velocities, result)
    for _ in range(n_steps):
        accel = result["forces"] * CONV_ACCEL / MASSES[:, None]
        v_half = velocities + 0.5 * accel * dt
        positions = positions + v_half * dt
        result = evaluate(positions, de)
        accel_next = result["forces"] * CONV_ACCEL / MASSES[:, None]
        velocities = v_half + 0.5 * accel_next * dt
        store(positions, velocities, result)
    return {k: np.asarray(v) for k, v in frames.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=RELATIVE_SPEED)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry", action="store_true", help="print diagnostics without saving")
    args = parser.parse_args()
    out = DATA_DIR / f"{STEM}.npz"
    if out.exists() and not args.force and not args.dry:
        print(f"[INFO] {out} exists; use --force")
        return
    de = fit_bond_energies()
    pos0 = initial_geometry()
    start = evaluate(pos0, de)
    fd = finite_difference_error(pos0 + 0.03 * np.sin(np.arange(18).reshape(6, 3)), de)
    print("De", de, "radii", {k: round(v, 3) for k, v in RADII.items()})
    print("start |F|max", np.abs(start["forces"]).max(), "BO", start["bo"].round(3), "delta", start["delta"].round(3))
    print("q", start["q"].round(3), "FD err", fd)
    frames = run(de, args.speed, args.steps, DT_FS)
    r_clo = np.linalg.norm(frames["positions"][:, 1] - frames["positions"][:, 0], axis=1)
    total_energy = frames["energy"] + 0.5 * (MASSES[None, :, None] * frames["velocities"] ** 2).sum(axis=(1, 2)) / CONV_ACCEL
    print("r_clo", r_clo[::50].round(2), "BO ClOH", frames["bo"][::50, 0].round(2))
    print("E tot drift", float(total_energy.max() - total_energy.min()), "max|F|", np.abs(frames["forces"]).max())
    print("E pot", frames["energy"][::50].round(2))
    if args.dry:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        elements=ELEMENTS,
        masses=MASSES,
        bonds=np.asarray(BONDS),
        angles=np.asarray(ANGLES),
        valence=VALENCE,
        r_clo=r_clo,
        dt_fs=np.asarray(DT_FS),
        total_energy_ev=total_energy,
        **frames,
    )
    manifest = {
        "schema_version": 1,
        "stem": STEM,
        "method": "schematic ReaxFF-style energy (bond order, over-coordination, valence angle, EEM charges, core repulsion)",
        "published_parameter_set": False,
        "lammps_run": False,
        "reaction": "HClO4 -> OH radical + ClO3 radical (Cl-OH bond order fades)",
        "geometry_source": "03b generate_uks_hclo4.initial_geometry()",
        "kick": {"relative_speed_ang_fs": args.speed, "function": "generate_uks_hclo4.kick_velocities"},
        "dt_fs": DT_FS,
        "n_steps": int(args.steps),
        "integrator": "velocity Verlet with analytic autograd forces",
        "bond_energies_ev": de,
        "reference_radii_ang": RADII,
        "bond_order_parameters": {"sigma": P_SIGMA, "pi": P_PI, "p_be1": PBE1, "p_be2": PBE2},
        "valence": VALENCE.tolist(),
        "eem": {"chi_ev": CHI.tolist(), "eta_ev": ETA.tolist(), "gamma_inv_ang": GAMMA_SHIELD},
        "angle_force_constant_ev_rad2": K_ANGLE,
        "over_coordination": {"k_ev": K_OVER, "lambda": LAMBDA_OVER},
        "repulsion": {"A_ev": REP_A, "rho_ang": REP_RHO},
        "checks": {
            "start_max_force_ev_ang": float(np.abs(start["forces"]).max()),
            "force_finite_difference_max_error": fd,
            "total_energy_drift_ev": float(total_energy.max() - total_energy.min()),
            "r_clo_final_ang": float(r_clo[-1]),
            "bo_clo_h_final": float(frames["bo"][-1, 0]),
        },
        "npz": str(out),
        "source_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (DATA_DIR / f"{STEM}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
