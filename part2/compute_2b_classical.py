"""
compute_2b_classical.py — Classical FF MD compute script for H₂O
=================================================================
Runs a SPC/E-like harmonic force field MD trajectory and saves the
result to classical_data.npz (ready for render_2b_classical.py).

Usage:
    python part2/compute_2b_classical.py           # run and save
    python part2/compute_2b_classical.py --force   # re-run even if npz exists
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from engine_md import kinetic_energy, CONV_ACCEL

# ══════════════════════════════════════════════════════════════════════════════
# Force Field Parameters (SPC/E-like, eV / Å / amu / fs)
# ══════════════════════════════════════════════════════════════════════════════
# Softened force constants for visualization stability at dt=0.5 fs, T=150 K
# Real SPC/E: K_bond=45.93 eV/Å², but that requires dt<0.05 fs for thermal velocities
# Visualization values: ~10x softer, same equilibrium geometry
K_BOND      = 4.5                      # eV/Å²  (softened for dt=0.5 fs stability)
R0_OH       = 1.012                    # Å
K_ANGLE     = 0.33                     # eV/rad²  (softened proportionally)
THETA0      = np.radians(113.24)       # rad
M_O, M_H   = 15.9994, 1.008
MASSES      = np.array([M_O, M_H, M_H])

DT_FS         = 1.0    # fs — displayed timestep (each animation frame = 1 fs)
DT_INNER      = 0.02   # fs — inner integration timestep for stability
N_INNER       = 50     # DT_FS / DT_INNER = 50 sub-steps per displayed frame
N_STEPS       = 20     # 20 × 1 fs = 20 fs trajectory

NPZ_PATH = os.path.join(os.path.dirname(__file__), "classical_data.npz")


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers  (SPC/E-like force field — distinct from engine_pes.py)
# ══════════════════════════════════════════════════════════════════════════════

def initial_geometry():
    r, theta = R0_OH, THETA0
    O  = np.array([0.0, 0.0, 0.0])
    H1 = np.array([ r * np.sin(theta / 2), 0.0,  r * np.cos(theta / 2)])
    H2 = np.array([-r * np.sin(theta / 2), 0.0,  r * np.cos(theta / 2)])
    return np.array([O, H1, H2])


def get_geometry(pos):
    r_OH1 = pos[1] - pos[0]
    r_OH2 = pos[2] - pos[0]
    r1 = np.linalg.norm(r_OH1)
    r2 = np.linalg.norm(r_OH2)
    cos_t = np.clip(np.dot(r_OH1, r_OH2) / (r1 * r2), -1.0, 1.0)
    theta = np.arccos(cos_t)
    return r1, r2, theta, r_OH1, r_OH2


def potential_energy(pos):
    r1, r2, theta, _, _ = get_geometry(pos)
    Vb1 = 0.5 * K_BOND  * (r1 - R0_OH)**2
    Vb2 = 0.5 * K_BOND  * (r2 - R0_OH)**2
    Va  = 0.5 * K_ANGLE * (theta - THETA0)**2
    return Vb1 + Vb2 + Va, Vb1 + Vb2, Va


def compute_forces(pos):
    r1, r2, theta, r_OH1, r_OH2 = get_geometry(pos)
    forces  = np.zeros((3, 3))
    f_bond  = np.zeros((3, 3))
    f_angle = np.zeros((3, 3))

    # Bond forces
    for idx, (r, r_vec) in enumerate([(r1, r_OH1), (r2, r_OH2)]):
        dV = K_BOND * (r - R0_OH)
        r_hat = r_vec / r
        f_bond[idx + 1] -= dV * r_hat
        f_bond[0]       += dV * r_hat

    # Angle force
    dV_dt = K_ANGLE * (theta - THETA0)
    sin_t = np.sin(theta)
    if abs(sin_t) > 1e-8:
        cos_t = np.cos(theta)
        r_hat1 = r_OH1 / r1
        r_hat2 = r_OH2 / r2
        dt_drH1 = (r_hat2 - cos_t * r_hat1) / (r1 * sin_t)
        dt_drH2 = (r_hat1 - cos_t * r_hat2) / (r2 * sin_t)
        dt_drO  = -(dt_drH1 + dt_drH2)
        f_angle[1] -= dV_dt * dt_drH1
        f_angle[2] -= dV_dt * dt_drH2
        f_angle[0] -= dV_dt * dt_drO

    forces = f_bond + f_angle
    return forces, f_bond, f_angle


def _vv_step_inner(pos, vel, forces, masses, dt):
    """Pure Velocity Verlet inner step — no thermostat or projection inside."""
    accel   = forces * CONV_ACCEL / masses[:, None]
    v_half  = vel + 0.5 * accel * dt
    new_pos = pos + v_half * dt
    new_forces, new_fb, new_fa = compute_forces(new_pos)
    new_accel = new_forces * CONV_ACCEL / masses[:, None]
    new_vel   = v_half + 0.5 * new_accel * dt
    return new_pos, new_vel, new_forces, new_fb, new_fa


# ══════════════════════════════════════════════════════════════════════════════
# Run MD
# ══════════════════════════════════════════════════════════════════════════════

def run_md():
    print("=" * 60)
    print("Classical FF MD  (SPC/E-like harmonic bond + angle)")
    print(f"  K_bond={K_BOND:.3f} eV/Å²  r0={R0_OH} Å")
    print(f"  K_angle={K_ANGLE:.4f} eV/rad²  θ0={np.degrees(THETA0):.2f}°")
    print(f"  dt={DT_FS} fs  N={N_STEPS}")
    print("=" * 60)

    pos = initial_geometry()
    # Asymmetric O-H stretch to seed multi-mode vibrational motion.
    # Amplitude chosen so PE ~ kT at 150 K (kT ≈ 0.013 eV):
    #   PE = 0.5*K_BOND*dr^2 = 0.013 eV  →  dr = sqrt(0.013/4.5) ≈ 0.054 Å
    d1 = pos[1] - pos[0]; d1 /= np.linalg.norm(d1)
    d2 = pos[2] - pos[0]; d2 /= np.linalg.norm(d2)
    pos[1] += 0.05 * d1   # stretch H1 by ~0.05 Å
    pos[2] -= 0.03 * d2   # compress H2 by ~0.03 Å
    # Zero initial velocities — pure NVE microcanonical dynamics.
    vel = np.zeros((3, 3))
    forces, f_bond, f_angle = compute_forces(pos)

    traj = []
    for step in range(N_STEPS):
        r1, r2, theta, _, _ = get_geometry(pos)
        V_tot, V_bond, V_angle = potential_energy(pos)
        KE = kinetic_energy(vel, MASSES)
        traj.append({
            "step": step,
            "time_fs": step * DT_FS,
            "positions": pos.copy(),
            "velocities": vel.copy(),
            "forces": forces.copy(),
            "f_bond": f_bond.copy(),
            "f_angle": f_angle.copy(),
            "r1": r1, "r2": r2,
            "theta_deg": np.degrees(theta),
            "V_bond": V_bond, "V_angle": V_angle, "V_total": V_tot,
            "KE": KE, "TE": KE + V_tot,
        })
        # Sub-stepping: N_INNER inner VV steps of DT_INNER per displayed frame
        for _ in range(N_INNER):
            pos, vel, forces, f_bond, f_angle = _vv_step_inner(
                pos, vel, forces, MASSES, DT_INNER)
        # Remove CM drift once per displayed frame (keeps molecule centred)
        vcm = (MASSES[:, None] * vel).sum(0) / MASSES.sum()
        vel -= vcm

    print(f"Done. Final KE={traj[-1]['KE']:.4f} eV  PE={traj[-1]['V_total']:.4f} eV")
    return traj


# ══════════════════════════════════════════════════════════════════════════════
# Save to NPZ
# ══════════════════════════════════════════════════════════════════════════════

def save_data(traj, path=NPZ_PATH):
    n = len(traj)
    np.savez(
        path,
        step       = np.array([d["step"]      for d in traj], dtype=int),
        time_fs    = np.array([d["time_fs"]   for d in traj]),
        positions  = np.array([d["positions"] for d in traj]),
        velocities = np.array([d["velocities"]for d in traj]),
        forces     = np.array([d["forces"]    for d in traj]),
        f_bond     = np.array([d["f_bond"]    for d in traj]),
        f_angle    = np.array([d["f_angle"]   for d in traj]),
        r1         = np.array([d["r1"]        for d in traj]),
        r2         = np.array([d["r2"]        for d in traj]),
        theta_deg  = np.array([d["theta_deg"] for d in traj]),
        V_bond     = np.array([d["V_bond"]    for d in traj]),
        V_angle    = np.array([d["V_angle"]   for d in traj]),
        V_total    = np.array([d["V_total"]   for d in traj]),
        KE         = np.array([d["KE"]        for d in traj]),
        TE         = np.array([d["TE"]        for d in traj]),
    )
    print(f"Saved {n} steps → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run classical FF MD for H₂O and save classical_data.npz"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if classical_data.npz already exists"
    )
    args = parser.parse_args()

    if os.path.exists(NPZ_PATH) and not args.force:
        print(f"[INFO] {NPZ_PATH} already exists.  Use --force to re-run.")
        sys.exit(0)

    traj = run_md()
    save_data(traj)
