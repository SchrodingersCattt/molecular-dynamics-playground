"""
worker_deepmd.py — Remote worker script for DeePMD-kit MD on Bohrium.
======================================================================
Runs inside Docker image: registry.dp.tech/dptech/dpmd:2.2.8-cuda12.0

Self-contained: no local imports from the project.  Reads the DeePMD model
from the current working directory (uploaded by dpdispatcher), runs a
Velocity Verlet MD trajectory, and saves results.npz.

Model: H2O-Phase-Diagram-model_compressed.pb
  Source: https://store.aissquare.com/models/4560428e-db9c-11ee-9b22-506b4b2349d8/H2O-Phase-Diagram-model_compressed.pb

Usage (inside container):
    python worker_deepmd.py [--model H2O-Phase-Diagram-model_compressed.pb] [--steps 20] [--dt 0.5]

Output:
    results.npz       — trajectory arrays for download by compute_2c_deepmd.py
    worker_crash.log  — written only if a fatal error occurs (for diagnosis)
"""

import argparse
import os
import sys
import subprocess
import traceback
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# Crash logger — writes to worker_crash.log so dpdispatcher can download it
# ══════════════════════════════════════════════════════════════════════════════

def _crash(msg: str, exc: Exception = None):
    """Write a fatal error to worker_crash.log and stderr, then exit(1)."""
    lines = [
        "=" * 60,
        "[worker_deepmd] FATAL ERROR",
        msg,
    ]
    if exc is not None:
        lines.append(traceback.format_exc())
    lines.append("=" * 60)
    text = "\n".join(lines) + "\n"
    print(text, file=sys.stderr, flush=True)
    try:
        with open("worker_crash.log", "w") as fh:
            fh.write(text)
    except Exception:
        pass
    sys.exit(1)


# ── Ensure ASE is available (install if missing) ───────────────────────────────
def _ensure_ase():
    try:
        import ase  # noqa: F401
        print("[worker] ASE already available.", flush=True)
    except ImportError:
        print("[worker] ASE not found — installing via pip...", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "ase", "--quiet"]
            )
            print("[worker] ASE installed.", flush=True)
        except Exception as e:
            _crash("pip install ase failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

ATOM_SYMBOLS = ["O", "H", "H"]
# deepmd-kit type map: O=0, H=1  (must match the model's type_map)
ATOM_TYPES   = np.array([0, 1, 1], dtype=int)
ATOM_MASSES  = np.array([15.9994, 1.008, 1.008])   # amu

# Equilibrium water geometry (Angstrom)
WATER_POS = np.array([
    [0.000,  0.000,  0.119],
    [0.000,  0.757, -0.477],
    [0.000, -0.757, -0.477],
], dtype=float)

# MD parameters
TEMPERATURE_K = 300.0
KB_EV         = 8.617333e-5          # eV / K

# Unit conversions
# Force→acceleration: F [eV/Å] / m [amu] * CONV_ACCEL = a [Å/fs²]
#   1 eV/Å/amu = 1.60218e-19 J / (1e-10 m · 1.66054e-27 kg)
#              = 9.6485e17 m/s²  = 9.6485e-3 Å/fs²
CONV_ACCEL    = 0.009648   # (eV/Å)/amu → Å/fs²

# KE conversion: KE [eV] = 0.5 · Σ(m [amu] · v² [Å²/fs²]) · AMU_AFS2_TO_EV
#   1 amu·(Å/fs)² = 1.66054e-27 kg · (1e5 m/s)² / 1.60218e-19 J/eV = 103.6427 eV
# NOTE: CONV_ACCEL = 1/AMU_AFS2_TO_EV ≈ 0.009648 — they are reciprocals.
# Using CONV_ACCEL in the KE formula (as a multiplier) would give KE/103.64²,
# which is wrong by a factor of ~10,741.  Always use AMU_AFS2_TO_EV for KE.
AMU_AFS2_TO_EV = 103.6427  # eV per (amu · Å²/fs²)

# Descriptor cutoff (Å) — used for manual s_ij computation
R_CUT    = 6.0
R_INNER  = 0.5

# Large simulation box for non-periodic (isolated molecule) calculation.
# deepmd-kit 2.x requires an explicit cell even for non-periodic systems;
# passing cells=None raises a TypeError in some builds.
# A 50 Å box is large enough that periodic images never interact.
CELL_BOX = np.diag([50.0, 50.0, 50.0])   # shape (3,3), Å


# ══════════════════════════════════════════════════════════════════════════════
# Smooth cutoff  f_cut(r)
# ══════════════════════════════════════════════════════════════════════════════

def f_cut(r_arr, r_cut=R_CUT, r_inner=R_INNER):
    """
    Smooth cutoff function:
      1          for r < r_inner
      polynomial for r_inner ≤ r < r_cut
      0          for r ≥ r_cut
    """
    r = np.asarray(r_arr, dtype=float)
    out = np.zeros_like(r)
    m_in  = r < r_inner
    m_mid = (r >= r_inner) & (r < r_cut)
    out[m_in] = 1.0
    u = (r[m_mid] - r_inner) / (r_cut - r_inner)
    out[m_mid] = u**3 * (-6*u**2 + 15*u - 10) + 1.0
    return out


def compute_descriptors(positions):
    """
    For each atom i, compute s_ij = f_cut(r_ij) / r_ij for all neighbors j.

    Returns list of dicts (one per atom):
        neighbors : list[int]
        r_ij      : list[float]   distances (Å)
        s_ij      : list[float]   descriptor values (Å⁻¹)
    """
    result = []
    for i in range(len(positions)):
        nbrs, r_list, s_list = [], [], []
        for j in range(len(positions)):
            if j == i:
                continue
            r_vec = positions[j] - positions[i]
            r = float(np.linalg.norm(r_vec))
            if 1e-8 < r < R_CUT:
                fc = float(f_cut(np.array([r]))[0])
                nbrs.append(j)
                r_list.append(r)
                s_list.append(fc / r)
        result.append({"neighbors": nbrs, "r_ij": r_list, "s_ij": s_list})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MD helpers
# ══════════════════════════════════════════════════════════════════════════════

def maxwell_boltzmann_velocities(masses, T_K, seed=42):
    """Sample velocities from Maxwell-Boltzmann distribution (Å/fs).

    Equipartition: <0.5 · m · v_x²> · AMU_AFS2_TO_EV = 0.5 · kT
    → sigma_x = sqrt(kT / (m · AMU_AFS2_TO_EV))   [Å/fs]
    """
    rng = np.random.default_rng(seed)
    kT  = KB_EV * T_K
    vel = np.zeros((len(masses), 3))
    for i, m in enumerate(masses):
        sigma = np.sqrt(kT / (m * AMU_AFS2_TO_EV))
        vel[i] = rng.normal(0.0, sigma, 3)
    # Remove centre-of-mass drift
    v_cm = (masses[:, None] * vel).sum(0) / masses.sum()
    vel -= v_cm
    return vel


def kinetic_energy_ev(vel, masses):
    """KE in eV.  vel in Å/fs, masses in amu.

    KE [eV] = 0.5 · Σ(m [amu] · v² [Å²/fs²]) · AMU_AFS2_TO_EV
    """
    return 0.5 * float(np.sum(masses[:, None] * vel**2)) * AMU_AFS2_TO_EV


def eval_deepmd(dp, positions):
    """
    Call DeepPot.eval() and return (E_total, per_atom_e, forces).

    dp.eval signature (deepmd-kit >= 2.x):
        e, f, v = dp.eval(coords, cells, atom_types)
        e, f, v, ae, av = dp.eval(coords, cells, atom_types, atomic=True)

    coords : (1, natoms, 3)  Å
    cells  : (1, 9) or (3, 3) — explicit box required by deepmd-kit 2.x
    atom_types : (natoms,)  int
    """
    natoms = len(positions)
    coords = positions.reshape(1, natoms, 3)
    # Use explicit large box instead of None — deepmd-kit 2.x requires a cell
    cells  = CELL_BOX.reshape(1, 9)

    try:
        e, f, v, ae, av = dp.eval(
            coords, cells=cells, atom_types=ATOM_TYPES, atomic=True
        )
        per_atom_e = ae.reshape(natoms)          # (natoms,) eV
    except TypeError:
        # Older API without atomic keyword
        e, f, v = dp.eval(coords, cells=cells, atom_types=ATOM_TYPES)
        per_atom_e = np.full(natoms, float(e) / natoms)

    E_total = float(e.reshape(-1)[0])
    forces  = f.reshape(natoms, 3)               # (natoms, 3) eV/Å
    return E_total, per_atom_e, forces


def velocity_verlet_step(pos, vel, forces, masses, dt, dp):
    """One Velocity Verlet step.  Returns (new_pos, new_vel, new_forces, E, per_e)."""
    accel   = forces * CONV_ACCEL / masses[:, None]   # Å/fs²
    v_half  = vel + 0.5 * accel * dt
    new_pos = pos + v_half * dt
    E_new, per_e_new, new_forces = eval_deepmd(dp, new_pos)
    new_accel = new_forces * CONV_ACCEL / masses[:, None]
    new_vel   = v_half + 0.5 * new_accel * dt
    return new_pos, new_vel, new_forces, E_new, per_e_new


# ══════════════════════════════════════════════════════════════════════════════
# Main simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(model_path, n_steps, dt_fs):
    print("=" * 60, flush=True)
    print(f"DeePMD-kit worker  |  model={model_path}", flush=True)
    print(f"  n_steps={n_steps}  dt={dt_fs} fs  T={TEMPERATURE_K} K", flush=True)
    print("=" * 60, flush=True)

    # ── Diagnostic: print Python + environment info ────────────────────────
    print(f"[worker] Python: {sys.version}", flush=True)
    print(f"[worker] CWD: {os.getcwd()}", flush=True)
    print(f"[worker] Files in CWD: {os.listdir('.')}", flush=True)

    # ── Check model file exists ────────────────────────────────────────────
    if not os.path.exists(model_path):
        _crash(f"Model file not found: {model_path}\nCWD={os.getcwd()}\nFiles={os.listdir('.')}")

    model_size_mb = os.path.getsize(model_path) / 1e6
    print(f"[worker] Model file: {model_path}  ({model_size_mb:.1f} MB)", flush=True)

    # ── Import deepmd-kit (deferred to catch import errors clearly) ────────
    print("[worker] Importing deepmd.infer.DeepPot ...", flush=True)
    try:
        from deepmd.infer import DeepPot
        print("[worker] deepmd.infer.DeepPot imported OK", flush=True)
    except ImportError as e:
        # Try alternative import path used in some deepmd-kit builds
        print(f"[worker] deepmd.infer import failed ({e}), trying deepmd.calculator ...", flush=True)
        try:
            from deepmd.calculator import DP as DeepPot  # noqa: F401
            print("[worker] deepmd.calculator.DP imported as DeepPot OK", flush=True)
        except ImportError as e2:
            _crash(
                f"Cannot import DeepPot from deepmd.infer or deepmd.calculator.\n"
                f"Error 1: {e}\nError 2: {e2}\n"
                f"deepmd-kit may not be installed in this container.",
                e2,
            )

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[worker] Loading model: {model_path}", flush=True)
    try:
        dp = DeepPot(model_path)
        print("[worker] Model loaded OK", flush=True)
    except Exception as e:
        _crash(f"DeepPot({model_path!r}) failed", e)

    # ── Initial geometry — small perturbation + thermal velocities ──────────
    # 0.05 Å displacement is enough to break symmetry without blowing up.
    # The thermal velocities (now correctly ~0.027 Å/fs for H at 300 K) carry
    # the molecule through many oscillation cycles.
    rng = np.random.default_rng(13)
    pos = WATER_POS.copy() + rng.normal(0, 0.05, WATER_POS.shape)
    vel = maxwell_boltzmann_velocities(ATOM_MASSES, TEMPERATURE_K, seed=13)

    print("[worker] Running initial eval_deepmd ...", flush=True)
    try:
        E0, per_e0, forces = eval_deepmd(dp, pos)
        print(f"[worker] Initial E={E0:.6f} eV", flush=True)
    except Exception as e:
        _crash("eval_deepmd failed on initial geometry", e)

    # Storage arrays
    steps_arr    = np.zeros(n_steps, dtype=int)
    time_arr     = np.zeros(n_steps)
    pos_arr      = np.zeros((n_steps, 3, 3))
    vel_arr      = np.zeros((n_steps, 3, 3))
    forces_arr   = np.zeros((n_steps, 3, 3))
    E_arr        = np.zeros(n_steps)
    per_e_arr    = np.zeros((n_steps, 3))
    KE_arr       = np.zeros(n_steps)
    TE_arr       = np.zeros(n_steps)
    r_ij_arr     = np.zeros((n_steps, 3, 2))
    s_ij_arr     = np.zeros((n_steps, 3, 2))
    n_nbr_arr    = np.zeros((n_steps, 3), dtype=int)

    # forces_cur holds forces at the current position.
    # Initialised from the first eval_deepmd call above.
    forces_cur = forces

    for step in range(n_steps):
        KE = kinetic_energy_ev(vel, ATOM_MASSES)
        E_total, per_e = E0, per_e0

        # Descriptor
        desc = compute_descriptors(pos)
        for i in range(3):
            nn = len(desc[i]["r_ij"])
            n_nbr_arr[step, i] = nn
            for k in range(min(nn, 2)):
                r_ij_arr[step, i, k] = desc[i]["r_ij"][k]
                s_ij_arr[step, i, k] = desc[i]["s_ij"][k]

        steps_arr[step]  = step
        time_arr[step]   = step * dt_fs
        pos_arr[step]    = pos
        vel_arr[step]    = vel
        forces_arr[step] = forces_cur
        E_arr[step]      = E_total
        per_e_arr[step]  = per_e
        KE_arr[step]     = KE
        TE_arr[step]     = KE + E_total

        print(f"  Step {step+1:3d}/{n_steps}:  E={E_total:.6f} eV  "
              f"KE={KE:.6f} eV  TE={KE+E_total:.6f} eV", flush=True)

        # Advance: VV step returns forces at the NEW position, which become
        # forces_cur for the next iteration.
        pos, vel, forces_cur, E0, per_e0 = velocity_verlet_step(
            pos, vel, forces_cur, ATOM_MASSES, dt_fs, dp
        )

    # Save
    np.savez(
        "results.npz",
        steps    = steps_arr,
        time_fs  = time_arr,
        positions= pos_arr,
        velocities=vel_arr,
        forces   = forces_arr,
        E_total  = E_arr,
        per_atom_e=per_e_arr,
        KE       = KE_arr,
        TE       = TE_arr,
        r_ij     = r_ij_arr,
        s_ij     = s_ij_arr,
        n_nbr    = n_nbr_arr,
    )
    print("\n[worker] Saved results.npz", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Ensure ASE is available before anything else
    _ensure_ase()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="H2O-Phase-Diagram-model_compressed.pb")
    parser.add_argument("--steps",  type=int,   default=20)
    parser.add_argument("--dt",     type=float, default=0.5)
    args = parser.parse_args()

    try:
        run_simulation(args.model, args.steps, args.dt)
    except SystemExit:
        raise   # _crash() already wrote worker_crash.log
    except Exception as e:
        _crash("Unhandled exception in run_simulation", e)
