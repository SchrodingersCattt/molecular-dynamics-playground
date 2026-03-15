"""
compute_2a_rhf.py — RHF/STO-3G MD compute script for H₂O
==========================================================
Runs a 20-step RHF/STO-3G MD trajectory and saves the result to
md_data.npz (ready for render_2a_rhf.py to render).

Usage:
    python part2/compute_2a_rhf.py           # run and save (~2 min)
    python part2/compute_2a_rhf.py --force   # re-run even if npz exists
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from engine_rhf import (
    run_rhf_scf,
    density_on_grid,
    scf_density_history,
    HARTREE_TO_EV,
)
from engine_pes import (
    potential_energy  as pes_potential,
    forces            as pes_forces,
    get_geometry,
    R0_OH, THETA0,
)
from engine_md import (
    maxwell_boltzmann_velocities,
    kinetic_energy,
    velocity_verlet_step,
    CONV_ACCEL,
)

# ── MD parameters ──────────────────────────────────────────────────────────────
MASSES      = np.array([15.9994, 1.008, 1.008])
DT_FS       = 1.0      # fs
N_STEPS     = 20
TEMPERATURE = 300.0    # K
N_SCF_SHOW  = 4        # SCF iterations to record per step
GRID_SIZE   = 40

NPZ_PATH = os.path.join(os.path.dirname(__file__), "md_data.npz")

# Reference geometry (NIST)
WATER_EQ = np.array([
    [0.000,  0.000,  0.119],
    [0.000,  0.757, -0.477],
    [0.000, -0.757, -0.477],
])


# ══════════════════════════════════════════════════════════════════════════════
# MD runner
# ══════════════════════════════════════════════════════════════════════════════

def run_md():
    print("=" * 60)
    print("RHF/STO-3G MD  (H₂O, STO-3G basis)")
    print(f"  dt={DT_FS} fs  N={N_STEPS}  T={TEMPERATURE} K")
    print("=" * 60)

    pos = WATER_EQ.copy()
    vel = maxwell_boltzmann_velocities(MASSES, TEMPERATURE, seed=42)

    # Reference SCF energy at equilibrium
    E_ref_ha, _, _, _ = run_rhf_scf(WATER_EQ, max_iter=100, conv_tol=1e-8)
    E_ref_ev = E_ref_ha * HARTREE_TO_EV

    md_data = []

    for step in range(N_STEPS):
        print(f"  Step {step+1:3d}/{N_STEPS} ...", end=" ", flush=True)

        # Full SCF at current geometry
        E_ha, C, scf_energies, basis = run_rhf_scf(pos, max_iter=100, conv_tol=1e-8)
        E_ev = E_ha * HARTREE_TO_EV

        # Electron density on grid
        Y, Z, rho = density_on_grid(C, basis, pos, grid_size=GRID_SIZE)

        # SCF density history (first N_SCF_SHOW iterations)
        C_hist = scf_density_history(pos, N_SCF_SHOW)
        density_hist = []
        for C_i in C_hist:
            _, _, rho_i = density_on_grid(C_i, basis, pos, grid_size=GRID_SIZE)
            density_hist.append(rho_i)

        # Forces from PES (analytic)
        F = pes_forces(pos)
        PE_ev = pes_potential(pos)

        # Geometry
        r1, r2, r_HH, theta, _, _ = get_geometry(pos)
        KE = kinetic_energy(vel, MASSES)

        md_data.append({
            "step":         step,
            "time_fs":      step * DT_FS,
            "positions":    pos.copy(),
            "velocities":   vel.copy(),
            "forces":       F.copy(),
            "scf_energies": scf_energies,
            "density_grid": (Y, Z, rho),
            "density_hist": density_hist,
            "KE":           KE,
            "PE_ev":        PE_ev,
            "PE_ref_ev":    E_ev - E_ref_ev,
            "r1":           r1,
            "r2":           r2,
            "r_HH":         r_HH,
            "theta_deg":    np.degrees(theta),
        })

        print(f"E={E_ev:.4f} eV  KE={KE:.4f} eV  PE={PE_ev:.4f} eV")

        # Velocity Verlet step
        pos, vel, _ = velocity_verlet_step(pos, vel, pes_forces, MASSES, DT_FS)

    print(f"\nDone. {N_STEPS} steps completed.")
    return md_data


# ══════════════════════════════════════════════════════════════════════════════
# Save to NPZ
# ══════════════════════════════════════════════════════════════════════════════

def save_md_data(md_data, path=NPZ_PATH):
    n = len(md_data)
    # Pad scf_energies to fixed length
    max_scf = max(len(d["scf_energies"]) for d in md_data)
    scf_arr = np.zeros((n, max_scf))
    for i, d in enumerate(md_data):
        e = d["scf_energies"]
        scf_arr[i, :len(e)] = e

    # Pad density_hist to N_SCF_SHOW frames
    G = GRID_SIZE
    dh_arr = np.zeros((n, N_SCF_SHOW, G, G))
    for i, d in enumerate(md_data):
        for j, rho_j in enumerate(d["density_hist"]):
            dh_arr[i, j] = rho_j

    Y0, Z0, _ = md_data[0]["density_grid"]

    np.savez(
        path,
        step         = np.array([d["step"]       for d in md_data], dtype=int),
        time_fs      = np.array([d["time_fs"]     for d in md_data]),
        positions    = np.array([d["positions"]   for d in md_data]),
        velocities   = np.array([d["velocities"]  for d in md_data]),
        forces       = np.array([d["forces"]      for d in md_data]),
        scf_energies = scf_arr,
        density_Y    = np.array([d["density_grid"][0] for d in md_data]),
        density_Z    = np.array([d["density_grid"][1] for d in md_data]),
        density_rho  = np.array([d["density_grid"][2] for d in md_data]),
        density_hist = dh_arr,
        KE           = np.array([d["KE"]          for d in md_data]),
        PE_ev        = np.array([d["PE_ev"]        for d in md_data]),
        PE_ref_ev    = np.array([d["PE_ref_ev"]    for d in md_data]),
        r1           = np.array([d["r1"]           for d in md_data]),
        r2           = np.array([d["r2"]           for d in md_data]),
        r_HH         = np.array([d["r_HH"]         for d in md_data]),
        theta_deg    = np.array([d["theta_deg"]    for d in md_data]),
    )
    print(f"Saved {n} steps → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RHF/STO-3G MD for H₂O and save md_data.npz"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if md_data.npz already exists"
    )
    args = parser.parse_args()

    if os.path.exists(NPZ_PATH) and not args.force:
        print(f"[INFO] {NPZ_PATH} already exists.  Use --force to re-run.")
        sys.exit(0)

    md_data = run_md()
    save_md_data(md_data)
