"""
engine_md.py — Velocity Verlet MD integrator, Maxwell-Boltzmann sampling, KE
=============================================================================
Provides a self-contained, unit-consistent MD engine for H₂O simulations.

Unit system
-----------
  positions  : Å
  velocities : Å/fs
  forces     : eV/Å
  masses     : amu
  time       : fs
  energy     : eV
  temperature: K

Conversion factor
-----------------
  CONV_ACCEL = 9.6485e-3  (eV/Å)/amu → Å/fs²
  Derivation: 1 eV/Å/amu
            = 1.60218e-19 J / (1e-10 m) / (1.66054e-27 kg)
            = 9.6485e17 m/s²
            = 9.6485e17 × 1e-10 Å / (1e15 fs)²
            = 9.6485e-3 Å/fs²

Public API
----------
maxwell_boltzmann_velocities(masses, T_K, seed)  → ndarray (N, 3) [Å/fs]
kinetic_energy(vel, masses)                       → float [eV]
temperature_from_ke(ke, n_atoms)                  → float [K]
velocity_verlet_step(pos, vel, forces_fn, masses, dt)
    → (pos_new, vel_new, forces_new)
run_md(pos0, vel0, forces_fn, masses, dt, n_steps, callback)
    → list of trajectory dicts
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KB_EV      = 8.617333e-5   # Boltzmann constant  [eV/K]
CONV_ACCEL = 9.6485e-3     # (eV/Å)/amu → Å/fs²


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann velocity sampling
# ---------------------------------------------------------------------------

def maxwell_boltzmann_velocities(masses, T_K, seed=42):
    """
    Sample initial velocities from the Maxwell-Boltzmann distribution.

    Each Cartesian component is drawn from N(0, σ) where
    σ = √(k_B T / m)  [Å/fs].

    Centre-of-mass velocity is removed to prevent drift.

    Parameters
    ----------
    masses : ndarray (N,)  — atomic masses in amu
    T_K    : float         — temperature in Kelvin
    seed   : int           — RNG seed for reproducibility

    Returns
    -------
    vel : ndarray (N, 3)  [Å/fs]
    """
    rng = np.random.default_rng(seed)
    kT  = KB_EV * T_K
    vel = np.zeros((len(masses), 3))
    for i, m in enumerate(masses):
        sigma    = np.sqrt(kT * CONV_ACCEL / m)   # Å/fs
        vel[i]   = rng.normal(0.0, sigma, 3)
    # Remove centre-of-mass drift
    v_cm = (masses[:, None] * vel).sum(axis=0) / masses.sum()
    vel -= v_cm
    return vel


# ---------------------------------------------------------------------------
# Kinetic energy and temperature
# ---------------------------------------------------------------------------

def kinetic_energy(vel, masses):
    """
    Kinetic energy KE = ½ Σᵢ mᵢ vᵢ²  in eV.

    Parameters
    ----------
    vel    : ndarray (N, 3)  [Å/fs]
    masses : ndarray (N,)    [amu]

    Returns
    -------
    KE : float  [eV]
    """
    return float(0.5 * np.sum(masses[:, None] * vel**2) / CONV_ACCEL)


def temperature_from_ke(ke, n_atoms):
    """
    Instantaneous temperature from kinetic energy.

    T = 2 KE / (3 N k_B)   [K]

    Parameters
    ----------
    ke      : float  — kinetic energy [eV]
    n_atoms : int    — number of atoms

    Returns
    -------
    T : float  [K]
    """
    return float(2.0 * ke / (3.0 * n_atoms * KB_EV))


# ---------------------------------------------------------------------------
# Velocity Verlet integrator
# ---------------------------------------------------------------------------

def velocity_verlet_step(pos, vel, forces_fn, masses, dt):
    """
    Single Velocity Verlet step.

    Algorithm:
      1. v(t + ½dt) = v(t) + ½ a(t) dt
      2. r(t + dt)  = r(t) + v(t + ½dt) dt
      3. a(t + dt)  = F(r(t + dt)) / m
      4. v(t + dt)  = v(t + ½dt) + ½ a(t + dt) dt

    Parameters
    ----------
    pos       : ndarray (N, 3)  — positions [Å]
    vel       : ndarray (N, 3)  — velocities [Å/fs]
    forces_fn : callable        — forces_fn(pos) → ndarray (N, 3) [eV/Å]
    masses    : ndarray (N,)    — masses [amu]
    dt        : float           — timestep [fs]

    Returns
    -------
    pos_new    : ndarray (N, 3)  [Å]
    vel_new    : ndarray (N, 3)  [Å/fs]
    forces_new : ndarray (N, 3)  [eV/Å]
    """
    F_cur  = forces_fn(pos)
    accel  = F_cur * CONV_ACCEL / masses[:, None]   # Å/fs²

    v_half   = vel + 0.5 * accel * dt
    pos_new  = pos + v_half * dt

    F_new    = forces_fn(pos_new)
    accel_new = F_new * CONV_ACCEL / masses[:, None]

    vel_new  = v_half + 0.5 * accel_new * dt

    return pos_new, vel_new, F_new


# ---------------------------------------------------------------------------
# MD trajectory runner
# ---------------------------------------------------------------------------

def run_md(pos0, vel0, forces_fn, masses, dt, n_steps, callback=None):
    """
    Run a Velocity Verlet MD trajectory.

    Parameters
    ----------
    pos0      : ndarray (N, 3)  — initial positions [Å]
    vel0      : ndarray (N, 3)  — initial velocities [Å/fs]
    forces_fn : callable        — forces_fn(pos) → ndarray (N, 3) [eV/Å]
    masses    : ndarray (N,)    — masses [amu]
    dt        : float           — timestep [fs]
    n_steps   : int             — number of MD steps
    callback  : callable or None
        If provided, called as callback(step, pos, vel, forces, ke, pe)
        after each step.  Return value is ignored.

    Returns
    -------
    traj : list of dict, one per step (including step 0)
        Each dict has keys:
          'step'      : int
          'time'      : float  [fs]
          'positions' : ndarray (N, 3)  [Å]
          'velocities': ndarray (N, 3)  [Å/fs]
          'forces'    : ndarray (N, 3)  [eV/Å]
          'ke'        : float  [eV]
    """
    pos = pos0.copy()
    vel = vel0.copy()
    F   = forces_fn(pos)
    ke  = kinetic_energy(vel, masses)

    traj = []

    def _record(step, pos, vel, F, ke):
        traj.append({
            "step":       step,
            "time":       step * dt,
            "positions":  pos.copy(),
            "velocities": vel.copy(),
            "forces":     F.copy(),
            "ke":         ke,
        })

    _record(0, pos, vel, F, ke)
    if callback is not None:
        callback(0, pos, vel, F, ke, 0.0)

    for step in range(1, n_steps + 1):
        pos, vel, F = velocity_verlet_step(pos, vel, forces_fn, masses, dt)
        ke = kinetic_energy(vel, masses)
        _record(step, pos, vel, F, ke)
        if callback is not None:
            callback(step, pos, vel, F, ke, 0.0)

    return traj


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from engine_pes import potential_energy, forces as pes_forces, WATER_EQ
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    MASSES = np.array([15.9994, 1.008, 1.008])
    T_K    = 300.0
    DT     = 0.5    # fs
    N_STEPS = 20

    print("engine_md.py self-test")
    print("=" * 48)

    vel0 = maxwell_boltzmann_velocities(MASSES, T_K, seed=42)
    ke0  = kinetic_energy(vel0, MASSES)
    T0   = temperature_from_ke(ke0, 3)
    print(f"  Initial KE = {ke0:.4f} eV  (T ≈ {T0:.1f} K, target {T_K:.0f} K)")

    def forces_fn(pos):
        return pes_forces(pos)

    traj = run_md(WATER_EQ.copy(), vel0, forces_fn, MASSES, DT, N_STEPS)

    pe0 = potential_energy(traj[0]["positions"])
    E0  = traj[0]["ke"] + pe0
    pe_last = potential_energy(traj[-1]["positions"])
    E_last  = traj[-1]["ke"] + pe_last
    drift   = abs(E_last - E0)
    print(f"  E(t=0)    = {E0:.6f} eV  (KE={traj[0]['ke']:.4f}, PE={pe0:.4f})")
    print(f"  E(t={N_STEPS*DT:.0f} fs) = {E_last:.6f} eV  (KE={traj[-1]['ke']:.4f}, PE={pe_last:.4f})")
    print(f"  |ΔE|      = {drift:.2e} eV  {'PASS' if drift < 0.01 else 'WARN'}")
    print(f"  Steps     = {len(traj) - 1}")
