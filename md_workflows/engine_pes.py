"""
engine_pes.py — Internal-coordinate PES and analytic forces for H₂O
====================================================================
Provides a physically reasonable intramolecular potential energy surface
for an isolated H₂O molecule in internal coordinates:

    V = ½ k_r Σᵢ (Δrᵢ)² + ¼ k_r4 Σᵢ (Δrᵢ)⁴ + ½ k_θ (Δθ)²

Force constants are tuned to reproduce literature-scale O-H stretch and
H-O-H bend frequencies while remaining numerically stable at dt = 1 fs.

Public API
----------
get_geometry(pos)           → (r1, r2, r_HH, theta, u_OH1, u_OH2)
potential_energy(pos)       → float  [eV]
forces(pos)                 → ndarray (3, 3)  [eV/Å]
forces_decomposed(pos)      → (f_total, f_bond, f_angle)  each (3, 3) [eV/Å]

All positions in Angstrom.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Equilibrium geometry (NIST)
# ---------------------------------------------------------------------------
WATER_EQ = np.array([
    [0.000,  0.000,  0.119],   # O
    [0.000,  0.757, -0.477],   # H1
    [0.000, -0.757, -0.477],   # H2
])

R0_OH   = float(np.linalg.norm(WATER_EQ[1] - WATER_EQ[0]))   # Å  ≈ 0.9572
THETA0  = float(np.arccos(np.clip(
    np.dot(WATER_EQ[1] - WATER_EQ[0], WATER_EQ[2] - WATER_EQ[0])
    / (np.linalg.norm(WATER_EQ[1] - WATER_EQ[0])
       * np.linalg.norm(WATER_EQ[2] - WATER_EQ[0])),
    -1.0, 1.0)))                                               # rad ≈ 1.8238

# ---------------------------------------------------------------------------
# Force constants  (eV/Å², eV/Å⁴, eV/rad²)
# ---------------------------------------------------------------------------
K_BOND      = 32.0    # eV/Å²   — harmonic O-H stretch
K_BOND4     = 80.0    # eV/Å⁴   — quartic stabilisation
K_ANGLE     = 7.5     # eV/rad² — H-O-H bend

_ANGLE_EPS  = 1.0e-10


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def get_geometry(pos):
    """
    Extract internal coordinates from Cartesian positions.

    Parameters
    ----------
    pos : ndarray (3, 3)  — [O, H1, H2] in Angstrom

    Returns
    -------
    r1     : float   — |O-H1| bond length (Å)
    r2     : float   — |O-H2| bond length (Å)
    r_HH   : float   — |H1-H2| distance (Å)
    theta  : float   — H-O-H bond angle (rad)
    u_OH1  : ndarray (3,) — unit vector O→H1
    u_OH2  : ndarray (3,) — unit vector O→H2
    """
    v1 = pos[1] - pos[0]
    v2 = pos[2] - pos[0]
    r1 = float(np.linalg.norm(v1))
    r2 = float(np.linalg.norm(v2))
    r_HH = float(np.linalg.norm(pos[1] - pos[2]))
    cos_t = float(np.clip(np.dot(v1, v2) / max(r1 * r2, _ANGLE_EPS), -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    u_OH1 = v1 / max(r1, _ANGLE_EPS)
    u_OH2 = v2 / max(r2, _ANGLE_EPS)
    return r1, r2, r_HH, theta, u_OH1, u_OH2


# ---------------------------------------------------------------------------
# Potential energy
# ---------------------------------------------------------------------------

def potential_energy(pos):
    """
    Intramolecular H₂O potential energy in eV.

    V = ½ k_r [(Δr₁)² + (Δr₂)²]
      + ¼ k_r4 [(Δr₁)⁴ + (Δr₂)⁴]
      + ½ k_θ (Δθ)²

    Parameters
    ----------
    pos : ndarray (3, 3)  — [O, H1, H2] in Angstrom

    Returns
    -------
    V : float  [eV]
    """
    r1, r2, _, theta, _, _ = get_geometry(pos)
    dr1 = r1 - R0_OH
    dr2 = r2 - R0_OH
    dtheta = theta - THETA0
    V_bond  = 0.5 * K_BOND  * (dr1**2 + dr2**2)
    V_bond4 = 0.25 * K_BOND4 * (dr1**4 + dr2**4)
    V_angle = 0.5 * K_ANGLE * dtheta**2
    return float(V_bond + V_bond4 + V_angle)


# ---------------------------------------------------------------------------
# Analytic forces
# ---------------------------------------------------------------------------

def forces(pos):
    """
    Analytic forces −∂V/∂r in eV/Å.

    Parameters
    ----------
    pos : ndarray (3, 3)  — [O, H1, H2] in Angstrom

    Returns
    -------
    F : ndarray (3, 3)  — forces on [O, H1, H2] in eV/Å
    """
    F, _, _ = forces_decomposed(pos)
    return F


def forces_decomposed(pos):
    """
    Analytic forces decomposed into bond and angle contributions.

    Parameters
    ----------
    pos : ndarray (3, 3)  — [O, H1, H2] in Angstrom

    Returns
    -------
    f_total : ndarray (3, 3)  [eV/Å]
    f_bond  : ndarray (3, 3)  [eV/Å]
    f_angle : ndarray (3, 3)  [eV/Å]
    """
    r1, r2, _, theta, u1, u2 = get_geometry(pos)
    dr1 = r1 - R0_OH
    dr2 = r2 - R0_OH
    dtheta = theta - THETA0

    # ── Bond force magnitudes ─────────────────────────────────────────────
    dV_dr1 = K_BOND * dr1 + K_BOND4 * dr1**3
    dV_dr2 = K_BOND * dr2 + K_BOND4 * dr2**3

    # ── Angle force magnitude ─────────────────────────────────────────────
    dV_dtheta = K_ANGLE * dtheta

    # ── Angle gradient w.r.t. H positions ────────────────────────────────
    # dθ/dr_H1 = −(u2 − cos θ · u1) / (r1 · sin θ)
    sin_theta = float(np.sqrt(max(1.0 - np.cos(theta)**2, _ANGLE_EPS)))
    dtheta_dH1 = -(u2 - np.cos(theta) * u1) / max(r1 * sin_theta, _ANGLE_EPS)
    dtheta_dH2 = -(u1 - np.cos(theta) * u2) / max(r2 * sin_theta, _ANGLE_EPS)

    # ── Assemble gradient (∂V/∂r) ─────────────────────────────────────────
    grad_bond  = np.zeros((3, 3))
    grad_angle = np.zeros((3, 3))

    grad_bond[1] += dV_dr1 * u1
    grad_bond[2] += dV_dr2 * u2
    grad_bond[0]  = -(grad_bond[1] + grad_bond[2])

    grad_angle[1] += dV_dtheta * dtheta_dH1
    grad_angle[2] += dV_dtheta * dtheta_dH2
    grad_angle[0]  = -(grad_angle[1] + grad_angle[2])

    f_bond  = -grad_bond
    f_angle = -grad_angle
    f_total = f_bond + f_angle
    return f_total, f_bond, f_angle


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pos = WATER_EQ.copy()
    print("engine_pes.py self-test at equilibrium geometry")
    print("=" * 48)
    r1, r2, r_HH, theta, _, _ = get_geometry(pos)
    print(f"  r_OH1  = {r1:.4f} Å  (ref {R0_OH:.4f} Å)")
    print(f"  r_OH2  = {r2:.4f} Å")
    print(f"  θ      = {np.degrees(theta):.2f}°  (ref {np.degrees(THETA0):.2f}°)")
    V = potential_energy(pos)
    print(f"  V(eq)  = {V:.6e} eV  (should be ~0)")
    F = forces(pos)
    print(f"  |F|max = {np.abs(F).max():.6e} eV/Å  (should be ~0)")

    eps = 1e-5
    F_fd = np.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            pos_p = pos.copy(); pos_p[i, j] += eps
            pos_m = pos.copy(); pos_m[i, j] -= eps
            F_fd[i, j] = -(potential_energy(pos_p) - potential_energy(pos_m)) / (2 * eps)
    err = np.abs(F - F_fd).max()
    print(f"  FD check |F_anal - F_fd|_max = {err:.2e}  "
          f"{'PASS' if err < 1e-6 else 'FAIL'}")
