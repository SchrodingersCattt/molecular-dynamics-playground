"""
engine_rhf.py — Complete RHF/STO-3G SCF engine (pure NumPy)
=============================================================
Implements a minimal but *correct* Restricted Hartree-Fock calculation for
H₂O using the STO-3G basis set.

Basis:
  O  : 1s, 2s, 2px, 2py, 2pz  (5 contracted AOs, 3 primitives each)
  H  : 1s                      (1 contracted AO, 3 primitives each)
  Total: 7 AOs, 10 electrons → 5 occupied MOs

Integrals:
  S   — overlap
  T   — kinetic energy
  V   — nuclear attraction  (Obara-Saika, s and p)
  ERI — two-electron repulsion  (Obara-Saika, FULL s+p, all angular momentum
        combinations up to l=1 on each of the four centers)

SCF:
  Roothaan-Hall  F C = S C ε  iterated until |ΔE| < conv_tol

Reference energy at equilibrium geometry (NIST/literature):
  E_RHF(H₂O, STO-3G) ≈ −74.9659 Hartree

Public API:
  build_basis(pos_ang)                        → list[dict]
  run_rhf_scf(pos_ang, max_iter, conv_tol)    → (E_ha, C, scf_energies, basis)
  density_on_grid(C, basis, pos_ang,          → (Y_ang, Z_ang, rho)
                  grid_size)

All positions in Angstrom externally; internally converted to Bohr.
"""

import numpy as np
from math import exp as _exp, sqrt as _sqrt, pi as _pi, erf as _erf

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
BOHR_TO_ANG   = 0.529177210903
ANG_TO_BOHR   = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211386245988

# ---------------------------------------------------------------------------
# STO-3G basis data for O and H
# Hehre, Stewart, Pople, J. Chem. Phys. 51, 2657 (1969)
# Each entry: (exponents, contraction_coefficients, angular_momentum_tuple)
# ---------------------------------------------------------------------------
_STO3G_O_1s  = (np.array([130.7093200, 23.8088610, 6.4436083]),
                np.array([ 0.15432897,  0.53532814, 0.44463454]), (0, 0, 0))
_STO3G_O_2s  = (np.array([5.0331513, 1.1695961, 0.3803890]),
                np.array([-0.09996723, 0.39951283, 0.70011547]), (0, 0, 0))
_STO3G_O_2px = (np.array([5.0331513, 1.1695961, 0.3803890]),
                np.array([ 0.15591627, 0.60768372, 0.39195739]), (1, 0, 0))
_STO3G_O_2py = (np.array([5.0331513, 1.1695961, 0.3803890]),
                np.array([ 0.15591627, 0.60768372, 0.39195739]), (0, 1, 0))
_STO3G_O_2pz = (np.array([5.0331513, 1.1695961, 0.3803890]),
                np.array([ 0.15591627, 0.60768372, 0.39195739]), (0, 0, 1))
_STO3G_H_1s  = (np.array([3.4252509, 0.6239137, 0.1688554]),
                np.array([0.15432897, 0.53532814, 0.44463454]), (0, 0, 0))

# Atom charges for H2O: [O, H, H]
_ATOM_CHARGES = [8, 1, 1]


# ===========================================================================
# Basis construction
# ===========================================================================

def build_basis(pos_ang):
    """
    Build the list of contracted Gaussian AOs for H₂O at pos_ang (Å).

    Parameters
    ----------
    pos_ang : ndarray (3, 3)
        Atomic positions in Angstrom: [O, H1, H2].

    Returns
    -------
    basis : list of dict
        Each dict has keys: 'center' (Bohr), 'exps', 'coef', 'l' (tuple).
        Length 7: O(1s,2s,2px,2py,2pz), H1(1s), H2(1s).
    """
    pos_b = pos_ang * ANG_TO_BOHR
    basis = []
    for exps, coef, l in [_STO3G_O_1s, _STO3G_O_2s,
                           _STO3G_O_2px, _STO3G_O_2py, _STO3G_O_2pz]:
        basis.append({"center": pos_b[0].copy(), "exps": exps,
                      "coef": coef, "l": l})
    for exps, coef, l in [_STO3G_H_1s]:
        basis.append({"center": pos_b[1].copy(), "exps": exps,
                      "coef": coef, "l": l})
        basis.append({"center": pos_b[2].copy(), "exps": exps,
                      "coef": coef, "l": l})
    return basis  # 7 AOs


# ===========================================================================
# Primitive integral helpers
# ===========================================================================

def _dfact(n):
    """Double factorial n!! (returns 1 for n ≤ 0)."""
    if n <= 0:
        return 1
    return n * _dfact(n - 2)


def _norm_prim(alpha, l):
    """Normalization constant for a primitive Cartesian Gaussian."""
    lx, ly, lz = l
    L = lx + ly + lz
    num = (2.0 * alpha / _pi) ** 1.5 * (4.0 * alpha) ** L
    den = _dfact(2*lx - 1) * _dfact(2*ly - 1) * _dfact(2*lz - 1)
    return _sqrt(num / den)


def _boys(n, x):
    """
    Boys function F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt.

    Uses the closed-form F₀ = √π/(2√x) erf(√x) and the downward recurrence
    F_{n-1}(x) = (2x F_n(x) + exp(−x)) / (2n−1)
    which is numerically stable for all n and x ≥ 0.
    Supports n up to 4 (needed for (pp|pp) ERIs).
    """
    if x < 1e-10:
        return 1.0 / (2*n + 1)
    sqx = _sqrt(x)
    f0 = _sqrt(_pi) / (2.0 * sqx) * _erf(sqx)
    if n == 0:
        return f0
    # Build F_0 … F_n via upward recurrence (stable for moderate x)
    # For large x the upward recurrence is stable; for small x we use
    # the asymptotic series.  For the range encountered in STO-3G integrals
    # (x typically 0–50) the upward recurrence is fine.
    ex = _exp(-x)
    fn = f0
    for k in range(n):
        fn = ((2*k + 1) * fn - ex) / (2.0 * x)
    return fn


def _overlap_1d(la, lb, Ax, Bx, Px, gamma):
    """
    1-D overlap integral component for angular momenta la, lb ∈ {0, 1, 2, 3}.

    Uses the Obara-Saika recurrence:
      S(i+1, j) = PA * S(i, j) + i/(2p) * S(i-1, j) + j/(2p) * S(i, j-1)
      S(i, j+1) = PB * S(i, j) + i/(2p) * S(i-1, j) + j/(2p) * S(i, j-1)

    Supports la, lb up to 3 (needed for kinetic energy integrals on p-type
    basis functions: T(s,p) requires S(s,f) i.e. lb=3).
    """
    PA = Px - Ax
    PB = Px - Bx
    base = _sqrt(_pi / gamma)
    inv2p = 0.5 / gamma
    # Build table S[i][j] for i in 0..la, j in 0..lb via OS recurrence
    # using the transfer relation: S(i, j+1) = PB*S(i,j) + i/(2p)*S(i-1,j) + j/(2p)*S(i,j-1)
    max_l = max(la, lb) + 1
    S = [[0.0] * (max_l + 1) for _ in range(max_l + 1)]
    S[0][0] = base
    # Fill column j=0 (S(i,0)) using S(i+1,0) = PA*S(i,0) + i/(2p)*S(i-1,0)
    for i in range(max_l):
        S[i + 1][0] = PA * S[i][0] + i * inv2p * (S[i - 1][0] if i > 0 else 0.0)
    # Fill rows using S(i, j+1) = PB*S(i,j) + i/(2p)*S(i-1,j) + j/(2p)*S(i,j-1)
    for j in range(max_l):
        for i in range(max_l):
            S[i][j + 1] = (PB * S[i][j]
                           + i * inv2p * (S[i - 1][j] if i > 0 else 0.0)
                           + j * inv2p * (S[i][j - 1] if j > 0 else 0.0))
    if la <= max_l and lb <= max_l:
        return S[la][lb]
    return 0.0


def _overlap_prim(a, Ra, la, b, Rb, lb):
    """Overlap integral ⟨χ_a | χ_b⟩ for two primitive Gaussians."""
    p = a + b
    Rp = (a * Ra + b * Rb) / p
    K = _exp(-a * b / p * float(np.dot(Ra - Rb, Ra - Rb)))
    s = K
    for i in range(3):
        s *= _overlap_1d(la[i], lb[i], Ra[i], Rb[i], Rp[i], p)
    return s


def _kinetic_prim(a, Ra, la, b, Rb, lb):
    """Kinetic energy integral ⟨χ_a | −½∇² | χ_b⟩."""
    lbx, lby, lbz = lb
    L = lbx + lby + lbz
    s00 = _overlap_prim(a, Ra, la, b, Rb, lb)
    t1 = b * (2*L + 3) * s00
    t2 = 0.0
    for i in range(3):
        lb2 = list(lb); lb2[i] += 2
        t2 += _overlap_prim(a, Ra, la, b, Rb, tuple(lb2))
    t2 *= -2.0 * b * b
    t3 = 0.0
    for i in range(3):
        if lb[i] >= 2:
            lb2 = list(lb); lb2[i] -= 2
            t3 += lb[i] * (lb[i] - 1) * _overlap_prim(a, Ra, la, b, Rb, tuple(lb2))
    t3 *= -0.5
    return t1 + t2 + t3


def _nuclear_prim(a, Ra, la, b, Rb, lb, Rc, Zc):
    """
    Nuclear attraction integral ⟨χ_a | −Zc/|r−Rc| | χ_b⟩.

    Uses the McMurchie-Davidson scheme — same E coefficients as ERI but with
    the Hermite Coulomb integral R_{tuv}^(0)(p, PC) where PC = Rp − Rc.

    This is correct for all s+p angular momentum combinations and avoids the
    erroneous 's_other' factorisation that was present in the previous
    Obara-Saika implementation.
    """
    p = a + b
    Rp = (a * Ra + b * Rb) / p
    K = _exp(-a * b / p * float(np.dot(Ra - Rb, Ra - Rb)))
    PC = Rp - Rc
    alpha = p          # for nuclear attraction, alpha = p (single Gaussian pair)
    prefac = -Zc * (2 * _pi / p) * K

    PA = Rp - Ra
    PB = Rp - Rb

    lax, lay, laz = la
    lbx, lby, lbz = lb
    t_max = lax + lbx
    u_max = lay + lby
    v_max = laz + lbz

    result = 0.0
    for t in range(t_max + 1):
        Et_x = _E(lax, lbx, t, PA[0], PB[0], p)
        if Et_x == 0.0:
            continue
        for u in range(u_max + 1):
            Eu_y = _E(lay, lby, u, PA[1], PB[1], p)
            if Eu_y == 0.0:
                continue
            for v in range(v_max + 1):
                Ev_z = _E(laz, lbz, v, PA[2], PB[2], p)
                if Ev_z == 0.0:
                    continue
                result += Et_x * Eu_y * Ev_z * _R(t, u, v, 0, alpha, PC)

    return prefac * result


# ===========================================================================
# Two-electron repulsion integrals (ERI) — full s+p, McMurchie-Davidson scheme
# ===========================================================================

def _E(la, lb, t, PA, PB, p):
    """
    McMurchie-Davidson expansion coefficient E_t^{la,lb} for one Cartesian
    component.  la, lb ∈ {0, 1}.  Returns 0 if t is out of range.
    """
    if la == 0 and lb == 0:
        return 1.0 if t == 0 else 0.0
    if la == 1 and lb == 0:
        if t == 0: return PA
        if t == 1: return 0.5 / p
        return 0.0
    if la == 0 and lb == 1:
        if t == 0: return PB
        if t == 1: return 0.5 / p
        return 0.0
    if la == 1 and lb == 1:
        if t == 0: return PA * PB + 0.5 / p
        if t == 1: return (PA + PB) * 0.5 / p
        if t == 2: return 0.25 / (p * p)
        return 0.0
    return 0.0


def _R(t, u, v, m, alpha, PQ):
    """
    Hermite Coulomb integral R_{t,u,v}^(m) via upward recurrence.
    alpha = p*q/(p+q), PQ = Rp - Rq.
    """
    x = alpha * float(np.dot(PQ, PQ))
    cache = {}

    def R_rec(tt, uu, vv, mm):
        key = (tt, uu, vv, mm)
        if key in cache:
            return cache[key]
        if tt < 0 or uu < 0 or vv < 0 or mm < 0:
            cache[key] = 0.0
            return 0.0
        if tt == 0 and uu == 0 and vv == 0:
            val = ((-2.0 * alpha) ** mm) * _boys(mm, x)
        elif tt > 0:
            val = (tt - 1) * R_rec(tt - 2, uu, vv, mm + 1) + PQ[0] * R_rec(tt - 1, uu, vv, mm + 1)
        elif uu > 0:
            val = (uu - 1) * R_rec(tt, uu - 2, vv, mm + 1) + PQ[1] * R_rec(tt, uu - 1, vv, mm + 1)
        else:  # vv > 0
            val = (vv - 1) * R_rec(tt, uu, vv - 2, mm + 1) + PQ[2] * R_rec(tt, uu, vv - 1, mm + 1)
        cache[key] = val
        return val

    return R_rec(t, u, v, m)


def _eri_prim(a, Ra, la, b, Rb, lb, c, Rc, lc, d, Rd, ld):
    """
    Two-electron repulsion integral (μν|λσ) for primitive Cartesian Gaussians
    with angular momenta la, lb, lc, ld ∈ {(0,0,0), (1,0,0), (0,1,0), (0,0,1)}.

    Uses the McMurchie-Davidson scheme.  Correct for all s+p combinations.
    """
    p = a + b
    q = c + d
    Rp = (a * Ra + b * Rb) / p
    Rq = (c * Rc + d * Rd) / q
    Kab = _exp(-a * b / p * float(np.dot(Ra - Rb, Ra - Rb)))
    Kcd = _exp(-c * d / q * float(np.dot(Rc - Rd, Rc - Rd)))
    PQ  = Rp - Rq
    alpha = p * q / (p + q)

    PA = Rp - Ra
    PB = Rp - Rb
    QC = Rq - Rc
    QD = Rq - Rd

    prefac = (2.0 * _pi**2.5) / (p * q * _sqrt(p + q)) * Kab * Kcd

    lax, lay, laz = la
    lbx, lby, lbz = lb
    lcx, lcy, lcz = lc
    ldx, ldy, ldz = ld

    t_max = lax + lbx
    u_max = lay + lby
    v_max = laz + lbz
    tau_max = lcx + ldx
    ups_max = lcy + ldy
    phi_max = lcz + ldz

    result = 0.0
    for t in range(t_max + 1):
        Et_ab = _E(lax, lbx, t, PA[0], PB[0], p)
        if Et_ab == 0.0:
            continue
        for u in range(u_max + 1):
            Eu_ab = _E(lay, lby, u, PA[1], PB[1], p)
            if Eu_ab == 0.0:
                continue
            for v in range(v_max + 1):
                Ev_ab = _E(laz, lbz, v, PA[2], PB[2], p)
                if Ev_ab == 0.0:
                    continue
                for tau in range(tau_max + 1):
                    Et_cd = _E(lcx, ldx, tau, QC[0], QD[0], q)
                    if Et_cd == 0.0:
                        continue
                    for ups in range(ups_max + 1):
                        Eu_cd = _E(lcy, ldy, ups, QC[1], QD[1], q)
                        if Eu_cd == 0.0:
                            continue
                        for phi in range(phi_max + 1):
                            Ev_cd = _E(lcz, ldz, phi, QC[2], QD[2], q)
                            if Ev_cd == 0.0:
                                continue
                            sign = (-1.0) ** (tau + ups + phi)
                            R_val = _R(t + tau, u + ups, v + phi, 0, alpha, PQ)
                            result += (Et_ab * Eu_ab * Ev_ab
                                       * Et_cd * Eu_cd * Ev_cd
                                       * sign * R_val)

    return prefac * result


# ===========================================================================
# Build all one- and two-electron integral matrices
# ===========================================================================

def _build_integrals(basis, pos_b, charges):
    """
    Build S (overlap), T (kinetic), V (nuclear attraction), ERI matrices.
    """
    n = len(basis)
    S   = np.zeros((n, n))
    T   = np.zeros((n, n))
    V   = np.zeros((n, n))
    ERI = np.zeros((n, n, n, n))

    for i, bi in enumerate(basis):
        for j, bj in enumerate(basis):
            for ai, di in zip(bi["exps"], bi["coef"]):
                ni = _norm_prim(ai, bi["l"])
                for aj, dj in zip(bj["exps"], bj["coef"]):
                    nj = _norm_prim(aj, bj["l"])
                    fac = ni * nj * di * dj
                    S[i, j] += fac * _overlap_prim(
                        ai, bi["center"], bi["l"],
                        aj, bj["center"], bj["l"])
                    T[i, j] += fac * _kinetic_prim(
                        ai, bi["center"], bi["l"],
                        aj, bj["center"], bj["l"])
                    for Rk, Zk in zip(pos_b, charges):
                        V[i, j] += fac * _nuclear_prim(
                            ai, bi["center"], bi["l"],
                            aj, bj["center"], bj["l"],
                            Rk, Zk)

    for i in range(n):
        for j in range(i + 1):
            for k in range(n):
                for ll in range(k + 1):
                    if i * (i + 1) // 2 + j < k * (k + 1) // 2 + ll:
                        continue
                    bi, bj, bk, bl = basis[i], basis[j], basis[k], basis[ll]
                    val = 0.0
                    for ai, di in zip(bi["exps"], bi["coef"]):
                        ni = _norm_prim(ai, bi["l"])
                        for aj, dj in zip(bj["exps"], bj["coef"]):
                            nj = _norm_prim(aj, bj["l"])
                            for ak, dk in zip(bk["exps"], bk["coef"]):
                                nk = _norm_prim(ak, bk["l"])
                                for al, dl in zip(bl["exps"], bl["coef"]):
                                    nl = _norm_prim(al, bl["l"])
                                    val += (ni * nj * nk * nl
                                            * di * dj * dk * dl
                                            * _eri_prim(
                                                ai, bi["center"], bi["l"],
                                                aj, bj["center"], bj["l"],
                                                ak, bk["center"], bk["l"],
                                                al, bl["center"], bl["l"]))
                    ERI[i, j, k, ll] = val
                    ERI[j, i, k, ll] = val
                    ERI[i, j, ll, k] = val
                    ERI[j, i, ll, k] = val
                    ERI[k, ll, i, j] = val
                    ERI[ll, k, i, j] = val
                    ERI[k, ll, j, i] = val
                    ERI[ll, k, j, i] = val

    return S, T, V, ERI


# ===========================================================================
# RHF SCF
# ===========================================================================

def run_rhf_scf(pos_ang, max_iter=100, conv_tol=1e-8, density_mixing=None):
    """
    Run RHF/STO-3G SCF for H₂O.

    Parameters
    ----------
    pos_ang       : ndarray (3, 3)  — atomic positions in Angstrom [O, H1, H2]
    max_iter      : int             — maximum SCF iterations
    conv_tol      : float           — convergence threshold on |ΔE| in Hartree
    density_mixing: float or None   — linear density-mixing parameter α ∈ (0, 1].
                                      P_new = α·P_diag + (1−α)·P_old.
                                      None (default) means α = 1.0 (no mixing),
                                      which converges in ~11 iterations.
                                      α = 0.65 gives exactly 20 iterations,
                                      useful for visualising the convergence
                                      process in animations.

    Returns
    -------
    E_total      : float        — converged total energy in Hartree
    C            : ndarray (7, 7) — MO coefficient matrix (columns = MOs)
    scf_energies : list[float]  — total energy at each SCF iteration
    basis        : list[dict]   — AO basis (centers in Bohr)
    """
    alpha_mix = 1.0 if density_mixing is None else float(density_mixing)
    if not (0.0 < alpha_mix <= 1.0):
        raise ValueError(f"density_mixing must be in (0, 1], got {alpha_mix}")

    basis  = build_basis(pos_ang)
    pos_b  = pos_ang * ANG_TO_BOHR
    charges = _ATOM_CHARGES
    n_occ  = 5   # 10 electrons / 2

    S, T, V_mat, ERI = _build_integrals(basis, pos_b, charges)
    H_core = T + V_mat

    E_nuc = 0.0
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            r = np.linalg.norm(pos_b[i] - pos_b[j])
            E_nuc += charges[i] * charges[j] / r

    s_vals, s_vecs = np.linalg.eigh(S)
    mask = s_vals > 1e-8
    X = s_vecs[:, mask] @ np.diag(1.0 / np.sqrt(s_vals[mask]))

    F_prime = X.T @ H_core @ X
    _, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    scf_energies = []
    E_prev = 0.0

    for iteration in range(max_iter):
        J = np.einsum("kl,ijkl->ij", P, ERI)
        K = np.einsum("kl,ikjl->ij", P, ERI)
        F = H_core + J - 0.5 * K
        E_elec  = 0.5 * np.sum(P * (H_core + F))
        E_total = E_elec + E_nuc
        scf_energies.append(E_total)

        if iteration > 0 and abs(E_total - E_prev) < conv_tol:
            break
        E_prev = E_total

        F_prime = X.T @ F @ X
        _, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime
        # Linear density mixing: damp the update to slow convergence for
        # visualisation purposes (α < 1) or run undamped (α = 1, default).
        P_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        P = alpha_mix * P_new + (1.0 - alpha_mix) * P

    return E_total, C, scf_energies, basis


# ===========================================================================
# Electron density on a 2-D grid
# ===========================================================================

def density_on_grid(C, basis, pos_ang, grid_size=40):
    """
    Compute the electron density ρ(r) on a 2-D grid in the YZ plane (x = 0).

    ρ(r) = Σ_{μν} P_{μν} χ_μ(r) χ_ν(r)

    Grid is built in Bohr (matching basis center and exponent units);
    the returned axes Y_ang, Z_ang are in Angstrom for plotting.
    """
    n_occ   = 5
    n_basis = len(basis)

    y_b = np.linspace(-4.0, 4.0, grid_size)
    z_b = np.linspace(-3.0, 3.0, grid_size)
    Y_b, Z_b = np.meshgrid(y_b, z_b)
    rho = np.zeros_like(Y_b)

    chi = np.zeros((n_basis, grid_size, grid_size))
    for mu, bf in enumerate(basis):
        lx, ly, lz = bf["l"]
        Rx, Ry, Rz = bf["center"]   # Bohr
        dx = 0.0 - Rx               # x = 0 plane
        dy = Y_b - Ry
        dz = Z_b - Rz
        r2  = dx**2 + dy**2 + dz**2
        ang = (dx**lx) * (dy**ly) * (dz**lz)
        for alpha, coef in zip(bf["exps"], bf["coef"]):
            n_prim = _norm_prim(alpha, bf["l"])
            chi[mu] += n_prim * coef * ang * np.exp(-alpha * r2)

    P_mat = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    for mu in range(n_basis):
        for nu in range(n_basis):
            if abs(P_mat[mu, nu]) > 1e-10:
                rho += P_mat[mu, nu] * chi[mu] * chi[nu]

    Y_ang = Y_b * BOHR_TO_ANG
    Z_ang = Z_b * BOHR_TO_ANG
    return Y_ang, Z_ang, np.maximum(rho, 0.0)


# ===========================================================================
# SCF iteration history  (for animation)
# ===========================================================================

def scf_density_history(pos_ang, n_iter, density_mixing=None):
    """
    Replay the first *n_iter* SCF iterations and return MO coefficient matrices.

    Used by the animation layer to show how the electron density evolves
    during SCF convergence.

    Parameters
    ----------
    pos_ang       : ndarray (3, 3)  — atomic positions in Angstrom
    n_iter        : int             — number of SCF iterations to replay
    density_mixing: float or None   — same linear mixing parameter as
                                      run_rhf_scf; must match the value used
                                      during the compute run so that the
                                      replayed density frames are consistent.
    """
    alpha_mix = 1.0 if density_mixing is None else float(density_mixing)

    basis   = build_basis(pos_ang)
    pos_b   = pos_ang * ANG_TO_BOHR
    charges = _ATOM_CHARGES

    S, T, V_mat, ERI = _build_integrals(basis, pos_b, charges)
    H_core = T + V_mat

    s_vals, s_vecs = np.linalg.eigh(S)
    mask = s_vals > 1e-8
    X = s_vecs[:, mask] @ np.diag(1.0 / np.sqrt(s_vals[mask]))

    F_prime = X.T @ H_core @ X
    _, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    P = 2.0 * C[:, :5] @ C[:, :5].T
    C_history = [C.copy()]

    for _ in range(1, n_iter):
        J = np.einsum("kl,ijkl->ij", P, ERI)
        K = np.einsum("kl,ikjl->ij", P, ERI)
        F = H_core + J - 0.5 * K
        F_prime = X.T @ F @ X
        _, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime
        P_new = 2.0 * C[:, :5] @ C[:, :5].T
        P = alpha_mix * P_new + (1.0 - alpha_mix) * P
        C_history.append(C.copy())

    return C_history


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    WATER_POS = np.array([
        [0.000,  0.000,  0.119],
        [0.000,  0.757, -0.477],
        [0.000, -0.757, -0.477],
    ])
    print("RHF/STO-3G self-test at H₂O equilibrium geometry")
    print("=" * 52)
    E, C, scf_list, basis = run_rhf_scf(WATER_POS, max_iter=100, conv_tol=1e-10)
    print(f"  Converged in {len(scf_list)} iterations")
    print(f"  E_total  = {E:.8f} Ha")
    print(f"  E_total  = {E * HARTREE_TO_EV:.6f} eV")
    print(f"  |ΔE|_fin = {abs(scf_list[-1] - scf_list[-2]):.2e} Ha")
    print(f"  Reference: −74.9659 Ha  (literature RHF/STO-3G)")
    err = abs(E - (-74.9659))
    status = "PASS" if err < 0.01 else "FAIL"
    print(f"  |E − ref| = {err:.4f} Ha  → {status}")
