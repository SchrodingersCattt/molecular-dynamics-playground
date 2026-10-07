"""UKS electronic-structure backend for the 03b reactive AIMD story.

The real backend is deliberately lazy-imported: the visualization repository can
still be inspected and its analytic demo can be rendered on machines that do not
have a compiler-backed PySCF installation.  A generated trajectory always
records its backend in the JSON manifest, so the analytic fallback cannot be
mistaken for a UKS calculation.

External units
--------------
positions and gradients are in Angstrom/eV/Angstrom, energies in eV, and time
is in fs.  PySCF internally uses Bohr and Hartree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211386245988
GRAD_HA_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG


class PySCFUnavailable(RuntimeError):
    """Raised when a real UKS calculation is requested without PySCF."""


def pyscf_available() -> bool:
    """Return whether the optional real UKS backend can be imported."""

    try:
        import pyscf  # noqa: F401
    except Exception:
        return False
    return True


def _require_pyscf() -> tuple[Any, Any, Any, Any]:
    try:
        from pyscf import dft, gto
        from pyscf.dft import numint
    except Exception as exc:  # pragma: no cover - depends on external install
        raise PySCFUnavailable(
            "PySCF is required for a real 03b UKS trajectory. "
            "Install a platform-supported PySCF build, then rerun without "
            "--demo."
        ) from exc
    return dft, gto, None, numint


@dataclass
class UKSResult:
    """The minimum electronic state needed by the MD and visual layers."""

    energy_ev: float
    forces_ev_ang: np.ndarray
    dm_alpha: np.ndarray
    dm_beta: np.ndarray
    scf_energies_ev: np.ndarray
    scf_residuals: np.ndarray
    spin_square: float
    spin_multiplicity: float
    spin_density_x: np.ndarray
    spin_density_y: np.ndarray
    rho_alpha: np.ndarray
    rho_beta: np.ndarray
    converged: bool
    iterations: int
    branch_id: int = 0
    density_grid_x: np.ndarray | None = None
    density_grid_y: np.ndarray | None = None
    density_grid_z: np.ndarray | None = None
    rho_alpha_3d: np.ndarray | None = None
    rho_beta_3d: np.ndarray | None = None

    @property
    def density_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return self.dm_alpha, self.dm_beta

    @property
    def spin_density(self) -> np.ndarray:
        return self.rho_alpha - self.rho_beta

    @property
    def spin_density_metric(self) -> float:
        """A grid-dependent but reproducible visual magnitude indicator."""

        if self.spin_density.size == 0:
            return 0.0
        return float(np.mean(np.abs(self.spin_density)))


def make_molecule(
    positions_ang: np.ndarray,
    elements: Iterable[str],
    *,
    charge: int = 0,
    spin: int = 0,
    basis: str = "def2-svp",
) -> Any:
    """Create a PySCF molecule using the 03b charge and spin convention."""

    _, gto, _, _ = _require_pyscf()
    positions_ang = np.asarray(positions_ang, dtype=float)
    atom = [(str(element), tuple(map(float, position))) for element, position in zip(elements, positions_ang)]
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.charge = int(charge)
    mol.spin = int(spin)  # N_alpha - N_beta; BS singlet uses 0.
    mol.basis = basis
    mol.verbose = 0
    mol.build()
    return mol


def _ao_fragment_weights(mol: Any, fragment_atoms: Iterable[int]) -> np.ndarray:
    """Return a smooth AO mask for a set of atoms."""

    atoms = {int(index) for index in fragment_atoms}
    weights = np.zeros(mol.nao_nr(), dtype=float)
    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        try:
            atom_index = int(str(label).split()[0])
        except (ValueError, IndexError):
            atom_index = -1
        weights[ao_index] = 1.0 if atom_index in atoms else 0.0
    return weights


def seed_broken_symmetry_guess(
    mf: Any,
    *,
    oh_atoms: Iterable[int] = (1, 2),
    seed_strength: float = 0.035,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a charge-preserving alpha/beta perturbation.

    The initial total density comes from PySCF's MINAO guess.  The perturbation
    only separates the two spin channels on the OH and ClO3 AO blocks; it is
    intentionally small so the converged state, rather than the seed, controls
    the result.
    """

    dm = np.asarray(mf.get_init_guess(key="minao"), dtype=float)
    if dm.ndim == 3:
        dm = np.sum(dm, axis=0)
    overlap = np.asarray(mf.get_ovlp(), dtype=float)
    oh = _ao_fragment_weights(mf.mol, oh_atoms)
    other = 1.0 - oh
    mask = np.diag(oh - other)
    delta = overlap @ mask @ overlap
    # Remove the component that would alter the total S-weighted spin count.
    trace_delta = float(np.trace(overlap @ delta))
    trace_metric = max(float(np.trace(overlap @ overlap)), 1.0e-12)
    delta = delta - (trace_delta / trace_metric) * overlap
    delta *= float(seed_strength)
    dm_alpha = 0.5 * dm + delta
    dm_beta = 0.5 * dm - delta
    return dm_alpha, dm_beta


def _plane_grid(positions_ang: np.ndarray, *, nx: int = 96, ny: int = 72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a fixed XY grid that contains the full kick trajectory."""

    positions_ang = np.asarray(positions_ang, dtype=float)
    x_min = float(np.min(positions_ang[:, 0]) - 3.0)
    x_max = float(np.max(positions_ang[:, 0]) + 3.0)
    y_min = float(np.min(positions_ang[:, 1]) - 3.0)
    y_max = float(np.max(positions_ang[:, 1]) + 3.0)
    x = np.linspace(x_min, x_max, int(nx))
    y = np.linspace(y_min, y_max, int(ny))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    zz = np.zeros_like(xx)
    return x, y, np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def evaluate_spin_density_plane(
    mol: Any,
    dm_alpha: np.ndarray,
    dm_beta: np.ndarray,
    positions_ang: np.ndarray,
    *,
    nx: int = 96,
    ny: int = 72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate alpha and beta density on a fixed Cartesian XY plane."""

    dft, _, _, numint = _require_pyscf()
    x, y, coords_ang = _plane_grid(positions_ang, nx=nx, ny=ny)
    coords_bohr = coords_ang * ANG_TO_BOHR
    ao = dft.numint.eval_ao(mol, coords_bohr)
    rho_alpha = np.asarray(numint.eval_rho(mol, ao, dm_alpha), dtype=float)
    rho_beta = np.asarray(numint.eval_rho(mol, ao, dm_beta), dtype=float)
    shape = (len(y), len(x))
    return x, y, rho_alpha.reshape(shape), rho_beta.reshape(shape)


def evaluate_spin_density_grid(
    mol: Any,
    dm_alpha: np.ndarray,
    dm_beta: np.ndarray,
    grid_x_ang: np.ndarray,
    grid_y_ang: np.ndarray,
    grid_z_ang: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate alpha/beta density on one fixed Cartesian 3-D grid.

    The returned values are in electron/Angstrom^3 and use ``(x, y, z)``
    array order.  Keeping the grid in the caller-owned Cartesian frame is
    deliberate: it lets Cube export and MatterVis overlays share the exact
    same atom coordinates without a hidden per-frame recentering.
    """

    dft, _, _, numint = _require_pyscf()
    x = np.asarray(grid_x_ang, dtype=float)
    y = np.asarray(grid_y_ang, dtype=float)
    z = np.asarray(grid_z_ang, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords_ang = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    ao = dft.numint.eval_ao(mol, coords_ang * ANG_TO_BOHR)
    rho_alpha = np.asarray(numint.eval_rho(mol, ao, dm_alpha), dtype=float)
    rho_beta = np.asarray(numint.eval_rho(mol, ao, dm_beta), dtype=float)
    # PySCF reports electron/Bohr^3; the project stores spatial fields in the
    # same Angstrom coordinate frame as the trajectory and Cube atom records.
    scale = BOHR_TO_ANG**3
    shape = (len(x), len(y), len(z))
    return (rho_alpha.reshape(shape) * scale, rho_beta.reshape(shape) * scale)


class UKSEngine:
    """Small stateful PySCF UKS wrapper for Born--Oppenheimer MD."""

    def __init__(
        self,
        elements: Iterable[str],
        *,
        charge: int = 0,
        spin: int = 0,
        basis: str = "def2-svp",
        xc: str = "pbe0",
        grid_level: int = 2,
        conv_tol: float = 1.0e-8,
        max_cycle: int = 100,
        density_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.elements = tuple(str(element) for element in elements)
        self.charge = int(charge)
        self.spin = int(spin)
        self.basis = str(basis)
        self.xc = str(xc)
        self.grid_level = int(grid_level)
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.density_grid = density_grid

    def evaluate(
        self,
        positions_ang: np.ndarray,
        *,
        dm0: tuple[np.ndarray, np.ndarray] | None = None,
        branch_id: int = 0,
        seed_if_missing: bool = True,
    ) -> UKSResult:
        dft, _, _, _ = _require_pyscf()
        mol = make_molecule(
            positions_ang,
            self.elements,
            charge=self.charge,
            spin=self.spin,
            basis=self.basis,
        )
        mf = dft.UKS(mol)
        mf.xc = self.xc
        mf.grids.level = self.grid_level
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.verbose = 0
        mf.diis_space = 8
        mf.level_shift = 0.1
        energy_history: list[float] = []

        def callback(envs: dict[str, Any]) -> None:
            value = envs.get("e_tot")
            if value is not None and np.isfinite(value):
                energy_history.append(float(value) * HARTREE_TO_EV)

        mf.callback = callback
        if dm0 is None and seed_if_missing:
            dm0 = seed_broken_symmetry_guess(mf)
        energy_ha = float(mf.kernel(dm0=dm0))
        if not mf.converged:
            raise RuntimeError(
                f"UKS did not converge for 03b geometry; cycles={mf.cycles}, "
                f"last energy={energy_ha:.12f} Ha"
            )
        dm_alpha, dm_beta = (np.asarray(item, dtype=float) for item in mf.make_rdm1())
        gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
        forces = -gradient * GRAD_HA_BOHR_TO_EV_ANG
        ss, multiplicity = mf.spin_square()
        x, y, rho_alpha, rho_beta = evaluate_spin_density_plane(
            mol, dm_alpha, dm_beta, np.asarray(positions_ang), nx=96, ny=72
        )
        grid_x = grid_y = grid_z = rho_alpha_3d = rho_beta_3d = None
        if self.density_grid is not None:
            grid_x, grid_y, grid_z = (np.asarray(axis, dtype=float) for axis in self.density_grid)
            rho_alpha_3d, rho_beta_3d = evaluate_spin_density_grid(
                mol,
                dm_alpha,
                dm_beta,
                grid_x,
                grid_y,
                grid_z,
            )
        if not energy_history:
            energy_history = [energy_ha * HARTREE_TO_EV]
        energy_array = np.asarray(energy_history, dtype=float)
        residuals = np.empty_like(energy_array)
        residuals[0] = max(abs(energy_array[0] - energy_array[-1]), 1.0e-12)
        if len(energy_array) > 1:
            residuals[1:] = np.maximum(np.abs(np.diff(energy_array)), 1.0e-12)
        return UKSResult(
            energy_ev=energy_ha * HARTREE_TO_EV,
            forces_ev_ang=forces,
            dm_alpha=dm_alpha,
            dm_beta=dm_beta,
            scf_energies_ev=energy_array,
            scf_residuals=residuals,
            spin_square=float(ss),
            spin_multiplicity=float(multiplicity),
            spin_density_x=x,
            spin_density_y=y,
            rho_alpha=rho_alpha,
            rho_beta=rho_beta,
            converged=True,
            iterations=len(energy_array),
            branch_id=int(branch_id),
            density_grid_x=grid_x,
            density_grid_y=grid_y,
            density_grid_z=grid_z,
            rho_alpha_3d=rho_alpha_3d,
            rho_beta_3d=rho_beta_3d,
        )


def electron_count(elements: Iterable[str], charge: int = 0) -> int:
    """Return the molecular electron count for a neutral/charged formula."""

    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "Cl": 17}
    total = sum(atomic_numbers[str(element)] for element in elements)
    return int(total - int(charge))


def gaussian_density(
    x: np.ndarray,
    y: np.ndarray,
    centres: np.ndarray,
    widths: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Utility used by the explicitly labelled analytic demo backend."""

    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="xy")
    result = np.zeros_like(xx)
    for centre, width, amplitude in zip(centres, widths, amplitudes):
        dx = xx - float(centre[0])
        dy = yy - float(centre[1])
        sigma = max(float(width), 1.0e-6)
        result += float(amplitude) * np.exp(-0.5 * (dx * dx + dy * dy) / sigma**2)
    return result


def gaussian_density_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    centres: np.ndarray,
    widths: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Analytic 3-D Gaussian field used only by the labelled demo backend."""

    xx, yy, zz = np.meshgrid(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        indexing="ij",
    )
    result = np.zeros_like(xx, dtype=float)
    for centre, width, amplitude in zip(centres, widths, amplitudes):
        delta = (
            (xx - float(centre[0])) ** 2
            + (yy - float(centre[1])) ** 2
            + (zz - float(centre[2])) ** 2
        )
        sigma = max(float(width), 1.0e-6)
        result += float(amplitude) * np.exp(-0.5 * delta / sigma**2)
    return result

