"""Generate the 03b UKS TNT C2--NO2 dissociation dataset.

The real path is a PySCF UKS/PBE0/def2-SVP Born--Oppenheimer trajectory.  The
``--demo`` path is retained only for local renderer regression and is labelled
as an analytic surrogate in every manifest.  A fixed Cartesian 3-D grid is
saved for every accepted ionic frame so later MatterVis/Cube plots use the
same coordinates as the nuclei.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "product" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run_md"))

from engine_md import CONV_ACCEL  # noqa: E402
from engine_uks import (  # noqa: E402
    PySCFUnavailable,
    UKSEngine,
    UKSResult,
    electron_count,
    gaussian_density,
    gaussian_density_3d,
    make_molecule,
    pyscf_available,
)


STEM = "03b_uks_reaction"
DATA_STEM = "uks_tnt_reaction"
ELEMENTS = np.asarray(
    [
        "C", "C", "C", "C", "C", "C", "C",  # ring + methyl carbon
        "H", "H", "H", "H", "H",                    # ring/methyl H
        "N", "N", "N", "O", "O", "O", "O", "O", "O",
    ]
)
CHARGE = 0
SPIN = 0
MASSES = np.asarray(
    [12.011] * 7 + [1.008] * 5 + [14.007] * 3 + [15.999] * 6,
    dtype=float,
)
RING_ATOMS = np.arange(6, dtype=int)
METHYL_C = 6
RING_H = np.asarray([7, 8], dtype=int)
METHYL_H = np.asarray([9, 10, 11], dtype=int)
NITRO_N = np.asarray([12, 13, 14], dtype=int)
NITRO_O = np.asarray([[15, 16], [17, 18], [19, 20]], dtype=int)
BREAKING_C = 1  # C2 when the methyl-bearing ring carbon is C1.
BREAKING_N = int(NITRO_N[0])
NO2_ATOMS = np.asarray([BREAKING_N, *NITRO_O[0].tolist()], dtype=int)
ARYL_ATOMS = np.asarray([i for i in range(len(ELEMENTS)) if i not in set(NO2_ATOMS)], dtype=int)
DT_FS = 0.10
N_STEPS = 100
RELATIVE_SPEED = 0.05

# A moderate fixed grid: axes are Angstrom, values are electron/Angstrom^3.
GRID_SPACING = 0.22
ANG_TO_BOHR = 1.0 / 0.529177210903
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}


def initial_geometry() -> np.ndarray:
    """Generate 2,4,6-TNT from a topology-safe SMILES and return project order.

    RDKit is deliberately used for the initial 3-D embedding so nitro oxygens
    cannot be accidentally placed close enough to trigger a spurious O--O bond
    in MatterVis.  PBE0/def2-SVP optimization remains the scientific geometry
    used by the real Bohrium UKS run.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TNT generation requires RDKit for topology-safe SMILES embedding") from exc

    smiles = "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if molecule is None or molecule.GetNumAtoms() != 21:
        raise RuntimeError("RDKit failed to build the 21-atom C7H5N3O6 TNT molecule")
    status = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True)
    if status != 0:
        raise RuntimeError(f"RDKit TNT embedding failed with status {status}")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
    conformer = molecule.GetConformer()
    rdkit_positions = np.asarray(conformer.GetPositions(), dtype=float)
    # RDKit atom order for the explicit SMILES: methyl C, ring C1/C2/C3/C4/C5/C6,
    # three nitro N/O groups, then H.  Reorder to the stable project map.
    order = [1, 2, 6, 7, 11, 12, 0, 19, 20, 16, 17, 18, 3, 8, 13, 4, 5, 9, 10, 14, 15]
    return rdkit_positions[np.asarray(order, dtype=int)]


def optimize_geometry() -> np.ndarray:
    """Optimize the fixed atom ordering with PySCF/geomeTRIC on Bohrium."""

    if not pyscf_available():
        raise PySCFUnavailable("PySCF is required for the TNT PBE0/def2-SVP geometry optimization")
    try:
        from pyscf import dft
        from pyscf.geomopt.geometric_solver import optimize
    except Exception as exc:  # pragma: no cover - depends on Bohrium image
        raise PySCFUnavailable("PySCF geomeTRIC geometry optimization is unavailable") from exc
    mol = make_molecule(initial_geometry(), ELEMENTS, charge=CHARGE, spin=SPIN, basis="def2-svp")
    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 2
    mf.conv_tol = 1.0e-9
    mf.verbose = 0
    mf.kernel()
    optimized = optimize(mf, maxsteps=100)
    return np.asarray(optimized.atom_coords(unit="Angstrom"), dtype=float)


def reaction_metrics(positions: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    r_cn = float(np.linalg.norm(positions[BREAKING_N] - positions[BREAKING_C]))
    r_no = float(np.linalg.norm(positions[NITRO_O[0, 0]] - positions[BREAKING_N]))
    aryl_centre = positions[ARYL_ATOMS].mean(axis=0)
    no2_centre = positions[NO2_ATOMS].mean(axis=0)
    direction = aryl_centre - no2_centre
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1.0e-12 else np.asarray([1.0, 0.0, 0.0])
    return r_cn, r_no, no2_centre, direction


def kick_velocities(positions: np.ndarray, relative_speed: float) -> np.ndarray:
    _, _, _, direction = reaction_metrics(positions)
    m_aryl = float(MASSES[ARYL_ATOMS].sum())
    m_no2 = float(MASSES[NO2_ATOMS].sum())
    total = m_aryl + m_no2
    velocities = np.zeros_like(positions, dtype=float)
    velocities[ARYL_ATOMS] = direction * float(relative_speed) * m_no2 / total
    velocities[NO2_ATOMS] = -direction * float(relative_speed) * m_aryl / total
    velocities -= (MASSES[:, None] * velocities).sum(axis=0) / MASSES.sum()
    return velocities


def density_grid(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    max_travel = float(RELATIVE_SPEED * DT_FS * N_STEPS) + 1.0
    lo = np.min(positions, axis=0) - (3.8 + max_travel)
    hi = np.max(positions, axis=0) + (3.8 + max_travel)
    axes = []
    for lower, upper in zip(lo, hi):
        count = int(np.ceil((upper - lower) / GRID_SPACING)) + 1
        axes.append(np.linspace(lower, upper, count, dtype=float))
    return tuple(axes)  # type: ignore[return-value]


def write_xyz(path: Path, positions: np.ndarray, *, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(ELEMENTS)), comment]
    lines.extend(
        f"{element} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}"
        for element, coord in zip(ELEMENTS, np.asarray(positions, dtype=float))
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cube(
    path: Path,
    positions: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    *,
    title: str,
) -> None:
    """Write one scalar field in Gaussian Cube convention."""

    x, y, z = (np.asarray(axis, dtype=float) for axis in grid)
    field = np.asarray(values, dtype=float) / (0.529177210903**3)
    if field.shape != (len(x), len(y), len(z)):
        raise ValueError(f"Cube field shape {field.shape} does not match grid {(len(x), len(y), len(z))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    origin = np.asarray([x[0], y[0], z[0]], dtype=float) * ANG_TO_BOHR
    axes = np.asarray(
        [
            [x[1] - x[0], 0.0, 0.0],
            [0.0, y[1] - y[0], 0.0],
            [0.0, 0.0, z[1] - z[0]],
        ],
        dtype=float,
    ) * ANG_TO_BOHR
    lines = [title, "TNT UKS density; coordinates and grid are in the same frame"]
    lines.append(f"{len(ELEMENTS):5d} {origin[0]: .8f} {origin[1]: .8f} {origin[2]: .8f}")
    for count, axis in zip(field.shape, axes):
        lines.append(f"{count:5d} {axis[0]: .8f} {axis[1]: .8f} {axis[2]: .8f}")
    for element, coord in zip(ELEMENTS, np.asarray(positions, dtype=float)):
        xyz = np.asarray(coord) * ANG_TO_BOHR
        lines.append(f"{ATOMIC_NUMBERS[str(element)]:5d} 0.00000000 {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}")
    flat = field.ravel(order="C")
    for start in range(0, len(flat), 6):
        lines.append(" ".join(f"{value: .8E}" for value in flat[start : start + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bond_energy(positions: np.ndarray, i: int, j: int, k: float, r0: float) -> tuple[float, np.ndarray]:
    delta = positions[j] - positions[i]
    distance = float(np.linalg.norm(delta))
    unit = delta / max(distance, 1.0e-12)
    dr = distance - r0
    forces = np.zeros_like(positions)
    forces[i] += k * dr * unit
    forces[j] -= k * dr * unit
    return 0.5 * k * dr * dr, forces


def analytic_demo_evaluate(
    positions: np.ndarray,
    *,
    step: int,
    branch_id: int = 0,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> UKSResult:
    """Reaction-shaped surrogate for local renderer and density QA only."""

    positions = np.asarray(positions, dtype=float)
    r_cn, r_no, no2_centre, _ = reaction_metrics(positions)
    reference = initial_geometry()
    energy = -100.0
    forces = np.zeros_like(positions)
    cn0 = float(np.linalg.norm(reference[BREAKING_N] - reference[BREAKING_C]))
    value, contribution = _bond_energy(positions, BREAKING_C, BREAKING_N, 8.0, cn0)
    energy += value
    forces += contribution
    bonds = [(int(i), int(j)) for i, j in zip(RING_ATOMS, np.roll(RING_ATOMS, -1))]
    bonds += [(0, METHYL_C), (2, RING_H[0]), (4, RING_H[1])]
    bonds += [(METHYL_C, int(h)) for h in METHYL_H]
    for group, ring_index in enumerate((1, 3, 5)):
        bonds.append((ring_index, int(NITRO_N[group])))
        bonds.extend((int(NITRO_N[group]), int(o)) for o in NITRO_O[group])
    for i, j in bonds:
        r0 = float(np.linalg.norm(reference[j] - reference[i]))
        value, contribution = _bond_energy(positions, i, j, 7.0 if (i, j) != (BREAKING_C, BREAKING_N) else 2.0, r0)
        energy += value
        forces += contribution

    x = np.linspace(-8.0, 8.0, 128)
    y = np.linspace(-7.0, 7.0, 96)
    rho_alpha = gaussian_density(x, y, positions[:, :2], np.full(len(ELEMENTS), 0.38), np.full(len(ELEMENTS), 0.35))
    separation = float(np.clip((r_cn - cn0) / 2.4, 0.0, 1.0))
    spin = gaussian_density(x, y, no2_centre[None, :2], np.asarray([0.55]), np.asarray([0.55 * separation]))
    rho_beta = rho_alpha.copy()
    rho_alpha += 0.5 * spin
    rho_beta -= 0.5 * spin
    rho_alpha_3d = rho_beta_3d = None
    gx = gy = gz = None
    if grid is not None:
        gx, gy, gz = grid
        rho_alpha_3d = gaussian_density_3d(gx, gy, gz, positions, np.full(len(ELEMENTS), 0.38), np.full(len(ELEMENTS), 0.35))
        spin3 = gaussian_density_3d(gx, gy, gz, no2_centre[None, :], np.asarray([0.55]), np.asarray([0.55 * separation]))
        rho_beta_3d = rho_alpha_3d.copy()
        rho_alpha_3d += 0.5 * spin3
        rho_beta_3d -= 0.5 * spin3
    iterations = 10 + int(4 * separation) + int(step % 3 == 0)
    residuals = np.geomspace(1.0e-2, 1.0e-9, iterations)
    scf_energies = np.linspace(energy + 0.02, energy, iterations)
    return UKSResult(
        energy_ev=float(energy),
        forces_ev_ang=forces,
        dm_alpha=np.zeros((1, 1)),
        dm_beta=np.zeros((1, 1)),
        scf_energies_ev=scf_energies,
        scf_residuals=residuals,
        spin_square=float(separation**2),
        spin_multiplicity=float(1.0 + separation),
        spin_density_x=x,
        spin_density_y=y,
        rho_alpha=rho_alpha,
        rho_beta=rho_beta,
        converged=True,
        iterations=iterations,
        branch_id=int(branch_id),
        density_grid_x=gx,
        density_grid_y=gy,
        density_grid_z=gz,
        rho_alpha_3d=rho_alpha_3d,
        rho_beta_3d=rho_beta_3d,
    )


def run_trajectory(
    evaluator: Callable[[np.ndarray, int, int], UKSResult],
    *,
    geometry: np.ndarray,
    relative_speed: float,
    n_steps: int,
    dt_fs: float,
) -> list[dict[str, object]]:
    positions = np.asarray(geometry, dtype=float)
    velocities = kick_velocities(positions, relative_speed)
    frames: list[dict[str, object]] = []
    branch_id = 0
    result = evaluator(positions, 0, branch_id)
    for step in range(int(n_steps) + 1):
        r_cn, r_no, _, _ = reaction_metrics(positions)
        frames.append(
            {
                "step": step,
                "positions": positions.copy(),
                "velocities": velocities.copy(),
                "forces": np.asarray(result.forces_ev_ang, dtype=float),
                "energy_ev": float(result.energy_ev),
                "r_cn": r_cn,
                "r_no": r_no,
                "scf_energies_ev": np.asarray(result.scf_energies_ev, dtype=float),
                "scf_residuals": np.asarray(result.scf_residuals, dtype=float),
                "spin_square": float(result.spin_square),
                "spin_multiplicity": float(result.spin_multiplicity),
                "spin_density_x": np.asarray(result.spin_density_x, dtype=float),
                "spin_density_y": np.asarray(result.spin_density_y, dtype=float),
                "rho_alpha": np.asarray(result.rho_alpha, dtype=float),
                "rho_beta": np.asarray(result.rho_beta, dtype=float),
                "dm_alpha": np.asarray(result.dm_alpha, dtype=float),
                "dm_beta": np.asarray(result.dm_beta, dtype=float),
                "rho_alpha_3d": None if result.rho_alpha_3d is None else np.asarray(result.rho_alpha_3d, dtype=np.float32),
                "rho_beta_3d": None if result.rho_beta_3d is None else np.asarray(result.rho_beta_3d, dtype=np.float32),
                "converged": bool(result.converged),
                "branch_id": int(result.branch_id),
            }
        )
        if step == int(n_steps):
            break
        acceleration = result.forces_ev_ang * CONV_ACCEL / MASSES[:, None]
        velocity_half = velocities + 0.5 * acceleration * dt_fs
        next_positions = positions + velocity_half * dt_fs
        try:
            result_next = evaluator(next_positions, step + 1, branch_id)
        except RuntimeError:
            branch_id += 1
            result_next = evaluator(next_positions, step + 1, branch_id)
        acceleration_next = result_next.forces_ev_ang * CONV_ACCEL / MASSES[:, None]
        velocities = velocity_half + 0.5 * acceleration_next * dt_fs
        positions = next_positions
        result = result_next
    return frames


def _pad(values: list[np.ndarray]) -> np.ndarray:
    width = max(len(value) for value in values)
    output = np.full((len(values), width), np.nan, dtype=float)
    for index, value in enumerate(values):
        output[index, : len(value)] = value
    return output


def save_dataset(frames: list[dict[str, object]], *, backend: str, geometry: np.ndarray, grid: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.asarray([frame["positions"] for frame in frames], dtype=float)
    velocities = np.asarray([frame["velocities"] for frame in frames], dtype=float)
    forces = np.asarray([frame["forces"] for frame in frames], dtype=float)
    rho_alpha_3d = np.asarray([frame["rho_alpha_3d"] for frame in frames], dtype=np.float32)
    rho_beta_3d = np.asarray([frame["rho_beta_3d"] for frame in frames], dtype=np.float32)
    dm_alpha = np.asarray([frame["dm_alpha"] for frame in frames], dtype=np.float32)
    dm_beta = np.asarray([frame["dm_beta"] for frame in frames], dtype=np.float32)
    output = DATA_DIR / f"{DATA_STEM}.npz"
    np.savez_compressed(
        output,
        elements=ELEMENTS,
        positions=positions,
        velocities=velocities,
        forces=forces,
        energy_ev=np.asarray([frame["energy_ev"] for frame in frames]),
        r_cn=np.asarray([frame["r_cn"] for frame in frames]),
        r_no=np.asarray([frame["r_no"] for frame in frames]),
        spin_square=np.asarray([frame["spin_square"] for frame in frames]),
        spin_multiplicity=np.asarray([frame["spin_multiplicity"] for frame in frames]),
        scf_energies_ev=_pad([np.asarray(frame["scf_energies_ev"]) for frame in frames]),
        scf_residuals=_pad([np.asarray(frame["scf_residuals"]) for frame in frames]),
        rho_alpha=np.asarray([frame["rho_alpha"] for frame in frames]),
        rho_beta=np.asarray([frame["rho_beta"] for frame in frames]),
        spin_density_x=np.asarray(frames[0]["spin_density_x"]),
        spin_density_y=np.asarray(frames[0]["spin_density_y"]),
        charge=np.asarray(CHARGE),
        spin=np.asarray(SPIN),
        dt_fs=np.asarray(DT_FS),
    )
    density_output = DATA_DIR / f"{DATA_STEM}_density3d.npz"
    np.savez_compressed(
        density_output,
        rho_alpha_3d=rho_alpha_3d,
        rho_beta_3d=rho_beta_3d,
        dm_alpha=dm_alpha,
        dm_beta=dm_beta,
        grid_x_ang=np.asarray(grid[0]),
        grid_y_ang=np.asarray(grid[1]),
        grid_z_ang=np.asarray(grid[2]),
        positions_ang=positions,
        elements=ELEMENTS,
        frame=np.arange(len(frames), dtype=int),
    )
    cube_dir = REPO_ROOT / "product" / "qa" / "03b_uks_reaction" / "source" / "density3d"
    cube_indices = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    cube_outputs = []
    for frame_index in cube_indices:
        alpha_path = cube_dir / f"frame_{frame_index:04d}_alpha.cube"
        beta_path = cube_dir / f"frame_{frame_index:04d}_beta.cube"
        write_cube(alpha_path, positions[frame_index], grid, rho_alpha_3d[frame_index], title=f"TNT alpha density frame {frame_index}")
        write_cube(beta_path, positions[frame_index], grid, rho_beta_3d[frame_index], title=f"TNT beta density frame {frame_index}")
        cube_outputs.extend([str(alpha_path), str(beta_path)])
    manifest = {
        "schema_version": 1,
        "stem": STEM,
        "data_stem": DATA_STEM,
        "backend": backend,
        "method": "UKS/PBE0/def2-SVP" if backend == "pyscf_uks" else "analytic surrogate for visual QA",
        "elements": ELEMENTS.tolist(),
        "formula": "C7H5N3O6",
        "smiles": "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
        "atom_map": {
            "ring_c1_c6": RING_ATOMS.tolist(),
            "methyl_c": int(METHYL_C),
            "ring_h": RING_H.tolist(),
            "methyl_h": METHYL_H.tolist(),
            "nitro_n": NITRO_N.tolist(),
            "nitro_o": NITRO_O.tolist(),
        },
        "charge": CHARGE,
        "spin": SPIN,
        "electron_count": electron_count(ELEMENTS, CHARGE),
        "reaction": "TNT -> aryl radical + NO2 radical (C2-NO2 homolytic dissociation)",
        "breaking_bond": [BREAKING_C, BREAKING_N],
        "no2_atoms": NO2_ATOMS.tolist(),
        "aryl_atoms": ARYL_ATOMS.tolist(),
        "geometry_source": "PBE0/def2-SVP RKS optimization from deterministic TNT reference geometry",
        "dt_fs": DT_FS,
        "n_steps": len(frames) - 1,
        "relative_speed_ang_fs": RELATIVE_SPEED,
        "density3d": {
            "npz": str(density_output),
            "representative_cubes": cube_outputs,
            "units": "electron/angstrom^3",
            "grid_order": "x,y,z",
            "shape": [len(grid[0]), len(grid[1]), len(grid[2])],
            "spacing_ang": GRID_SPACING,
        },
        "npz": str(output),
        "source_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "density3d_sha256": hashlib.sha256(density_output.read_bytes()).hexdigest(),
        "notes": [
            "The analytic backend is for renderer/layout QA only; it is not UKS data."
        ] if backend != "pyscf_uks" else [
            "Alpha/beta 3-D density is saved on one fixed Cartesian grid for every accepted ionic frame.",
            "Energetics are qualitative and are not a production barrier or rate calculation.",
        ],
    }
    (DATA_DIR / f"{DATA_STEM}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--speed", type=float, default=RELATIVE_SPEED)
    args = parser.parse_args()
    output = DATA_DIR / f"{DATA_STEM}.npz"
    if output.exists() and not args.force:
        print(f"[INFO] {output} exists; use --force to regenerate")
        return
    geometry = initial_geometry() if args.demo else optimize_geometry()
    write_xyz(
        REPO_ROOT / "product" / "qa" / "03b_uks_reaction" / "source" / "tnt_optimized.xyz",
        geometry,
        comment="2,4,6-TNT PBE0/def2-SVP geometry; C2-N2 selected for dissociation",
    )
    grid = density_grid(geometry)
    if args.demo:
        def evaluator(positions: np.ndarray, step: int, branch: int) -> UKSResult:
            return analytic_demo_evaluate(positions, step=step, branch_id=branch, grid=grid)
        backend = "analytic_demo"
    else:
        engine = UKSEngine(
            ELEMENTS,
            charge=CHARGE,
            spin=SPIN,
            basis="def2-svp",
            xc="pbe0",
            density_grid=grid,
        )
        state: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"dm": None}

        def evaluator(positions: np.ndarray, step: int, branch: int) -> UKSResult:
            result = engine.evaluate(
                positions,
                dm0=state["dm"],
                branch_id=branch,
                seed_if_missing=state["dm"] is None,
            )
            state["dm"] = result.density_matrix
            return result
        backend = "pyscf_uks"
    frames = run_trajectory(
        evaluator,
        geometry=geometry,
        relative_speed=args.speed,
        n_steps=args.steps,
        dt_fs=DT_FS,
    )
    save_dataset(frames, backend=backend, geometry=geometry, grid=grid)
    print(f"Saved {len(frames)} frames ({backend}) -> {output}")


if __name__ == "__main__":
    main()
