"""Broken-symmetry PBE0 UKS for the 03b TNT C–NO2 path.

The recorded kick ends at 2.40 Å, where the singlet is only partly spin-polarized.
Frames through the 2.20 Å mark are kept. The same outgoing velocity is then
continued until 3.45 Å, where the aryl and NO2 fragments are a radical pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/aisi-nas/guomingyu/personal/mlip-playground/261013_tnt_uks_bs")
NO2 = (12, 15, 16)
BREAKING_C = 1
BREAKING_N = 12
HEAD_INDEX = np.asarray(
    [0, 6, 11, 17, 22, 28, 33, 39, 44, 50, 56, 61, 67, 72, 78],
    dtype=int,
)
TAIL_TARGETS = np.linspace(2.35, 3.45, 10)
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
DT_FS = 0.1


def to_np(value):
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def ao_atoms(mol) -> np.ndarray:
    atoms = []
    for label in mol.ao_labels(fmt=False):
        if isinstance(label, (tuple, list)):
            atoms.append(int(label[0]))
        else:
            atoms.append(int(str(label).split()[0]))
    return np.asarray(atoms, dtype=int)


def r_cn(positions: np.ndarray) -> float:
    return float(np.linalg.norm(positions[BREAKING_N] - positions[BREAKING_C]))


def r_no(positions: np.ndarray) -> float:
    return float(np.linalg.norm(positions[15] - positions[BREAKING_N]))


def select_frames(positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    head_pos = np.asarray(positions[HEAD_INDEX], dtype=float)
    head_vel = np.asarray(velocities[HEAD_INDEX], dtype=float)
    if r_cn(head_pos[-2]) >= 2.195 or r_cn(head_pos[-1]) < 2.195:
        raise RuntimeError(
            f"Expected the retained path to cross 2.20 Å at the last kept frame, "
            f"got {r_cn(head_pos[-2]):.3f} then {r_cn(head_pos[-1]):.3f}"
        )
    pos = head_pos[-1].copy()
    vel = head_vel[-1].copy()
    tail_pos = []
    tail_vel = []
    for target in TAIL_TARGETS:
        steps = 0
        while r_cn(pos) < float(target) and steps < 500:
            pos = pos + vel * DT_FS
            steps += 1
        if steps == 0 or steps >= 500:
            raise RuntimeError(f"Could not reach r_CN={target:.2f} in {steps} steps")
        tail_pos.append(pos.copy())
        tail_vel.append(vel.copy())
    return (
        np.concatenate([head_pos, np.stack(tail_pos)], axis=0),
        np.concatenate([head_vel, np.stack(tail_vel)], axis=0),
    )


def make_mol(elements, coords, spin: int = 0):
    from pyscf import gto

    atom = [(str(element), np.asarray(coord, dtype=float).tolist()) for element, coord in zip(elements, coords)]
    return gto.M(
        atom=atom,
        basis="def2-svp",
        charge=0,
        spin=int(spin),
        unit="Angstrom",
        verbose=0,
    )


def make_uks(mol, level_shift: float):
    from gpu4pyscf.dft import uks

    mf = uks.UKS(mol)
    mf.xc = "pbe0"
    mf.grids.level = 2
    mf.conv_tol = 1.0e-6
    mf.max_cycle = 60
    mf.level_shift = float(level_shift)
    return mf


def fragment_guess(elements, coords) -> np.ndarray:
    aryl = tuple(i for i in range(len(elements)) if i not in NO2)
    dm_aryl = fragment_dm(elements, coords, aryl)
    dm_no2 = fragment_dm(elements, coords, NO2)
    mol = make_mol(elements, coords, spin=0)
    full_atoms = ao_atoms(mol)
    pools = {atom: np.flatnonzero(full_atoms == atom) for atom in range(len(elements))}
    aryl_idx = np.concatenate([pools[i] for i in aryl])
    no2_idx = np.concatenate([pools[i] for i in NO2])
    if len(aryl_idx) != dm_aryl.shape[-1] or len(no2_idx) != dm_no2.shape[-1]:
        raise RuntimeError("Fragment AO blocks do not match the full-molecule basis")
    nao = mol.nao
    dma = np.zeros((nao, nao))
    dmb = np.zeros((nao, nao))
    dma[np.ix_(aryl_idx, aryl_idx)] = dm_aryl[0]
    dmb[np.ix_(aryl_idx, aryl_idx)] = dm_aryl[1]
    dma[np.ix_(no2_idx, no2_idx)] = dm_no2[1]
    dmb[np.ix_(no2_idx, no2_idx)] = dm_no2[0]
    return np.stack([dma, dmb], axis=0)


def fragment_dm(elements, coords, atom_ids) -> np.ndarray:
    chosen_elements = [elements[i] for i in atom_ids]
    chosen_coords = coords[list(atom_ids)]
    mol = make_mol(chosen_elements, chosen_coords, spin=1)
    mf = make_uks(mol, 0.0)
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Fragment UKS did not converge, E={float(energy):.6f}, atoms={list(atom_ids)}")
    return to_np(mf.make_rdm1())


def solve(elements, coords, dm0, level_shift: float) -> dict[str, object]:
    mol = make_mol(elements, coords, spin=0)
    mf = make_uks(mol, level_shift)
    if dm0 is None:
        energy = mf.kernel()
    else:
        energy = mf.kernel(dm0=(np.asarray(dm0[0]), np.asarray(dm0[1])))
    ss, mult = mf.spin_square()
    grad = to_np(mf.nuc_grad_method().kernel())
    dm = to_np(mf.make_rdm1())
    overlap = to_np(mf.get_ovlp())
    spin = dm[0] - dm[1]
    atoms = ao_atoms(mol)
    pops = []
    for atom_id in (BREAKING_C, *NO2):
        mask = atoms == atom_id
        pops.append(float(np.einsum("ij,ji->", spin[np.ix_(mask, mask)], overlap[np.ix_(mask, mask)])))
    return {
        "dm": dm,
        "energy_ha": float(to_np(energy)),
        "s2": float(to_np(ss)),
        "mult": float(to_np(mult)),
        "forces": -grad * (HARTREE_TO_EV / BOHR_TO_ANG),
        "converged": bool(mf.converged),
        "pops": pops,
    }


def break_symmetry(elements, coords) -> dict[str, object]:
    guess = fragment_guess(elements, coords)
    held = solve(elements, coords, guess, 1.0)
    print(f"  level-shift hold S2={held['s2']:.4f} conv={held['converged']}", flush=True)
    return solve(elements, coords, held["dm"], 0.0)


def make_grid(positions: np.ndarray, spacing: float = 0.22, pad: float = 3.8):
    lo = positions.min(axis=(0, 1)) - pad
    hi = positions.max(axis=(0, 1)) + pad
    axes = []
    for axis in range(3):
        count = int(np.ceil((hi[axis] - lo[axis]) / spacing)) + 1
        axes.append(np.linspace(lo[axis], hi[axis], count))
    return tuple(axes)


def eval_density(elements, coords, dm, axes):
    from pyscf.dft import numint

    mol = make_mol(elements, coords, spin=0)
    gx, gy, gz = axes
    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])
    dz = float(gz[1] - gz[0])
    mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, indexing="ij")
    coords_bohr = np.stack((mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()), axis=1) * ANG_TO_BOHR
    ni = numint.NumInt()
    rho = np.empty((2, coords_bohr.shape[0]), dtype=np.float64)
    block = 32768
    for start in range(0, coords_bohr.shape[0], block):
        stop = min(start + block, coords_bohr.shape[0])
        ao = ni.eval_ao(mol, coords_bohr[start:stop], deriv=0)
        for spin in (0, 1):
            rho[spin, start:stop] = np.asarray(ni.eval_rho(mol, ao, dm[spin], xctype="LDA"), dtype=float)
    rho *= ANG_TO_BOHR**3
    rho = rho.reshape((2, mesh_x.shape[0], mesh_x.shape[1], mesh_x.shape[2]))
    electrons = rho.sum(axis=(1, 2, 3)) * dx * dy * dz
    return rho.astype(np.float32), electrons


def save_checkpoint(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)


def load_checkpoint(path: Path, n_frames: int, nao: int):
    if not path.exists():
        return None
    with np.load(path) as archive:
        if archive["done"].shape != (n_frames,) or archive["dms"].shape[-1] != nao:
            return None
        return {key: archive[key] for key in archive.files}


def write_outputs(root: Path, elements, positions, velocities, forces, energy_ha, s2, mult, converged, axes, rho):
    rcn = np.asarray([r_cn(frame) for frame in positions], dtype=float)
    rno = np.asarray([r_no(frame) for frame in positions], dtype=float)
    np.savez_compressed(
        root / "uks_tnt_reaction.npz",
        elements=np.asarray(elements),
        positions=positions,
        velocities=velocities,
        forces=forces,
        energy_ev=energy_ha * HARTREE_TO_EV,
        r_cn=rcn,
        r_no=rno,
        spin_square=s2,
        spin_multiplicity=mult,
        converged=converged,
        charge=np.asarray(0),
        spin=np.asarray(0),
        dt_fs=np.asarray(DT_FS),
    )
    np.savez_compressed(
        root / "uks_tnt_reaction_density3d.npz",
        rho_alpha_3d=rho[:, 0],
        rho_beta_3d=rho[:, 1],
        grid_x_ang=axes[0],
        grid_y_ang=axes[1],
        grid_z_ang=axes[2],
        positions_ang=positions,
        elements=np.asarray(elements),
        frame=np.arange(len(positions), dtype=int),
    )
    manifest = {
        "backend": "gpu4pyscf_uks",
        "method": "UKS/PBE0/def2-SVP",
        "charge": 0,
        "spin": 0,
        "frames": int(len(positions)),
        "head_source_indices": HEAD_INDEX.tolist(),
        "tail_r_cn_targets": [float(value) for value in TAIL_TARGETS],
        "continuation": "constant velocity from the 2.20 Å frame of job 20864421",
        "force_unit": "eV/Angstrom",
        "density_unit": "electron/Angstrom^3",
        "r_cn": [float(value) for value in rcn],
        "spin_square": [float(value) for value in s2],
    }
    (root / "uks_tnt_reaction.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(root: Path, input_path: Path, geometries_only: bool) -> None:
    data = np.load(input_path)
    elements = [str(item) for item in data["elements"]]
    positions, velocities = select_frames(data["positions"], data["velocities"])
    print(f"frames {len(positions)}", flush=True)
    for index, frame in enumerate(positions):
        print(f"geom {index:02d} r_cn={r_cn(frame):.3f} r_no={r_no(frame):.3f}", flush=True)
    if geometries_only:
        return

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    probe = make_mol(elements, positions[0])
    nao = int(probe.nao)
    n_frames = len(positions)
    n_head = len(HEAD_INDEX)
    checkpoint_path = root / "scf_checkpoint.npz"
    restored = load_checkpoint(checkpoint_path, n_frames, nao)
    if restored is None:
        done = np.zeros(n_frames, dtype=bool)
        energy = np.zeros(n_frames)
        s2 = np.zeros(n_frames)
        mult = np.zeros(n_frames)
        forces = np.zeros((n_frames, len(elements), 3))
        converged = np.zeros(n_frames, dtype=bool)
        dms = np.zeros((n_frames, 2, nao, nao))
    else:
        done = restored["done"]
        energy = restored["energy_ha"]
        s2 = restored["s2"]
        mult = restored["mult"]
        forces = restored["forces"]
        converged = restored["converged"]
        dms = restored["dms"]
        print(f"resume {int(done.sum())}/{n_frames}", flush=True)

    def store(index: int, result: dict[str, object], mode: str, started: float) -> None:
        done[index] = True
        energy[index] = float(result["energy_ha"])
        s2[index] = float(result["s2"])
        mult[index] = float(result["mult"])
        forces[index] = np.asarray(result["forces"], dtype=float)
        converged[index] = bool(result["converged"])
        dms[index] = np.asarray(result["dm"], dtype=float)
        pops = result["pops"]
        print(
            f"frame {index:02d} r_cn={r_cn(positions[index]):.3f} mode={mode} "
            f"S2={s2[index]:.4f} conv={converged[index]} "
            f"pop C2={pops[0]:+.3f} N={pops[1]:+.3f} O={pops[2]:+.3f},{pops[3]:+.3f} "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        save_checkpoint(
            checkpoint_path,
            {
                "done": done,
                "energy_ha": energy,
                "s2": s2,
                "mult": mult,
                "forces": forces,
                "converged": converged,
                "dms": dms,
            },
        )

    for index in range(n_head):
        if done[index]:
            continue
        started = time.perf_counter()
        dm0 = dms[index - 1] if index and done[index - 1] else None
        store(index, solve(elements, positions[index], dm0, 0.0), "closed", started)

    last = n_frames - 1
    if not done[last]:
        started = time.perf_counter()
        store(last, break_symmetry(elements, positions[last]), "fragment", started)

    for index in range(n_frames - 2, n_head - 1, -1):
        if done[index]:
            continue
        started = time.perf_counter()
        result = solve(elements, positions[index], dms[index + 1], 0.0)
        mode = "warm"
        if result["s2"] < 0.05 and r_cn(positions[index]) >= 2.30:
            result = break_symmetry(elements, positions[index])
            mode = "fragment"
        store(index, result, mode, started)

    if not bool(done.all()):
        raise RuntimeError("SCF loop finished with missing frames")

    density_path = root / "uks_tnt_reaction_density3d.npz"
    axes = make_grid(positions)
    print(
        f"grid {[len(axis) for axis in axes]} points {int(np.prod([len(axis) for axis in axes]))}",
        flush=True,
    )
    rho = np.zeros((n_frames, 2, *(len(axis) for axis in axes)), dtype=np.float32)
    for index in range(n_frames):
        started = time.perf_counter()
        field, electrons = eval_density(elements, positions[index], dms[index], axes)
        total = float(electrons.sum())
        # 0.22 Å quadrature misses or overshoots the nuclear cusps, so the
        # integral wanders around 116.  A Bohr/Angstrom unit error would land
        # near 17 or 780 instead.
        if not 90.0 < total < 150.0:
            raise RuntimeError(f"Frame {index} integrates to {total:.2f} electrons, expected 116")
        rho[index] = field
        print(
            f"density {index:02d} ne={electrons[0]:.2f},{electrons[1]:.2f} "
            f"max={float(field.max()):.3f} max|a-b|={float(np.max(np.abs(field[0] - field[1]))):.4f} "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    write_outputs(root, elements, positions, velocities, forces, energy, s2, mult, converged, axes, rho)
    print("wrote uks_tnt_reaction.npz and uks_tnt_reaction_density3d.npz", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--geometries-only", action="store_true")
    args = parser.parse_args()
    incoming = args.input if args.input is not None else args.root / "input.npz"
    run(args.root, incoming, args.geometries_only)


if __name__ == "__main__":
    main()
