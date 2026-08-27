"""Generate the deterministic 64-water periodic box used by the DPMD story."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


DATA_DIR = Path(__file__).resolve().parent / "data"
AVOGADRO = 6.02214076e23
WATER_MOLAR_MASS = 18.01528
TARGET_DENSITY = 0.997
N_SIDE = 4
N_WATER = N_SIDE**3
BOND_LENGTH = 0.9572
ANGLE_DEG = 104.52
CUTOFF = 6.0
SEED = 20260826


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_box() -> dict[str, np.ndarray | float | int]:
    volume_angstrom3 = N_WATER * WATER_MOLAR_MASS / AVOGADRO / TARGET_DENSITY * 1.0e24
    box_length = volume_angstrom3 ** (1.0 / 3.0)
    spacing = box_length / N_SIDE
    angle = np.deg2rad(ANGLE_DEG)
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [BOND_LENGTH, 0.0, 0.0],
            [BOND_LENGTH * np.cos(angle), BOND_LENGTH * np.sin(angle), 0.0],
        ]
    )
    rng = np.random.default_rng(SEED)
    positions: list[np.ndarray] = []
    elements: list[str] = []
    molecule_ids: list[int] = []
    bonds: list[tuple[int, int]] = []
    oxygen_indices: list[int] = []
    molecule = 0
    for ix in range(N_SIDE):
        for iy in range(N_SIDE):
            for iz in range(N_SIDE):
                oxygen = spacing * (np.array([ix, iy, iz], dtype=float) + 0.5)
                oxygen += rng.normal(scale=0.08, size=3)
                rotated = local @ random_rotation(rng).T + oxygen
                start = len(positions)
                positions.extend(rotated)
                elements.extend(["O", "H", "H"])
                molecule_ids.extend([molecule] * 3)
                bonds.extend([(start, start + 1), (start, start + 2)])
                oxygen_indices.append(start)
                molecule += 1

    positions_array = np.asarray(positions)
    molecule_ids_array = np.asarray(molecule_ids, dtype=int)
    oxygen_indices_array = np.asarray(oxygen_indices, dtype=int)
    centre = np.full(3, box_length / 2.0)
    central_index = int(
        oxygen_indices_array[
            np.argmin(np.linalg.norm(positions_array[oxygen_indices_array] - centre, axis=1))
        ]
    )
    shifted = positions_array + (centre - positions_array[central_index])
    wrapped = shifted % box_length

    bond_view = np.empty_like(shifted)
    for molecule_id in range(N_WATER):
        indices = np.where(molecule_ids_array == molecule_id)[0]
        oxygen_index = int(indices[0])
        molecule_offset = shifted[oxygen_index] % box_length - shifted[oxygen_index]
        bond_view[indices] = shifted[indices] + molecule_offset

    delta = wrapped - centre
    delta -= box_length * np.rint(delta / box_length)
    neighbour_view = centre + delta
    distances = np.linalg.norm(delta, axis=1)
    neighbour_mask = (distances <= CUTOFF) & (np.arange(len(distances)) != central_index)
    return {
        "box_length": float(box_length),
        "positions_wrapped": wrapped,
        "positions_bond_view": bond_view,
        "positions_neighbor_view": neighbour_view,
        "elements": np.asarray(elements),
        "molecule_ids": molecule_ids_array,
        "source_ids": np.arange(len(positions_array), dtype=int),
        "bonds": np.asarray(bonds, dtype=int),
        "central_index": central_index,
        "neighbor_mask": neighbour_mask,
        "neighbor_distances": distances,
        "cutoff": CUTOFF,
        "target_density": TARGET_DENSITY,
        "seed": SEED,
    }


def main() -> None:
    data = build_box()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "water_box_64.npz", **data)
    atoms = Atoms(
        symbols=data["elements"].tolist(),
        positions=data["positions_wrapped"],
        cell=np.eye(3) * float(data["box_length"]),
        pbc=True,
    )
    atoms.arrays["molecule_id"] = np.asarray(data["molecule_ids"], dtype=int)
    atoms.arrays["source_id"] = np.asarray(data["source_ids"], dtype=int)
    write(DATA_DIR / "water_box_64.extxyz", atoms, format="extxyz")
    metadata = {
        "construction": "64 rigid H2O molecules on a perturbed 4x4x4 oxygen grid with deterministic random orientations",
        "equilibration_status": "prepared visualization structure; not MD-equilibrated",
        "n_water": N_WATER,
        "n_atoms": len(atoms),
        "box_length_angstrom": float(data["box_length"]),
        "density_g_cm3": TARGET_DENSITY,
        "cutoff_angstrom": CUTOFF,
        "central_source_id": int(data["central_index"]),
        "neighbor_atom_count": int(np.count_nonzero(data["neighbor_mask"])),
        "seed": SEED,
        "stable_id_field": "source_id",
        "molecule_id_field": "molecule_id",
    }
    (DATA_DIR / "water_box_64.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
