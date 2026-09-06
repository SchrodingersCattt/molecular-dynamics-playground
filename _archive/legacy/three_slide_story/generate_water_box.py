"""Build a deterministic 64-water periodic box for the Deep Potential slide.

The box has the experimental room-temperature mass density (0.997 g/cm^3).
Oxygen sites start from a 4x4x4 grid with small deterministic displacements and
each rigid water receives an independent random 3D orientation.  It is a
prepared visualization structure, not a claim of an equilibrated trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.io import write
import numpy as np


AVOGADRO = 6.02214076e23
WATER_MOLAR_MASS = 18.01528
TARGET_DENSITY = 0.997
N_SIDE = 4
N_WATER = N_SIDE**3
BOND_LENGTH = 0.9572
ANGLE_DEG = 104.52
CUTOFF = 6.0


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


def build_box(seed: int = 20260826) -> dict[str, np.ndarray | float | int]:
    volume_angstrom3 = (
        N_WATER * WATER_MOLAR_MASS / AVOGADRO / TARGET_DENSITY * 1.0e24
    )
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

    rng = np.random.default_rng(seed)
    positions = []
    elements = []
    molecule_ids = []
    bonds = []
    oxygen_indices = []

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

    positions = np.asarray(positions)
    molecule_ids = np.asarray(molecule_ids)
    oxygen_indices = np.asarray(oxygen_indices)
    center = np.full(3, box_length / 2.0)
    selected_oxygen = oxygen_indices[
        np.argmin(np.linalg.norm(positions[oxygen_indices] - center, axis=1))
    ]

    shift = center - positions[selected_oxygen]
    shifted = positions + shift
    wrapped = shifted % box_length

    bond_view = np.empty_like(shifted)
    for molecule_id in range(N_WATER):
        indices = np.where(molecule_ids == molecule_id)[0]
        oxygen_index = indices[0]
        oxygen_wrapped = shifted[oxygen_index] % box_length
        molecule_offset = oxygen_wrapped - shifted[oxygen_index]
        bond_view[indices] = shifted[indices] + molecule_offset

    delta = wrapped - center
    delta -= box_length * np.rint(delta / box_length)
    neighbor_view = center + delta
    distances = np.linalg.norm(delta, axis=1)
    neighbor_mask = (distances <= CUTOFF) & (np.arange(len(distances)) != selected_oxygen)

    return {
        "box_length": box_length,
        "positions_wrapped": wrapped,
        "positions_bond_view": bond_view,
        "positions_neighbor_view": neighbor_view,
        "elements": np.asarray(elements),
        "molecule_ids": molecule_ids,
        "bonds": np.asarray(bonds, dtype=int),
        "central_index": int(selected_oxygen),
        "neighbor_mask": neighbor_mask,
        "neighbor_distances": distances,
        "cutoff": CUTOFF,
        "target_density": TARGET_DENSITY,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("three_slide_story/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_box()

    npz_path = args.output_dir / "water_box_64.npz"
    np.savez_compressed(npz_path, **data)

    atoms = Atoms(
        symbols=data["elements"].tolist(),
        positions=data["positions_wrapped"],
        cell=np.eye(3) * float(data["box_length"]),
        pbc=True,
    )
    atoms.arrays["molecule_id"] = np.asarray(data["molecule_ids"], dtype=int)
    atoms.arrays["source_id"] = np.arange(len(atoms), dtype=int)
    extxyz_path = args.output_dir / "water_box_64.extxyz"
    write(extxyz_path, atoms, format="extxyz")

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
        "seed": int(data["seed"]),
        "stable_id_field": "source_id",
        "molecule_id_field": "molecule_id",
    }
    metadata_path = args.output_dir / "water_box_64.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"structure: {extxyz_path}")
    print(f"arrays: {npz_path}")


if __name__ == "__main__":
    main()
