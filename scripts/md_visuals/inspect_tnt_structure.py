"""Run the mandatory mat-vis structure preflight for the TNT C2--NO2 model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read


EXPECTED_FORMULA = "C7H5N3O6"
BREAKING_C = 1
BREAKING_N = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    args = parser.parse_args()

    atoms = read(args.structure, index=0)
    formula = atoms.get_chemical_formula(empirical=False)
    if formula != EXPECTED_FORMULA:
        raise SystemExit(f"Unexpected TNT formula: {formula} != {EXPECTED_FORMULA}")
    if len(atoms) != 21:
        raise SystemExit(f"Unexpected TNT atom count: {len(atoms)} != 21")
    cn_distance = float(np.linalg.norm(atoms.positions[BREAKING_N] - atoms.positions[BREAKING_C]))
    if not 1.15 <= cn_distance <= 1.90:
        raise SystemExit(f"C2-N2 distance is outside the preflight range: {cn_distance:.4f} A")
    oxygen = [index for index, symbol in enumerate(atoms.symbols) if symbol == "O"]
    oo_distances = [
        float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
        for offset, i in enumerate(oxygen)
        for j in oxygen[offset + 1 :]
    ]
    min_oo_distance = min(oo_distances)
    if min_oo_distance <= 1.80:
        raise SystemExit(f"Spurious O-O contact detected: {min_oo_distance:.4f} A")

    inspect = subprocess.run(
        ["mat-vis", "inspect", str(args.structure), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(inspect.stdout)
    if not payload.get("ok"):
        raise SystemExit("mat-vis inspect returned ok=false")
    args.render.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "mat-vis", "render", str(args.structure), "-o", str(args.render),
            "--backend", "cpu", "--style", "ball_stick", "--no-cell",
            "--orthogonal", "--json",
        ],
        check=True,
    )
    result = {
        "structure": str(args.structure),
        "formula": formula,
        "atom_count": len(atoms),
        "breaking_bond": [BREAKING_C, BREAKING_N],
        "breaking_bond_distance_angstrom": cn_distance,
        "minimum_oxygen_oxygen_distance_angstrom": min_oo_distance,
        "mat_vis_inspect": payload,
        "render": str(args.render),
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
