"""Evaluate the prepared 64-water periodic box with the bundled Deep Potential."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="H2O-Phase-Diagram-model_compressed.pb")
    parser.add_argument("--input", default="water_box_64.npz")
    parser.add_argument("--output", default="dpmd_water_box_results.npz")
    parser.add_argument("--metadata", default="dpmd_eval.json")
    args = parser.parse_args()

    from deepmd.infer import DeepPot

    with np.load(args.input) as data:
        positions = np.asarray(data["positions_wrapped"], dtype=float)
        elements = np.asarray(data["elements"]).astype(str)
        box_length = float(data["box_length"])
        cutoff = float(data["cutoff"])

    atom_types = np.asarray([0 if element == "O" else 1 for element in elements], dtype=int)
    coordinates = positions.reshape(1, len(positions), 3)
    cells = (np.eye(3) * box_length).reshape(1, 9)
    potential = DeepPot(args.model)

    try:
        energy, forces, virial, atomic_energy, atomic_virial = potential.eval(
            coordinates,
            cells=cells,
            atom_types=atom_types,
            atomic=True,
        )
        atomic_energy = np.asarray(atomic_energy).reshape(len(positions))
    except TypeError:
        energy, forces, virial = potential.eval(
            coordinates,
            cells=cells,
            atom_types=atom_types,
        )
        atomic_energy = np.full(len(positions), float(np.asarray(energy).reshape(-1)[0]) / len(positions))

    total_energy = float(np.asarray(energy).reshape(-1)[0])
    forces = np.asarray(forces).reshape(len(positions), 3)
    virial = np.asarray(virial).reshape(3, 3)
    np.savez_compressed(
        args.output,
        elements=elements,
        positions=positions,
        box_length=np.array(box_length),
        total_energy_ev=np.array(total_energy),
        atomic_energy_ev=atomic_energy,
        forces_ev_per_angstrom=forces,
        virial_ev=virial,
        cutoff_angstrom=np.array(cutoff),
        model=np.array(Path(args.model).name),
    )
    metadata = {
        "model": Path(args.model).name,
        "model_type_map": ["O", "H"],
        "n_atoms": len(positions),
        "box_length_angstrom": box_length,
        "cutoff_angstrom": cutoff,
        "total_energy_ev": total_energy,
        "max_force_ev_per_angstrom": float(np.linalg.norm(forces, axis=1).max()),
        "net_force_ev_per_angstrom": forces.sum(axis=0).tolist(),
        "output": args.output,
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
