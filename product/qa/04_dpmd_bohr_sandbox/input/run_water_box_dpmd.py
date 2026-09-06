"""Run a short, true periodic DeepMD Velocity-Verlet trajectory.

This is the data producer for the redesigned 04 Deep Potential MD story.  It
keeps the prepared 64-water box unchanged, evaluates the bundled frozen model
at every retained geometry, and records both the model outputs and the model
introspection that the installed DeepMD backend exposes.

The script is intentionally executable in the same environment as
``evaluate_water_box_deepmd.py``.  The local workstation may not have
DeepMD-kit installed; in that case this file can be copied to the existing
Bohrium worker image together with the model and ``water_box_64.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EV_A_TO_A_FS2 = 0.00964853399
AMU_AFS2_TO_EV = 1.0 / EV_A_TO_A_FS2
KB_EV = 8.617333262145e-5
O_MASS = 15.9994
H_MASS = 1.008
MODEL_TYPE_MAP = ["O", "H"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _find_keys(value: Any, names: set[str], prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in names:
                found[path] = _jsonable(item)
            found.update(_find_keys(item, names, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_keys(item, names, f"{prefix}[{index}]"))
    return found


def maxwell_boltzmann_velocities(
    masses: np.ndarray, temperature_k: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(KB_EV * temperature_k / (masses * AMU_AFS2_TO_EV))
    velocities = rng.normal(0.0, sigma[:, None], size=(len(masses), 3))
    velocities -= np.average(velocities, axis=0, weights=masses)
    return velocities


def kinetic_energy_ev(velocities: np.ndarray, masses: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(masses[:, None] * velocities**2)
        * AMU_AFS2_TO_EV
    )


def minimum_image_delta(
    positions: np.ndarray, centre: np.ndarray, box_length: float
) -> np.ndarray:
    delta = np.asarray(positions, dtype=float) - np.asarray(centre, dtype=float)
    return delta - box_length * np.rint(delta / box_length)


def neighbour_ids(
    positions: np.ndarray,
    centre_index: int,
    box_length: float,
    cutoff: float,
) -> np.ndarray:
    delta = minimum_image_delta(
        positions, positions[centre_index], box_length
    )
    distances = np.linalg.norm(delta, axis=1)
    mask = (distances > 1.0e-12) & (distances <= cutoff)
    return np.flatnonzero(mask)[np.argsort(distances[mask])]


def eval_model(
    potential: Any,
    positions: np.ndarray,
    cell: np.ndarray,
    atom_types: np.ndarray,
) -> dict[str, Any]:
    coordinates = positions.reshape(1, len(positions), 3)
    cells = cell.reshape(1, 9)
    try:
        energy, forces, virial, atomic_energy, _ = potential.eval(
            coordinates,
            cells=cells,
            atom_types=atom_types,
            atomic=True,
        )
        atomic_status = "model output"
        atomic_energy = np.asarray(atomic_energy).reshape(len(positions))
    except TypeError:
        energy, forces, virial = potential.eval(
            coordinates,
            cells=cells,
            atom_types=atom_types,
        )
        atomic_energy = np.full(
            len(positions),
            float(np.asarray(energy).reshape(-1)[0]) / len(positions),
        )
        atomic_status = "uniform fallback because atomic=True is unavailable"
    return {
        "energy": float(np.asarray(energy).reshape(-1)[0]),
        "forces": np.asarray(forces, dtype=float).reshape(len(positions), 3),
        "virial": np.asarray(virial, dtype=float).reshape(3, 3),
        "atomic_energy": np.asarray(atomic_energy, dtype=float),
        "atomic_status": atomic_status,
    }


def probe_model(
    potential: Any,
    positions: np.ndarray,
    cell: np.ndarray,
    atom_types: np.ndarray,
) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "type_map": _jsonable(potential.get_type_map()),
        "api": {},
        "descriptor": {"status": "not attempted"},
        "model_definition": {"status": "not attempted"},
    }
    for name in (
        "get_rcut",
        "get_sel",
        "get_nsel",
        "get_sel_type",
        "get_ntypes",
        "get_model_size",
    ):
        method = getattr(potential, name, None)
        if method is None:
            continue
        try:
            probe["api"][name] = _jsonable(method())
        except Exception as exc:  # model-version capability probe
            probe["api"][name] = {"error": f"{type(exc).__name__}: {exc}"}

    model_def = getattr(potential, "get_model_def_script", None)
    if model_def is not None:
        try:
            definition = _jsonable(model_def())
            probe["model_definition"] = {
                "status": "available",
                "data": definition,
                "relevant_keys": _find_keys(
                    definition,
                    {
                        "type",
                        "descriptor",
                        "rcut",
                        "rcut_smth",
                        "sel",
                        "neuron",
                        "fitting_net",
                        "fitting",
                        "embedding_width",
                    },
                ),
            }
        except Exception as exc:
            probe["model_definition"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }

    eval_descriptor = getattr(potential, "eval_descriptor", None)
    if eval_descriptor is not None:
        try:
            descriptor = np.asarray(
                eval_descriptor(
                    positions.reshape(1, len(positions), 3),
                    cell.reshape(1, 9),
                    atom_types,
                )
            )
            probe["descriptor"] = {
                "status": "available",
                "shape": list(descriptor.shape),
                "o126": descriptor[0, 126].tolist(),
                "o126_sha256": hashlib.sha256(
                    np.asarray(descriptor[0, 126]).tobytes()
                ).hexdigest(),
            }
        except Exception as exc:
            probe["descriptor"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        probe["descriptor"] = {
            "status": "unavailable",
            "error": "DeepPot backend does not expose eval_descriptor",
        }
    return probe


def run(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    steps: int,
    dt_fs: float,
    temperature_k: float,
    seed: int,
) -> None:
    from deepmd.infer import DeepPot

    with np.load(input_path, allow_pickle=False) as source:
        initial_positions = np.asarray(source["positions_wrapped"], dtype=float)
        elements = np.asarray(source["elements"]).astype(str)
        box_length = float(np.asarray(source["box_length"]).reshape(-1)[0])
        central_index = int(np.asarray(source["central_index"]).reshape(-1)[0])
        visualization_cutoff = float(
            np.asarray(source["cutoff"]).reshape(-1)[0]
        )
    if initial_positions.shape != (192, 3):
        raise ValueError(f"Expected a 192-atom box, got {initial_positions.shape}")
    if set(elements.tolist()) != {"O", "H"}:
        raise ValueError("Expected O/H water-box elements")

    atom_types = np.asarray([0 if item == "O" else 1 for item in elements])
    masses = np.where(elements == "O", O_MASS, H_MASS).astype(float)
    cell = np.eye(3, dtype=float) * box_length
    potential = DeepPot(str(model_path))
    type_map = list(potential.get_type_map())
    if type_map != MODEL_TYPE_MAP:
        raise RuntimeError(f"Unexpected model type map: {type_map}")

    initial = initial_positions.copy()
    probe = probe_model(potential, initial, cell, atom_types)
    api_cutoff = probe.get("api", {}).get("get_rcut")
    model_cutoff = (
        float(api_cutoff)
        if isinstance(api_cutoff, (float, int))
        else visualization_cutoff
    )
    first = eval_model(potential, initial, cell, atom_types)

    n_states = steps + 1
    positions = np.zeros((n_states, len(elements), 3), dtype=float)
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)
    atomic_energy = np.zeros((n_states, len(elements)), dtype=float)
    energies = np.zeros(n_states, dtype=float)
    virials = np.zeros((n_states, 3, 3), dtype=float)
    half_velocities = np.zeros((steps, len(elements), 3), dtype=float)
    displacements = np.zeros_like(half_velocities)
    neighbour_ids_by_state = np.full((n_states, len(elements)), -1, dtype=int)
    neighbour_counts = np.zeros(n_states, dtype=int)

    positions[0] = initial
    velocities[0] = maxwell_boltzmann_velocities(masses, temperature_k, seed)
    forces[0] = first["forces"]
    atomic_energy[0] = first["atomic_energy"]
    energies[0] = first["energy"]
    virials[0] = first["virial"]
    if not np.isclose(np.sum(atomic_energy[0]), energies[0], atol=1.0e-7):
        raise RuntimeError("Initial atomic energies do not sum to total energy")

    for state in range(n_states):
        ids = neighbour_ids(
            positions[state], central_index, box_length, model_cutoff
        )
        neighbour_counts[state] = len(ids)
        neighbour_ids_by_state[state, : len(ids)] = ids
        if state == steps:
            break
        acceleration = forces[state] * EV_A_TO_A_FS2 / masses[:, None]
        half_velocities[state] = velocities[state] + 0.5 * acceleration * dt_fs
        next_positions = (
            positions[state] + half_velocities[state] * dt_fs
        ) % box_length
        displacements[state] = next_positions - positions[state]
        displacements[state] -= box_length * np.rint(
            displacements[state] / box_length
        )
        next_eval = eval_model(potential, next_positions, cell, atom_types)
        next_acceleration = next_eval["forces"] * EV_A_TO_A_FS2 / masses[:, None]
        positions[state + 1] = next_positions
        velocities[state + 1] = (
            half_velocities[state] + 0.5 * next_acceleration * dt_fs
        )
        forces[state + 1] = next_eval["forces"]
        atomic_energy[state + 1] = next_eval["atomic_energy"]
        energies[state + 1] = next_eval["energy"]
        virials[state + 1] = next_eval["virial"]
        if not np.isclose(
            np.sum(atomic_energy[state + 1]), energies[state + 1], atol=1.0e-7
        ):
            raise RuntimeError(f"Atomic energy sum failed at state {state + 1}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        elements=elements,
        box_length=np.array(box_length),
        cell=cell,
        central_index=np.array(central_index),
        cutoff_angstrom=np.array(model_cutoff),
        positions=positions,
        velocities=velocities,
        half_velocities=half_velocities,
        displacements=displacements,
        forces_ev_per_angstrom=forces,
        atomic_energy_ev=atomic_energy,
        total_energy_ev=energies,
        virial_ev=virials,
        neighbour_ids=neighbour_ids_by_state,
        neighbour_counts=neighbour_counts,
        dt_fs=np.array(dt_fs),
        temperature_k=np.array(temperature_k),
        velocity_seed=np.array(seed),
    )
    metadata = {
        "schema": "dpmd_periodic_trajectory/v1",
        "model": model_path.name,
        "model_sha256": sha256_file(model_path),
        "input": input_path.name,
        "input_sha256": sha256_file(input_path),
        "n_atoms": int(len(elements)),
        "n_states": n_states,
        "n_updates": steps,
        "box_length_angstrom": box_length,
        "central_index": central_index,
        "visualization_cutoff_angstrom": visualization_cutoff,
        "model_cutoff_angstrom": model_cutoff,
        "dt_fs": dt_fs,
        "temperature_k": temperature_k,
        "velocity_seed": seed,
        "integrator": "full velocity Verlet with a fresh DeepMD force call at every new position",
        "ensemble_claim": "short NVE demonstration from a prepared box; not equilibrated",
        "atomic_energy_status": first["atomic_status"],
        "neighbour_counts": neighbour_counts.tolist(),
        "net_force_ev_per_angstrom": forces.sum(axis=1).tolist(),
        "max_force_ev_per_angstrom": np.linalg.norm(forces, axis=2).max(axis=1).tolist(),
        "max_displacement_angstrom": np.linalg.norm(displacements, axis=2).max(axis=1).tolist(),
        "model_probe": probe,
        "output": output_path.name,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=260906)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    run(
        args.model,
        args.input,
        args.output,
        args.metadata,
        args.steps,
        args.dt,
        args.temperature,
        args.seed,
    )


if __name__ == "__main__":
    main()
