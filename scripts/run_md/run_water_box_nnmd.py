"""Run a short, true periodic Velocity-Verlet trajectory with a DeePMD-kit model.

This is the single data producer for both neural-network MD stories in the
0314 project.  The same prepared 64-water box, the same Maxwell-Boltzmann
seed and the same integrator are used with either

* the frozen DeepPot-SE water model ``H2O-Phase-Diagram-model_compressed.pb``
  (DeepMD story), or
* the public ``DPA4C-Neo-OMat24`` PyTorch checkpoint (DPA4C story).

Every retained geometry gets a fresh model call; energies, atomic energies,
forces, virials, half-step velocities, displacements and the O126 cutoff
neighbourhood are recorded so the renderer never has to invent a number.

The workstation does not carry DeePMD-kit.  Copy this file together with the
model and ``water_box_64.npz`` to a Bohrium sandbox (see
``scripts/submit_calculation/bohr_sandbox_nnmd.sh``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np


EV_A_TO_A_FS2 = 0.00964853399
AMU_AFS2_TO_EV = 1.0 / EV_A_TO_A_FS2
KB_EV = 8.617333262145e-5
O_MASS = 15.9994
H_MASS = 1.008


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def maxwell_boltzmann_velocities(masses: np.ndarray, temperature_k: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(KB_EV * temperature_k / (masses * AMU_AFS2_TO_EV))
    velocities = rng.normal(0.0, sigma[:, None], size=(len(masses), 3))
    velocities -= np.average(velocities, axis=0, weights=masses)
    return velocities


def minimum_image_delta(positions: np.ndarray, centre: np.ndarray, box_length: float) -> np.ndarray:
    delta = np.asarray(positions, dtype=float) - np.asarray(centre, dtype=float)
    return delta - box_length * np.rint(delta / box_length)


def neighbour_ids(positions: np.ndarray, centre_index: int, box_length: float, cutoff: float) -> np.ndarray:
    delta = minimum_image_delta(positions, positions[centre_index], box_length)
    distances = np.linalg.norm(delta, axis=1)
    mask = (distances > 1.0e-12) & (distances <= cutoff)
    return np.flatnonzero(mask)[np.argsort(distances[mask])]


class ModelEvaluator:
    """Thin wrapper around ``deepmd.infer.DeepPot`` for .pb and .pt models."""

    def __init__(self, model_path: Path, elements: np.ndarray) -> None:
        from deepmd.infer import DeepPot

        self.model_path = model_path
        self.potential = DeepPot(str(model_path))
        self.type_map = [str(item) for item in self.potential.get_type_map()]
        missing = sorted(set(elements.tolist()) - set(self.type_map))
        if missing:
            raise RuntimeError(f"Model type map lacks {missing}")
        self.atom_types = np.asarray([self.type_map.index(item) for item in elements], dtype=int)
        self.fparam: np.ndarray | None = None
        self.atomic_status = "not evaluated"

    def _eval(self, coordinates: np.ndarray, cells: np.ndarray, *, atomic: bool) -> tuple:
        kwargs: dict[str, Any] = {"atomic": atomic}
        if self.fparam is not None:
            kwargs["fparam"] = self.fparam
        return self.potential.eval(coordinates, cells=cells, atom_types=self.atom_types, **kwargs)

    def evaluate(self, positions: np.ndarray, cell: np.ndarray) -> dict[str, Any]:
        n_atoms = len(positions)
        coordinates = positions.reshape(1, n_atoms, 3)
        cells = cell.reshape(1, 9)
        try:
            result = self._eval(coordinates, cells, atomic=True)
        except Exception as exc:  # DPA4C-Neo exposes charge/spin frame parameters
            if self.fparam is None and "fparam" in f"{exc}".lower():
                self.fparam = np.array([[0.0, 1.0]], dtype=float)
                result = self._eval(coordinates, cells, atomic=True)
            else:
                raise
        energy, forces, virial = result[0], result[1], result[2]
        if len(result) >= 4 and result[3] is not None:
            atomic_energy = np.asarray(result[3], dtype=float).reshape(n_atoms)
            self.atomic_status = "model output"
        else:
            atomic_energy = np.full(n_atoms, float(np.asarray(energy).reshape(-1)[0]) / n_atoms)
            self.atomic_status = "uniform fallback because atomic=True returned nothing"
        return {
            "energy": float(np.asarray(energy).reshape(-1)[0]),
            "forces": np.asarray(forces, dtype=float).reshape(n_atoms, 3),
            "virial": np.asarray(virial, dtype=float).reshape(3, 3),
            "atomic_energy": atomic_energy,
        }

    def probe(self, positions: np.ndarray, cell: np.ndarray, centre_index: int) -> dict[str, Any]:
        probe: dict[str, Any] = {
            "type_map": self.type_map,
            "atom_types_used": sorted(set(self.atom_types.tolist())),
            "fparam": None if self.fparam is None else self.fparam.tolist(),
            "api": {},
            "descriptor": {"status": "not attempted"},
            "model_definition": {"status": "not attempted"},
        }
        for name in ("get_rcut", "get_sel", "get_nsel", "get_ntypes", "get_dim_fparam", "get_dim_aparam", "get_model_size"):
            method = getattr(self.potential, name, None)
            if method is None:
                continue
            try:
                probe["api"][name] = _jsonable(method())
            except Exception as exc:
                probe["api"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        model_def = getattr(self.potential, "get_model_def_script", None)
        if model_def is not None:
            try:
                definition = model_def()
                if isinstance(definition, str):
                    try:
                        definition = json.loads(definition)
                    except json.JSONDecodeError:
                        pass
                probe["model_definition"] = {"status": "available", "data": _jsonable(definition)}
            except Exception as exc:
                probe["model_definition"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        eval_descriptor = getattr(self.potential, "eval_descriptor", None)
        if eval_descriptor is not None:
            try:
                kwargs = {} if self.fparam is None else {"fparam": self.fparam}
                descriptor = np.asarray(
                    eval_descriptor(positions.reshape(1, len(positions), 3), cell.reshape(1, 9), self.atom_types, **kwargs)
                )
                probe["descriptor"] = {
                    "status": "available",
                    "shape": list(descriptor.shape),
                    "centre": descriptor[0, centre_index].tolist(),
                    "centre_sha256": hashlib.sha256(np.ascontiguousarray(descriptor[0, centre_index]).tobytes()).hexdigest(),
                }
            except Exception as exc:
                probe["descriptor"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        else:
            probe["descriptor"] = {"status": "unavailable", "error": "DeepPot backend does not expose eval_descriptor"}
        return probe


def run(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    steps: int,
    dt_fs: float,
    temperature_k: float,
    seed: int,
    label: str,
) -> None:
    with np.load(input_path, allow_pickle=False) as source:
        initial_positions = np.asarray(source["positions_wrapped"], dtype=float)
        elements = np.asarray(source["elements"]).astype(str)
        box_length = float(np.asarray(source["box_length"]).reshape(-1)[0])
        central_index = int(np.asarray(source["central_index"]).reshape(-1)[0])
        visualization_cutoff = float(np.asarray(source["cutoff"]).reshape(-1)[0])
    if initial_positions.shape != (192, 3):
        raise ValueError(f"Expected a 192-atom box, got {initial_positions.shape}")
    if set(elements.tolist()) != {"O", "H"}:
        raise ValueError("Expected O/H water-box elements")

    masses = np.where(elements == "O", O_MASS, H_MASS).astype(float)
    cell = np.eye(3, dtype=float) * box_length
    evaluator = ModelEvaluator(model_path, elements)

    initial = initial_positions.copy()
    first = evaluator.evaluate(initial, cell)
    probe = evaluator.probe(initial, cell, central_index)
    api_cutoff = probe.get("api", {}).get("get_rcut")
    model_cutoff = float(api_cutoff) if isinstance(api_cutoff, (float, int)) else visualization_cutoff

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
    atomic_sum_ok = bool(np.isclose(np.sum(atomic_energy[0]), energies[0], atol=1.0e-4))

    for state in range(n_states):
        ids = neighbour_ids(positions[state], central_index, box_length, model_cutoff)
        neighbour_counts[state] = len(ids)
        neighbour_ids_by_state[state, : len(ids)] = ids
        if state == steps:
            break
        acceleration = forces[state] * EV_A_TO_A_FS2 / masses[:, None]
        half_velocities[state] = velocities[state] + 0.5 * acceleration * dt_fs
        next_positions = (positions[state] + half_velocities[state] * dt_fs) % box_length
        displacements[state] = next_positions - positions[state]
        displacements[state] -= box_length * np.rint(displacements[state] / box_length)
        next_eval = evaluator.evaluate(next_positions, cell)
        next_acceleration = next_eval["forces"] * EV_A_TO_A_FS2 / masses[:, None]
        positions[state + 1] = next_positions
        velocities[state + 1] = half_velocities[state] + 0.5 * next_acceleration * dt_fs
        forces[state + 1] = next_eval["forces"]
        atomic_energy[state + 1] = next_eval["atomic_energy"]
        energies[state + 1] = next_eval["energy"]
        virials[state + 1] = next_eval["virial"]
        atomic_sum_ok = atomic_sum_ok and bool(
            np.isclose(np.sum(atomic_energy[state + 1]), energies[state + 1], atol=1.0e-4)
        )
        print(f"[{label}] state {state + 1}/{steps}: E={energies[state + 1]:.6f} eV", flush=True)

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
    try:
        import deepmd

        deepmd_version = getattr(deepmd, "__version__", "unknown")
    except Exception:  # pragma: no cover - only for the metadata record
        deepmd_version = "unknown"
    metadata = {
        "schema": "nnmd_periodic_trajectory/v2",
        "label": label,
        "model": model_path.name,
        "model_sha256": sha256_file(model_path),
        "input": input_path.name,
        "input_sha256": sha256_file(input_path),
        "deepmd_version": deepmd_version,
        "python": platform.python_version(),
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
        "integrator": "full velocity Verlet with a fresh model force call at every new position",
        "ensemble_claim": "short NVE demonstration from a prepared box; not equilibrated",
        "atomic_energy_status": evaluator.atomic_status,
        "atomic_energy_sums_to_total": atomic_sum_ok,
        "neighbour_counts": neighbour_counts.tolist(),
        "net_force_ev_per_angstrom": forces.sum(axis=1).tolist(),
        "max_force_ev_per_angstrom": np.linalg.norm(forces, axis=2).max(axis=1).tolist(),
        "max_displacement_angstrom": np.linalg.norm(displacements, axis=2).max(axis=1).tolist(),
        "total_energy_ev": energies.tolist(),
        "model_probe": probe,
        "output": output_path.name,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "model_probe"}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--label", default="nnmd")
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
        steps=args.steps,
        dt_fs=args.dt,
        temperature_k=args.temperature,
        seed=args.seed,
        label=args.label,
    )


if __name__ == "__main__":
    main()
