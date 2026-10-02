"""Record every intermediate of a DPA4C forward pass for the centre atom.

The checkpoint is evaluated with DeePMD-kit's ``pt_expt`` backend.  The
descriptor stages (edge features, moment aggregation, invariant readout) are
captured by wrapping the ``DescrptDPA4C`` methods; every torch sub-module of
the model also gets a forward hook, which records the fitting-network layers.
Nothing is recomputed: the arrays are the tensors the model produced while
computing the energy and forces of the stored trajectory states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RECORD: dict[str, object] = {}


def _np(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().double().numpy()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, (tuple, list)):
        return [_np(v) for v in value]
    return value


def wrap(cls, name: str, key: str | None = None) -> None:
    original = getattr(cls, name)
    key = key or name

    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if key == "evaluate_graph":
            RECORD["_descriptor"] = self
        if RECORD.get("_on"):
            RECORD.setdefault(key, []).append({"args": _np(list(args)), "result": _np(result)})
        return result

    setattr(cls, name, wrapper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    from deepmd.dpmodel.descriptor.dpa4c import DescrptDPA4C
    from deepmd.infer import DeepPot

    for name in ("evaluate_graph", "build_edge_features", "aggregate_moments", "build_invariant_descriptor"):
        wrap(DescrptDPA4C, name)

    potential = DeepPot(str(args.model))
    type_map = potential.get_type_map()
    traj = np.load(args.trajectory)
    elements = traj["elements"].astype(str)
    types = np.asarray([type_map.index(e) for e in elements])
    box = float(traj["box_length"])
    centre = int(traj["central_index"])
    cell = (np.eye(3) * box).reshape(1, 9)
    fparam = np.array([[0.0, 1.0]]) if potential.get_dim_fparam() == 2 else None

    model = None
    potential.eval(traj["positions"][0].reshape(1, -1, 3), cell, types, atomic=True, fparam=fparam)
    descriptor = RECORD["_descriptor"]
    wrap(type(descriptor.radial_basis), "call", "radial_basis")
    wrap(type(descriptor.radial_embedding), "call_hidden", "radial_hidden")
    wrap(type(descriptor.radial_embedding), "call_output", "radial")
    wrap(DescrptDPA4C, "build_pair_conditioning")
    wrap(type(descriptor.readout), "call", "readout")
    RECORD["_atomic_model"] = potential.deep_eval._dpmodel.atomic_model
    RECORD["_centre_type"] = int(types[centre])
    readout_layout = {
        "degree_channels": [int(c) for c in descriptor.readout.degree_channels],
        "gram_offsets": [int(c) for c in descriptor.readout.gram_offsets],
        "dim_out": int(descriptor.readout.get_dim_out()),
    }

    states = range(1) if args.probe else range(len(traj["positions"]))
    out: dict[str, list] = {}
    checks = []
    for state in states:
        for key in [k for k in RECORD if not k.startswith("_")]:
            del RECORD[key]
        RECORD["_on"] = True
        result = potential.eval(traj["positions"][state].reshape(1, -1, 3), cell, types, atomic=True, fparam=fparam)
        RECORD["_on"] = False
        atom_e = np.asarray(result[3]).reshape(-1)
        if args.probe:
            def describe(v, depth=0):
                if isinstance(v, np.ndarray):
                    return f"ndarray{v.shape}"
                if isinstance(v, list):
                    return "[" + ", ".join(describe(x, depth + 1) for x in v) + "]"
                if isinstance(v, dict):
                    return "{" + ", ".join(f"{k}: {describe(x, depth + 1)}" for k, x in v.items()) + "}"
                return type(v).__name__
            print("readout layout", readout_layout)
            for key, value in RECORD.items():
                if not key.startswith("_"):
                    print(key, len(value), describe(value[0]) if key != "layer" else [describe(v["result"]) for v in value])
            graph = RECORD["evaluate_graph"][0]["args"][0]
            print("graph attrs", [a for a in dir(graph) if not a.startswith("_")])
            print("eps centre", atom_e[centre], "E", np.asarray(result[0]).reshape(-1))
            return
        rec = extract_centre(RECORD, centre, traj["positions"][state], box)
        rec["epsilon_model"] = float(atom_e[centre])
        check = rec.pop("_checks")
        check["epsilon_abs_err_ev"] = abs(check["epsilon_recomputed"] - rec["epsilon_model"])
        for key, value in rec.items():
            out.setdefault(key, []).append(value)
        checks.append({"state": state, **check})
        print(checks[-1], flush=True)

    packed = {}
    for key, values in out.items():
        if key == "_checks":
            continue
        first = np.asarray(values[0])
        if first.ndim >= 1 and len({np.asarray(v).shape for v in values}) > 1:
            n_max = max(np.asarray(v).shape[0] for v in values)
            arr = np.full((len(values), n_max) + first.shape[1:], np.nan)
            for i, v in enumerate(values):
                arr[i, : len(v)] = v
            packed[key] = arr
        else:
            packed[key] = np.asarray(values)
    np.savez_compressed(args.output, **packed)
    # The checkpoint runs in float32, so the recomputation tolerances are float32-sized.
    ok = all(c["moments_max_abs_err"] < 1e-5 and c["amplitude_max_abs_err"] < 1e-5 and c["epsilon_abs_err_ev"] < 1e-4 for c in checks)
    meta = {"model": args.model.name, "centre_index": centre, "readout_layout": readout_layout, "checks": checks, "reproduces_model": ok}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("reproduces_model", ok)
    if not ok:
        raise SystemExit(1)


def extract_centre(record, centre, positions, box):
    graph = record["evaluate_graph"][0]["args"][0]
    edge_index = np.asarray(_np(graph.edge_index))
    edge_vec = np.asarray(_np(graph.edge_vec))
    edge_mask = np.asarray(_np(graph.edge_mask)).astype(bool)
    n_local = int(np.asarray(_np(graph.n_local)).reshape(-1)[0]) if graph.n_local is not None else len(positions)
    amplitude, basis, envelope, _ = record["build_edge_features"][0]["result"]
    sel = np.flatnonzero((edge_index[1] == centre) & edge_mask & (envelope > 0))
    dist = np.linalg.norm(edge_vec[sel], axis=1)
    sel = sel[np.argsort(dist, kind="stable")]
    dist = np.linalg.norm(edge_vec[sel], axis=1)

    radial_basis = record["radial_basis"][0]["result"][sel]
    radial_hidden = record["radial_hidden"][0]["result"][sel]
    radial = record["radial"][0]["result"][sel]
    scale, shift = record["build_pair_conditioning"][0]["result"][:2]
    scale, shift = scale[sel], shift[sel]
    moments, divisors = record["aggregate_moments"][0]["result"]
    centre_type = record["build_invariant_descriptor"][0]["args"][1][centre]
    readout = record["readout"][0]["result"][centre]
    descriptor = record["build_invariant_descriptor"][0]["result"][centre]

    amp_check = np.abs((radial * scale + shift) * envelope[sel][:, None] - amplitude[sel]).max()
    descriptor_obj = record["_descriptor"]
    channel_index = np.asarray(_np(descriptor_obj.angular_channel_index)).astype(int)
    harmonic_index = np.asarray(_np(descriptor_obj.angular_harmonic_index)).astype(int)
    env = envelope[sel]
    reduced = np.concatenate([
        [np.sum(env**2), np.sum(env**4)],
        amplitude[sel].sum(axis=0),
        (amplitude[sel][:, channel_index] * basis[sel][:, harmonic_index] * env[:, None]).sum(axis=0),
    ])
    div = np.sqrt(reduced[:2] + descriptor_obj._DEGREE_NORM_FLOOR)
    n0 = len(amplitude[sel][0])
    moments_np = np.concatenate([reduced[2 : 2 + n0] / div[0], reduced[2 + n0 :] / div[1]])
    mom_check = np.abs(moments_np - moments[centre]).max()

    import torch

    atomic_model = record["_atomic_model"]
    ctype = record["_centre_type"]
    network = atomic_model.fitting_net.nets._networks[0]
    param = next(network.parameters())
    x = torch.as_tensor(descriptor, dtype=param.dtype, device=param.device)[None, :]
    fit_acts = []
    with torch.no_grad():
        for layer in network.layers:
            x = layer(x)
            fit_acts.append(_np(x)[0])
    out_bias = float(_np(atomic_model.out_bias)[0, ctype, 0])
    epsilon_recomputed = float(fit_acts[-1][0]) + out_bias

    return {
        "neighbour_ids": edge_index[0][sel] % n_local,
        "distance": dist,
        "edge_vec": edge_vec[sel],
        "radial_basis": radial_basis,
        "radial_hidden": radial_hidden,
        "radial": radial,
        "pair_scale": scale,
        "pair_shift": shift,
        "envelope": env,
        "amplitude": amplitude[sel],
        "harmonics": basis[sel],
        "moments": moments[centre],
        "divisors": divisors[centre],
        "readout": readout,
        "centre_type_embedding": centre_type,
        "descriptor": descriptor,
        "fit_layers": np.concatenate([np.asarray(a).reshape(-1) for a in fit_acts]),
        "fit_layer_sizes": np.asarray([np.asarray(a).size for a in fit_acts]),
        "epsilon_recomputed": epsilon_recomputed,
        "out_bias_ev": out_bias,
        "_checks": {"amplitude_max_abs_err": float(amp_check), "moments_max_abs_err": float(mom_check), "n_neighbours": int(len(sel)), "fit_layer_sizes": [int(np.asarray(a).size) for a in fit_acts], "epsilon_recomputed": epsilon_recomputed},
    }


if __name__ == "__main__":
    main()
