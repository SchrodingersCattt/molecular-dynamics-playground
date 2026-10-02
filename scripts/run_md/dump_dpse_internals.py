"""Dump every intermediate of a DeepPot-SE (se_e2_a) forward pass for one atom.

The frozen, *uncompressed* model is read with DeePMD-kit 2.x (TensorFlow).  The
embedding and fitting weights are taken from the graph constants and the
forward pass of the centre atom is recomputed layer by layer in NumPy.  The
recomputation is accepted only if it reproduces the model's own descriptor
(``eval_descriptor``) and atomic energy; the tolerances are stored with the
data.

Output per trajectory state (centre atom only):

* ``rmat_raw``      (N, 4)  s(r), s x/r, s y/r, s z/r for the N real neighbours,
                            ordered like DeePMD (by type, then distance)
* ``rmat``          (N, 4)  the same rows after the model's davg/dstd normalisation
* ``embed_1/2/3``   (N, 25/50/100) embedding-net activations, row-aligned
* ``T``             (4, 100) R^T G / n_sel
* ``D``             (100, 12) descriptor matrix (flattened = model descriptor)
* ``fit_1/2/3``     (240,) fitting-net hidden activations
* ``epsilon``       scalar atomic energy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def switch(r: np.ndarray, rcs: float, rc: float) -> np.ndarray:
    u = np.clip((r - rcs) / (rc - rcs), 0.0, 1.0)
    smooth = u**3 * (-6.0 * u**2 + 15.0 * u - 10.0) + 1.0
    return np.where(r < rcs, 1.0, np.where(r < rc, smooth, 0.0)) / r


def graph_constants(potential) -> dict[str, np.ndarray]:
    import tensorflow.compat.v1 as tf

    graph = potential.graph
    wanted = [op for op in graph.get_operations() if op.type == "Const"]
    names = {}
    prefixes = ("filter_type_", "layer_", "final_layer", "descrpt_attr/t_avg", "descrpt_attr/t_std", "fitting_attr/t_bias_atom_e", "train_attr/training_script")
    for op in wanted:
        short = op.name.split("/", 1)[1] if op.name.startswith("load/") else op.name
        if short.startswith(prefixes):
            names[short] = op.outputs[0]
    with tf.Session(graph=graph) as sess:
        values = sess.run(list(names.values()))
    return dict(zip(names.keys(), [np.asarray(v) for v in values]))


def resnet_layer(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = np.tanh(x @ w + b)
    if w.shape[1] == w.shape[0]:
        return x + y
    if w.shape[1] == 2 * w.shape[0]:
        return np.concatenate([x, x], axis=-1) + y
    return y


def centre_internals(consts, cfg, positions, types, box, centre):
    desc = cfg["descriptor"]
    rc, rcs = float(desc["rcut"]), float(desc["rcut_smth"])
    sel = list(desc["sel"])
    n_sel = int(sum(sel))
    axis = int(desc["axis_neuron"])
    ti = int(types[centre])

    delta = positions - positions[centre]
    delta -= box * np.rint(delta / box)
    dist = np.linalg.norm(delta, axis=1)
    rows_raw, order, neighbour_types = [], [], []
    for tj in range(len(sel)):
        ids = np.flatnonzero((types == tj) & (dist > 1e-10) & (dist < rc))
        ids = ids[np.argsort(dist[ids], kind="stable")][: sel[tj]]
        order.extend(ids.tolist())
        neighbour_types.extend([tj] * len(ids))
    order = np.asarray(order, dtype=int)
    neighbour_types = np.asarray(neighbour_types, dtype=int)
    r = dist[order]
    s = switch(r, rcs, rc)
    rows_raw = np.column_stack([s, s * delta[order, 0] / r, s * delta[order, 1] / r, s * delta[order, 2] / r])

    davg = consts["descrpt_attr/t_avg"].reshape(len(sel), n_sel, 4)[ti]
    dstd = consts["descrpt_attr/t_std"].reshape(len(sel), n_sel, 4)[ti]
    slot = np.concatenate([np.arange(sel[tj]) + sum(sel[:tj]) for tj in range(len(sel))])
    slot_of_row = np.concatenate([np.arange(np.sum(neighbour_types == tj)) + sum(sel[:tj]) for tj in range(len(sel))]).astype(int)
    rmat = (rows_raw - davg[slot_of_row]) / dstd[slot_of_row]
    # padded slots (r >= rc) still contribute (0 - davg)/dstd in DeePMD
    pad_slots = np.setdiff1d(slot, slot_of_row)
    pad_rows = (0.0 - davg[pad_slots]) / dstd[pad_slots]
    pad_types = np.searchsorted(np.cumsum(sel), pad_slots, side="right")

    neurons = list(desc["neuron"])

    def embed(x_s: np.ndarray, tj: int) -> list[np.ndarray]:
        acts, x = [], x_s.reshape(-1, 1)
        for layer in range(1, len(neurons) + 1):
            w = consts[f"filter_type_{ti}/matrix_{layer}_{tj}"]
            b = consts[f"filter_type_{ti}/bias_{layer}_{tj}"]
            x = resnet_layer(x, w, b) if layer > 1 else np.tanh(x @ w + b)
            acts.append(x)
        return acts

    layers = [np.zeros((len(order), n)) for n in neurons]
    for tj in range(len(sel)):
        mask = neighbour_types == tj
        if mask.any():
            for k, a in enumerate(embed(rmat[mask, 0], tj)):
                layers[k][mask] = a
    g_pad = np.zeros((len(pad_slots), neurons[-1]))
    for tj in range(len(sel)):
        mask = pad_types == tj
        if mask.any():
            g_pad[mask] = embed(pad_rows[mask, 0], tj)[-1]

    T = (rmat.T @ layers[-1] + pad_rows.T @ g_pad) / n_sel
    D = T.T @ T[:, :axis]

    fit = cfg["fitting_net"]
    x = D.reshape(-1)
    fit_acts = []
    for k in range(len(fit["neuron"])):
        w = consts[f"layer_{k}_type_{ti}/matrix"]
        b = consts[f"layer_{k}_type_{ti}/bias"]
        y = np.tanh(x @ w + b)
        idt = consts.get(f"layer_{k}_type_{ti}/idt")
        if idt is not None:
            y = y * idt
        if w.shape[0] == w.shape[1]:
            y = x + y
        x = y
        fit_acts.append(x.copy())
    w = consts[f"final_layer_type_{ti}/matrix"]
    b = consts[f"final_layer_type_{ti}/bias"]
    epsilon = float((x @ w + b).reshape(-1)[0]) + float(consts["fitting_attr/t_bias_atom_e"].reshape(-1)[ti])

    return {
        "order": order,
        "neighbour_types": neighbour_types,
        "distance": r,
        "rmat_raw": rows_raw,
        "rmat": rmat,
        "n_pad": len(pad_slots),
        "embed": layers,
        "T": T,
        "D": D,
        "fit": fit_acts,
        "epsilon": epsilon,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from deepmd.infer import DeepPot

    potential = DeepPot(str(args.model))
    consts = graph_constants(potential)
    script = consts.pop("train_attr/training_script")
    script = script.item() if hasattr(script, "item") else script
    cfg = json.loads(script.decode() if isinstance(script, bytes) else script)["model"]
    type_map = potential.get_type_map()

    traj = np.load(args.trajectory)
    elements = traj["elements"].astype(str)
    types = np.asarray([type_map.index(e) for e in elements])
    box = float(traj["box_length"])
    centre = int(traj["central_index"])
    cell = np.eye(3) * box

    out: dict[str, list] = {}
    checks = []
    for state, positions in enumerate(traj["positions"]):
        rec = centre_internals(consts, cfg, positions, types, box, centre)
        model_desc = np.asarray(potential.eval_descriptor(positions.reshape(1, -1, 3), cell.reshape(1, 9), types))[0, centre]
        _, _, _, atom_e, _ = potential.eval(positions.reshape(1, -1, 3), cell.reshape(1, 9), types, atomic=True)
        model_eps = float(np.asarray(atom_e).reshape(-1)[centre])
        desc_err = float(np.abs(model_desc - rec["D"].reshape(-1)).max())
        eps_err = abs(model_eps - rec["epsilon"])
        checks.append({"state": state, "descriptor_max_abs_err": desc_err, "epsilon_abs_err_ev": eps_err, "epsilon_model_ev": model_eps})
        print(checks[-1], flush=True)
        for key in ("order", "neighbour_types", "distance", "rmat_raw", "rmat", "T", "D"):
            out.setdefault(key, []).append(rec[key])
        for k, a in enumerate(rec["embed"]):
            out.setdefault(f"embed_{k + 1}", []).append(a)
        for k, a in enumerate(rec["fit"]):
            out.setdefault(f"fit_{k + 1}", []).append(a)
        out.setdefault("epsilon", []).append(rec["epsilon"])
        out.setdefault("n_pad", []).append(rec["n_pad"])

    ok = all(c["descriptor_max_abs_err"] < 1e-8 and c["epsilon_abs_err_ev"] < 1e-8 for c in checks)
    n_max = max(len(o) for o in out["order"])
    packed = {}
    for key, values in out.items():
        if key in ("order", "neighbour_types", "distance", "rmat_raw", "rmat") or key.startswith("embed_"):
            first = np.asarray(values[0])
            shape = (len(values), n_max) + first.shape[1:]
            arr = np.full(shape, -1 if key in ("order", "neighbour_types") else np.nan)
            for i, v in enumerate(values):
                arr[i, : len(v)] = v
            packed[key] = arr
        else:
            packed[key] = np.asarray(values)
    packed["neighbour_count"] = np.asarray([len(o) for o in out["order"]])
    np.savez_compressed(args.output, **packed)
    meta = {"model": args.model.name, "centre_index": centre, "config": cfg, "checks": checks, "reproduces_model": ok}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("reproduces_model", ok)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
