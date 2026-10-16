"""Forward-pass dump of a compressed DPA1 (se_atten_v2, attn_layer=0) TF graph for one centre atom.

Fetches the graph's own tensors (environment matrix, type-pair embedding, tabulated
embedding inputs, descriptor, fitting activations, atomic energy) and recovers the
per-neighbour embedding rows G_j by re-running the model's own TabulateFusionSeAtten
op with one-hot environment rows.  Checks: sum_j R_j^T G_j / sel reproduces the
graph's contraction, D = T^T T_< reproduces o_descriptor, epsilon matches o_atom_energy.
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from deepmd.infer import DeepPot
    from deepmd.env import op_module, tf
    dp = DeepPot(str(a.model))
    g, sess = dp.graph, dp.sess
    tmap = dp.get_type_map()
    traj = np.load(a.trajectory)
    el = traj["elements"].astype(str)
    types = np.array([tmap.index(e) for e in el])
    perm = np.argsort(types, kind="stable")  # sorted slot k <-> original atom perm[k]
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    L = np.asarray(traj["box_length"], float).reshape(3)
    centre = int(traj["central_index"]); cs = int(inv[centre])
    n = len(el)
    natoms = np.zeros(2 + len(tmap), dtype=np.int32); natoms[0] = natoms[1] = n
    for t in types: natoms[2 + t] += 1
    T = lambda name: g.get_tensor_by_name("load/" + name + ":0")
    names = {"rmat": "o_rmat", "nlist": "o_nlist", "desc": "o_descriptor", "ae": "o_atom_energy",
             "emx": "Reshape_23", "em": "filter_type_all/Reshape_4", "two": "filter_type_all/mul_1",
             "xyz": "filter_type_all/TabulateFusionSeAtten", "f1": "layer_0/Reshape", "f2": "add_7", "f3": "add_8",
             "table": "Const", "tinfo": "filter_type_all/TabulateFusionSeAtten/table_info"}
    fetch = {k: T(v) for k, v in names.items()}
    sel = int(fetch["em"].shape[1]); last = 100
    with g.as_default():
        ph = {k: tf.compat.v1.placeholder(tf.float64, s) for k, s in (("emx", [None, 1]), ("em", [None, sel, 4]), ("two", [None, last]))}
        tab_op = op_module.tabulate_fusion_se_atten(T(names["table"]), T(names["tinfo"]), ph["emx"], ph["em"], ph["two"], last_layer_size=last, is_sorted=True)
    out, checks = {}, []
    for s in range(len(traj["positions"])):
        pos = traj["positions"][s][perm]
        feed = {T("t_coord"): pos.reshape(-1), T("t_type"): types[perm], T("t_natoms"): natoms,
                T("t_box"): np.diag(L).reshape(-1), T("t_mesh"): np.array([0, 0, 0, 2, 2, 2], dtype=np.int32)}
        r = sess.run(fetch, feed_dict=feed)
        nl = r["nlist"].reshape(n, sel)[cs]
        m = int(np.sum(nl >= 0)); assert np.all(nl[:m] >= 0) and np.all(nl[m:] < 0)
        rmat = r["rmat"].reshape(n, sel, 4)[cs]
        emx = r["emx"].reshape(n, sel, 1)[cs]; em = r["em"][cs]; two = r["two"].reshape(n, sel, last)[cs]
        onehot = np.zeros((sel, sel, 4)); onehot[np.arange(sel), np.arange(sel), 0] = 1.0
        tile = lambda x: np.tile(x[None], (sel, 1, 1)).reshape(sel * sel, -1)
        G_all = sess.run(tab_op, {ph["emx"]: tile(emx), ph["em"]: onehot, ph["two"]: tile(two)})[:, 0, :]
        Gs = sess.run(tab_op, {ph["emx"]: tile(emx), ph["em"]: onehot, ph["two"]: np.zeros((sel * sel, last))})[:m, 0, :]
        G = G_all[:m]
        xyz = r["xyz"][cs]
        xyz_re = np.einsum("jk,jc->kc", em, G_all)  # real rows + padded slots
        Tm = xyz / sel
        D = Tm.T @ Tm[:, :12]
        desc = r["desc"].reshape(n, -1)[cs]
        eps = float(r["ae"].reshape(-1)[cs])
        eps_traj = float(traj["atomic_energy_ev"][s, centre])
        ids = perm[nl[:m]]
        dist = np.linalg.norm(((traj["positions"][s][ids] - traj["positions"][s][centre] + L / 2) % L) - L / 2, axis=1)
        c = {"state": s, "n_neighbours": m, "xyz_from_rows_max_abs_err": float(np.abs(xyz_re - xyz).max()),
             "descriptor_max_abs_err": float(np.abs(D.reshape(-1) - desc).max()),
             "G_vs_Gs_times_1_plus_tau_max_abs_err": float(np.abs(G - Gs * (1 + two[:m])).max()),
             "em_vs_rmat_max_abs_err": float(np.abs(em[:m] - rmat[:m]).max()),
             "epsilon_model_ev": eps, "epsilon_traj_abs_err_ev": abs(eps - eps_traj),
             "rows_sorted_by_distance": bool(np.all(np.diff(dist) >= -1e-9))}
        checks.append(c); print(c, flush=True)
        rec = {"neighbour_count": m, "order": np.pad(ids, (0, sel - m), constant_values=-1), "rmat": np.pad(rmat[:m], ((0, sel - m), (0, 0))),
               "embed_s": np.pad(Gs, ((0, sel - m), (0, 0))), "tau": np.pad(two[:m], ((0, sel - m), (0, 0))),
               "embed_3": np.pad(G, ((0, sel - m), (0, 0))), "T": Tm, "D": D, "descriptor": desc,
               "fit_1": r["f1"][cs], "fit_2": r["f2"][cs], "fit_3": r["f3"][cs], "epsilon": eps, "n_pad": sel - m}
        for k, v in rec.items(): out.setdefault(k, []).append(v)
    np.savez_compressed(a.output, **{k: np.asarray(v) for k, v in out.items()})
    ok = all(c["xyz_from_rows_max_abs_err"] < 1e-8 and c["descriptor_max_abs_err"] < 1e-8 and c["epsilon_traj_abs_err_ev"] < 1e-6 for c in checks)
    a.output.with_suffix(".json").write_text(json.dumps({"model": a.model.name, "centre_index": centre, "sel": sel,
        "descriptor": "se_atten_v2, attn_layer=0, compressed (tabulated embedding)", "checks": checks, "reproduces_model": ok}, indent=2) + "\n")
    print("reproduces_model", ok)


if __name__ == "__main__":
    main()
