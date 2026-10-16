"""Build the 05 starting state from the 300 K DPA4C NVT production run (last frame)."""
import json, numpy as np
from ase.io import read
from ase import units
at = read("tnt_nvt_300K_prod.extxyz", index=-1)
L = np.array(at.cell.lengths()); assert np.allclose(at.cell.array, np.diag(L))
pos = at.get_positions() % L
vel = at.get_velocities() * units.fs  # A/fs
el = np.array(at.get_chemical_symbols())
# centre: the N atom closest to the box centre
d = np.linalg.norm(pos - L / 2, axis=1); d[el != "N"] = np.inf
c = int(np.argmin(d))
np.savez("tnt_box_336.npz", elements=el, box_length=L, positions_wrapped=pos, velocities=vel,
         central_index=np.array(c), cutoff=np.array(6.0))
T = (0.5 * np.sum(at.get_masses()[:, None] * at.get_velocities()**2) * 2 / (3 * len(at) - 3)) / units.kB
print(json.dumps({"n_atoms": len(at), "box": L.tolist(), "centre": c, "centre_el": el[c], "T_K": T}))
