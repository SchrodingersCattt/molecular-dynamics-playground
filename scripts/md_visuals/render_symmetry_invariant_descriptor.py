"""Rigid H2O symmetry proof with full Cartesian and DeepMD environment matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static
from mattervis_story import camera_for_source, render_structure
from responsive_story import EMERALD, story_axes, panel_box, place_main

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "06_rigid_water_descriptor_invariance"
QA_DIR = ROOT / "qa" / "06_symmetry_invariance"
SOURCE = QA_DIR / "water_rigid.extxyz"
META = QA_DIR / "descriptor_provenance.json"
BOX = 12.0
RCUT = 6.0
RCUT_SMTH = 5.5
N_FRAMES = 240
FPS = 24


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    q = 1.0 - c
    return np.array([[c+x*x*q, x*y*q-z*s, x*z*q+y*s],
                     [y*x*q+z*s, c+y*y*q, y*z*q-x*s],
                     [z*x*q-y*s, z*y*q+x*s, c+z*z*q]])


def _mic(delta: np.ndarray) -> np.ndarray:
    return delta - BOX * np.round(delta / BOX)


def _find_water() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(ROOT / "data" / "water_box_64.npz", allow_pickle=False) as data:
        positions = np.asarray(data["positions_wrapped"], dtype=float)
        elements = np.asarray(data["elements"]).astype(str)
        molecule_ids = np.asarray(data["molecule_ids"], dtype=int)
        source_ids = np.asarray(data.get("source_ids", np.arange(len(elements))), dtype=int)
    for molecule in np.unique(molecule_ids):
        indices = np.flatnonzero(molecule_ids == molecule)
        oxy = indices[elements[indices] == "O"]
        hyd = indices[elements[indices] == "H"]
        if len(oxy) == 1 and len(hyd) >= 2:
            o = int(oxy[0])
            hyd = hyd[np.argsort(source_ids[hyd])[:2]]
            body = np.vstack((np.zeros(3), _mic(positions[hyd] - positions[o])))
            return body, np.asarray(["O", "H", "H"]), source_ids[np.asarray([o, *hyd])]
    raise RuntimeError("No one-O/two-H molecule found")


def _make_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body, symbols, source_ids = _find_water()
    frames = []
    for index in range(N_FRAMES):
        phase = index / max(N_FRAMES - 1, 1)
        rotation = _rotation(np.array([0.35, 0.55, 0.76]), 2.0 * np.pi * phase)
        translation = np.array([3.2 + 4.2 * phase, 5.6 + 0.55*np.sin(2*np.pi*phase), 5.8 + 0.45*np.cos(2*np.pi*phase)])
        frames.append(translation + body @ rotation.T)
    return np.asarray(frames), symbols, source_ids


def _switch(r: float) -> float:
    if r >= RCUT:
        return 0.0
    if r < RCUT_SMTH:
        return 1.0 / r
    u = (r - RCUT_SMTH) / (RCUT - RCUT_SMTH)
    return (u**3 * (-6*u**2 + 15*u - 10) + 1.0) / r


def _environment_matrix(position: np.ndarray) -> np.ndarray:
    rel = _mic(position[1:] - position[0])
    order = np.argsort(np.linalg.norm(rel, axis=1), kind="stable")
    rel = rel[order]
    ex = rel[0] / np.linalg.norm(rel[0])
    ez = np.cross(rel[0], rel[1]); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    if float(np.dot(rel[1], ey)) < 0:
        ey *= -1; ez *= -1
    basis = np.vstack((ex, ey, ez))
    local = rel @ basis.T
    rows = []
    for vector in local:
        radius = float(np.linalg.norm(vector)); s = _switch(radius)
        rows.append([s, s*vector[0]/radius, s*vector[1]/radius, s*vector[2]/radius])
    return np.asarray(rows)


def _write_source(frames: np.ndarray, symbols: np.ndarray) -> None:
    from ase import Atoms
    from ase.io import write
    QA_DIR.mkdir(parents=True, exist_ok=True)
    write(SOURCE, [Atoms(symbols=symbols.tolist(), positions=frame, cell=np.eye(3)*BOX, pbc=True) for frame in frames], format="extxyz")


def _render_assets(frames: np.ndarray) -> dict[str, object]:
    from common import sha256_file
    mattervis = QA_DIR / "mattervis"; mattervis.mkdir(parents=True, exist_ok=True)
    camera = camera_for_source(SOURCE, target=(BOX/2, BOX/2, BOX/2), ortho_scale=14.0, frame=0, direction=(0.25, -1.0, 0.65))
    for frame in range(N_FRAMES):
        render_structure(SOURCE, mattervis/f"frame_{frame:04d}.png", camera=camera, frame=frame, view="unit_cell", width=1700, height=1180, atom_scale=1.55, bond_radius=0.14, show_cell=True, cell_color="#8E8E8E", cell_width_px=1.4, include_boundary_replicas=False)
    matrices = np.asarray([_environment_matrix(frame) for frame in frames])
    bonds = np.asarray([[np.linalg.norm(_mic(f[1]-f[0])), np.linalg.norm(_mic(f[2]-f[0])), np.linalg.norm(_mic(f[2]-f[1]))] for f in frames])
    angles = np.asarray([np.dot(_mic(f[1]-f[0]), _mic(f[2]-f[0]))/(np.linalg.norm(_mic(f[1]-f[0]))*np.linalg.norm(_mic(f[2]-f[0]))) for f in frames])
    payload = {
        "source": str(SOURCE), "source_sha256": sha256_file(SOURCE), "box_angstrom": BOX, "frame_count": N_FRAMES,
        "rcut_angstrom": RCUT, "rcut_smth_angstrom": RCUT_SMTH,
        "descriptor_name": "DeepMD environment matrix R_i (descriptor input)",
        "descriptor_formula": "R_ij=[s(r_ij),s(r_ij)x_ij/r_ij,s(r_ij)y_ij/r_ij,s(r_ij)z_ij/r_ij] in molecule-local frame",
        "matrix_shape": [2,4], "environment_matrix_frame_0": matrices[0].tolist(),
        "cartesian_frame_0": frames[0].round(8).tolist(),
        "cartesian_frame_mid": frames[N_FRAMES//2].round(8).tolist(),
        "cartesian_frame_last": frames[-1].round(8).tolist(),
        "cartesian_max_abs_change": float(np.max(np.abs(frames-frames[0]))),
        "environment_matrix_max_abs_delta": float(np.max(np.abs(matrices-matrices[0]))),
        "bond_length_min_max_angstrom": [float(bonds.min()), float(bonds.max())],
        "bond_length_max_abs_delta_angstrom": float(np.ptp(bonds, axis=0).max()),
        "cos_angle_min_max": [float(angles.min()), float(angles.max())],
        "cos_angle_max_abs_delta": float(np.ptp(angles)),
        "self_check": {"rigid_bonds_pass": bool(np.ptp(bonds,axis=0).max()<1e-10), "rigid_angle_pass": bool(np.ptp(angles)<1e-10), "descriptor_invariance_pass": bool(np.max(np.abs(matrices-matrices[0]))<1e-10)},
    }
    META.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    return {"mattervis": mattervis, "matrix": matrices[0], "payload": payload}


def _table(ax: plt.Axes, registry: LayoutRegistry, x0: float, y0: float, width: float, height: float, values: np.ndarray, rows: list[str], cols: list[str], title: str, colour: str, precision: int = 3) -> None:
    registry.text(ax, x0+width/2, y0+height+0.045, title, ha="center", va="bottom", fontsize=11, color=NAVY, weight="bold")
    cw, ch = width/(len(cols)+1), height/(len(rows)+1)
    for col,label in enumerate([""]+cols): registry.text(ax,x0+(col+0.5)*cw,y0+height-0.5*ch,label,ha="center",va="center",fontsize=10,color=DARK_GRAY,weight="bold")
    for row,label in enumerate(rows):
        yy=y0+height-(row+1.5)*ch; registry.text(ax,x0+0.5*cw,yy,label,ha="center",va="center",fontsize=10,color=DARK_GRAY,weight="bold")
        for col in range(len(cols)):
            xx=x0+(col+1.5)*cw; ax.add_patch(Rectangle((xx-cw/2+0.003,yy-ch/2+0.003),cw-0.006,ch-0.006,fc="#F7FAFA",ec=LINE_GRAY,lw=0.6,zorder=3)); registry.text(ax,xx,yy,f"{values[row,col]:.{precision}f}",ha="center",va="center",fontsize=10,color=colour,zorder=4)


def _compose(fig: plt.Figure, t: float, registry: LayoutRegistry, frames: np.ndarray, asset: dict[str, object], *, video: bool) -> list[dict]:
    _rail,left,right=story_axes(fig); panel_box(left,registry,"RIGID WATER IN A BOX",video=video); panel_box(right,registry,"CARTESIAN + Rᵢ",video=video)
    frame=min(int(round(t*FPS)),N_FRAMES-1); place_main(left,asset["mattervis"]/f"frame_{frame:04d}.png",rect=(0.07,0.13,0.93,0.86))
    cart=frames[frame]; matrix=_environment_matrix(cart); registry.text(left,0.05,0.055,f"frame {frame:03d} · rigid translation + rotation",ha="left",va="bottom",fontsize=11 if video else 10,color=DARK_GRAY)
    _table(right,registry,0.08,0.54,0.84,0.22,cart,["O","H₁","H₂"],["x","y","z"],"r (Å)",NAVY)
    _table(right,registry,0.08,0.20,0.84,0.20,matrix,["H₁","H₂"],["s","x","y","z"],"Rᵢ (DeepMD)",EMERALD,precision=2)
    delta=float(np.max(np.abs(matrix-np.asarray(asset["matrix"])))); registry.text(right,0.50,0.125,"global Cartesian coordinates change",ha="center",va="center",fontsize=10,color=DARK_GRAY); registry.text(right,0.50,0.080,"Rᵢ = [s, sx/r, sy/r, sz/r]",ha="center",va="center",fontsize=10,color=DARK_GRAY); registry.text(right,0.50,0.040,"max |ΔRᵢ| = %.2e · rigid body"%delta,ha="center",va="center",fontsize=10,color=EMERALD,weight="bold")
    return [{"id":"descriptor","color":EMERALD,"min_pixels":80},{"id":"box","color":"#8E8E8E","min_pixels":80}]


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--static-only",action="store_true"); args=parser.parse_args(); frames,symbols,_ids=_make_trajectory(); _write_source(frames,symbols); asset=_render_assets(frames); fig=new_static_figure(); reg=LayoutRegistry(min_font_pt=10,max_font_pt=16,edge_pad_px=18); _compose(fig,0,reg,frames,asset,video=False); save_static(fig,STEM)
    if args.static_only: return
    audit={"panels":[{"id":"structure","rect":[0.29,0.075,0.755,0.905],"min_clearance_px":0,"allow_touch_edges":["left","right","top","bottom"]},{"id":"descriptor","rect":[0.775,0.105,0.970,0.905],"min_clearance_px":0,"allow_touch_edges":["left","right","top","bottom"]}],"whitespace":{"background_threshold":245,"min_ink_fraction":0.012,"min_panel_bbox_fill":0.16,"grid_rows":12,"grid_columns":24},"bands":[{"id":"gap","rect":[0.755,0.075,0.775,0.905],"max_ink_pixels":5000}]}
    render_video(stem=STEM,duration_seconds=10.0,draw_frame=lambda f,t,i,r:_compose(f,t,r,frames,asset,video=True),audit_config=audit,qa_directory=QA_DIR/"_qa",representative_times=[0.5,2.5,5.0,7.5,9.5])


if __name__ == "__main__": main()

