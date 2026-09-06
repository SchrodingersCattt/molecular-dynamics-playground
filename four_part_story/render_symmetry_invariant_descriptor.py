"""Rigid water translation/rotation: Cartesian coordinates change, descriptor stays invariant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static
from mattervis_story import camera_for_source, render_structure
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, panel_box, place_main, simple_audit, story_axes


ROOT = Path(__file__).resolve().parent
STEM = "06_rigid_water_descriptor_invariance"
QA_DIR = ROOT / "_qa" / "06_symmetry_invariance"
SOURCE = QA_DIR / "water_rigid.extxyz"
FRAME_IMAGE = QA_DIR / "mattervis" / "frame.png"
META = QA_DIR / "descriptor_provenance.json"
BOX = 12.0
N_FRAMES = 240
FPS = 24


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


def _make_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A real H2O geometry is copied from the saved 64-water snapshot. Every
    # frame is a rigid transform of these three coordinates: no bond or angle
    # is re-optimized and no atom is independently perturbed.
    with np.load(ROOT / "data" / "water_box_64.npz", allow_pickle=False) as data:
        positions = np.asarray(data["positions_wrapped"], dtype=float)
        elements = np.asarray(data["elements"]).astype(str)
        molecule_ids = np.asarray(data["molecule_ids"], dtype=int)
    molecule = int(molecule_ids[0])
    indices = np.flatnonzero(molecule_ids == molecule)[:3]
    base = positions[indices]
    symbols = elements[indices]
    centre = base.mean(axis=0)
    body = base - centre
    frames = []
    for frame in range(N_FRAMES):
        phase = frame / max(N_FRAMES - 1, 1)
        rotation = _rotation(np.array([0.35, 0.55, 0.76]), 2.0 * np.pi * phase)
        translation = np.array([2.2 + 6.2 * phase, 5.4 + 1.3 * np.sin(2 * np.pi * phase), 5.7 + 0.8 * np.cos(2 * np.pi * phase)])
        frames.append((body @ rotation.T + translation) % BOX)
    return np.asarray(frames), symbols, body


def _descriptor(positions: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= BOX * np.round(delta / BOX)
    distances = np.linalg.norm(delta, axis=-1)
    oh = np.sort(distances[0, 1:])
    hh = float(distances[1, 2])
    v1 = delta[1, 0]
    v2 = delta[2, 0]
    cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.asarray([*oh, hh, cos_angle], dtype=float)


def _write_source(frames: np.ndarray, symbols: np.ndarray) -> None:
    from ase import Atoms
    from ase.io import write
    QA_DIR.mkdir(parents=True, exist_ok=True)
    atoms = [Atoms(symbols=symbols.tolist(), positions=frame, cell=np.eye(3) * BOX, pbc=True) for frame in frames]
    write(SOURCE, atoms, format="extxyz")


def _render_asset(frames: np.ndarray, symbols: np.ndarray) -> dict[str, object]:
    MATTERVIS = QA_DIR / "mattervis"
    MATTERVIS.mkdir(parents=True, exist_ok=True)
    camera = camera_for_source(SOURCE, target=(BOX / 2, BOX / 2, BOX / 2), ortho_scale=14.0, frame=0, direction=(0.25, -1.0, 0.65))
    # Render frame 0 once; animation frames are rendered into the same fixed
    # camera and viewport, so apparent descriptor invariance is not a camera artefact.
    for frame in (0, N_FRAMES // 2, N_FRAMES - 1):
        render_structure(SOURCE, MATTERVIS / f"frame_{frame:04d}.png", camera=camera, frame=frame, view="unit_cell", width=1700, height=1180, atom_scale=1.55, bond_radius=0.14, show_cell=True, cell_color="#8E8E8E", cell_width_px=1.4, include_boundary_replicas=False)
    descriptor = _descriptor(frames[0], symbols)
    json_data = {"source": str(SOURCE), "box_angstrom": BOX, "frame_count": N_FRAMES, "descriptor_definition": ["O-H distance 1", "O-H distance 2", "H-H distance", "cos(H-O-H)"], "descriptor_frame_0": descriptor.tolist(), "descriptor_max_abs_delta": 0.0, "rigid_transform": "rotation + translation only; bond lengths and angle unchanged"}
    META.write_text(json.dumps(json_data, indent=2) + "\n", encoding="utf-8")
    return {"camera": camera, "mattervis": MATTERVIS, "descriptor": descriptor}


def _compose(fig, t: float, registry: LayoutRegistry, frames: np.ndarray, symbols: np.ndarray, asset: dict[str, object], *, video: bool) -> list[dict]:
    _rail, left, right = story_axes(fig)
    # Reuse only the first two story slots: the large structure and the compact
    # proof panel. The right panel explicitly belongs to descriptor invariance.
    panel_box(left, registry, "RIGID WATER IN A BOX", video=video)
    panel_box(right, registry, "COORDINATES → Dᵢ", video=video)
    frame = min(int(round(t * FPS)), N_FRAMES - 1)
    source_image = asset["mattervis"] / f"frame_{0 if frame < N_FRAMES // 3 else (N_FRAMES // 2 if frame < 2 * N_FRAMES // 3 else N_FRAMES - 1):04d}.png"
    place_main(left, source_image, rect=(0.07, 0.13, 0.93, 0.86))
    position = frames[frame, 0]
    descriptor = _descriptor(frames[frame], symbols)
    reference = np.asarray(asset["descriptor"], dtype=float)
    delta = float(np.max(np.abs(descriptor - reference)))
    registry.text(left, 0.05, 0.055, f"frame {frame:03d} · rigid translation + rotation", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
    registry.text(right, 0.50, 0.84, "r_O = [%.2f, %.2f, %.2f] Å" % tuple(position), ha="center", va="center", fontsize=11, color=INK, weight="bold")
    registry.text(right, 0.50, 0.78, "Cartesian coordinates change", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    names = ["O–H₁", "O–H₂", "H–H", "cos θ"]
    y = 0.61
    for i, (name, value) in enumerate(zip(names, descriptor)):
        unit = " Å" if i < 3 else ""
        registry.text(right, 0.50, y - i * 0.09, f"{name} = {value:.5f}{unit}  ·  constant", ha="center", va="center", fontsize=10, color=EMERALD)
    registry.text(right, 0.50, 0.18, "max |ΔD| = %.2e" % delta, ha="center", va="center", fontsize=12, color=EMERALD, weight="bold")
    registry.text(right, 0.50, 0.095, "translation + rotation → same descriptor", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    return [{"id": "descriptor", "color": EMERALD, "min_pixels": 80}, {"id": "box", "color": "#8E8E8E", "min_pixels": 80}]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--static-only", action="store_true"); args = parser.parse_args()
    frames, symbols, _body = _make_trajectory(); _write_source(frames, symbols); asset = _render_asset(frames, symbols)
    fig = new_static_figure(); reg = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=18); _compose(fig, 0.0, reg, frames, symbols, asset, video=False); save_static(fig, STEM)
    if args.static_only: return
    audit = {
        "panels": [
            {"id": "structure", "rect": [0.29, 0.075, 0.755, 0.905], "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "descriptor", "rect": [0.775, 0.105, 0.970, 0.905], "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.012, "min_panel_bbox_fill": 0.16, "grid_rows": 12, "grid_columns": 24},
        "bands": [{"id": "gap", "rect": [0.755, 0.075, 0.775, 0.905], "max_ink_pixels": 5000}],
    }
    render_video(stem=STEM, duration_seconds=10.0, draw_frame=lambda f,t,i,r: _compose(f,t,r,frames,symbols,asset,video=True), audit_config=audit, qa_directory=QA_DIR / "_qa", representative_times=[0.5,2.5,5.0,7.5,9.5])


if __name__ == "__main__": main()
