"""Responsive O--O Lennard--Jones story rendered with MatterVis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static, smoothstep
from mattervis_story import camera_for_source, draw_world_segment, make_vector_group, render_structure, write_provenance_index
from responsive_story import (
    EMERALD,
    LAKE_BLUE,
    PALE_OLIVE,
    draw_horizontal_key,
    panel_box,
    place_main,
    simple_audit,
    stage_rail,
    story_axes,
    timeline_stage,
)

ROOT = Path(__file__).resolve().parent
STEM = "02_classical_lj"
QA_DIR = ROOT / "_qa" / STEM
MATTERVIS_DIR = QA_DIR / "source" / "mattervis_v3"
SOURCE = ROOT / "data" / "classical_lj_motion.extxyz"
ARROW_STYLE = {
    "shaft_radius": 0.020,
    "head_length": 0.070,
    "head_radius": 0.046,
    "sides": 18,
}


def load_data() -> dict[str, np.ndarray]:
    with np.load(ROOT / "data" / "classical_lj.npz", allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def prepare_mattervis(data: dict[str, np.ndarray]) -> dict[str, object]:
    MATTERVIS_DIR.mkdir(parents=True, exist_ok=True)
    frame_count = len(data["motion_atomic_positions"])
    plain = [MATTERVIS_DIR / f"motion_{frame:02d}.png" for frame in range(frame_count)]
    position = [MATTERVIS_DIR / f"position_{frame:02d}.png" for frame in range(frame_count)]
    force_path = MATTERVIS_DIR / "lj_force.png"
    velocity_path = MATTERVIS_DIR / "velocity.png"
    cached = (plain[0], plain[-1], position[0], position[-1], force_path, velocity_path)
    def is_current(path: Path) -> bool:
        try:
            return path.exists() and Image.open(path).size == (1700, 1180)
        except Exception:
            return False
    if all(is_current(path) for path in cached):
        target = np.mean(data["motion_atomic_positions"], axis=(0, 1))
        camera = camera_for_source(
            SOURCE, target=target, ortho_scale=1.50, frame=0,
            direction=(0.12, -0.82, 0.56), up=(0.0, 0.0, 1.0),
        )
        return {"plain": plain, "position": position, "force": force_path, "velocity": velocity_path, "camera": camera}

    target = np.mean(data["motion_atomic_positions"], axis=(0, 1))
    # Look across (rather than along) the O--O axis so the interaction line
    # remains legible and does not run through an H--O stick in projection.
    camera = camera_for_source(
        SOURCE, target=target, ortho_scale=1.50, frame=0,
        direction=(0.12, -0.82, 0.56), up=(0.0, 0.0, 1.0),
    )
    q0 = data["motion_oxygen_positions"][0]
    displacement = data["motion_oxygen_positions"][-1] - q0
    position_vectors = make_vector_group("lj-position", q0, displacement,
                                         scale=float(data["display_displacement_scale"]), color=LAKE_BLUE,
                                         style=ARROW_STYLE)
    # Two equal-and-opposite vectors are anchored directly at the two O atoms;
    # this makes the displayed term unambiguously O--O rather than molecular.
    force_vectors = make_vector_group("oo-lj-force", data["oxygen_positions"][1], data["molecule_forces"][1],
                                      scale=float(data["display_force_scale"]), color=PALE_OLIVE,
                                      style=ARROW_STYLE)
    velocity_vectors = make_vector_group("oo-velocity", data["oxygen_positions"][1], data["molecule_velocities"][1],
                                         scale=float(data["display_velocity_scale"]), color=EMERALD,
                                         style=ARROW_STYLE)
    records: list[dict] = []
    for frame in range(frame_count):
        p = MATTERVIS_DIR / f"motion_{frame:02d}.png"
        q = MATTERVIS_DIR / f"position_{frame:02d}.png"
        records.append(render_structure(SOURCE, p, camera=camera, frame=frame, width=1700, height=1180,
                                        atom_scale=1.06, bond_radius=0.135))
        records.append(render_structure(SOURCE, q, camera=camera, frame=frame, width=1700, height=1180,
                                        atom_scale=1.06, bond_radius=0.135, vector_overlays=position_vectors))
    final = frame_count - 1
    records.append(render_structure(SOURCE, force_path, camera=camera, frame=final, width=1700, height=1180,
                                    atom_scale=1.06, bond_radius=0.135, vector_overlays=force_vectors))
    records.append(render_structure(SOURCE, velocity_path, camera=camera, frame=final, width=1700, height=1180,
                                    atom_scale=1.06, bond_radius=0.135, vector_overlays=velocity_vectors))
    write_provenance_index(MATTERVIS_DIR, records)
    return {"plain": plain, "position": position, "force": force_path, "velocity": velocity_path, "camera": camera}


def draw_info(ax, registry: LayoutRegistry, data: dict[str, np.ndarray], *, video: bool, active: int | None, returning: bool = False) -> None:
    panel_box(ax, registry, "TIP3P O–O LJ TERM", video=video)
    current_r = float(np.asarray(data["oo_separations_angstrom"]).reshape(-1)[-1])
    cards = [
        (0.76, r"$r_{\mathrm{OO}}$", f"{current_r:.2f} Å · current", NAVY),
        (0.57, r"$U_{\mathrm{LJ}}(r_{\mathrm{OO}})$", "pair energy", PALE_OLIVE),
        (0.38, r"$\mathbf{F}_{\mathrm{OO}}$", "equal + opposite", PALE_OLIVE),
    ]
    for idx, (y, value, label, colour) in enumerate(cards):
        selected = active == idx
        ax.add_patch(FancyBboxPatch((0.08, y - 0.070), 0.84, 0.14,
                                    boxstyle="round,pad=0.010,rounding_size=0.02",
                                    facecolor="#EAF2EE" if selected else "#F7F8F6",
                                    edgecolor=colour if selected else LINE_GRAY,
                                    linewidth=2.2 if video else 1.4, zorder=2))
        registry.text(ax, 0.17, y + 0.020, value, ha="left", va="center",
                      fontsize=12 if video else 11, color=colour, weight="bold", zorder=3)
        registry.text(ax, 0.17, y - 0.032, label, ha="left", va="center",
                      fontsize=10 if video else 10, color=INK, zorder=3)
    sigma = float(np.asarray(data["sigma_angstrom"]).reshape(-1)[0])
    eps = float(np.asarray(data["epsilon_ev"]).reshape(-1)[0])
    registry.text(ax, 0.50, 0.19, f"σ = {sigma:.2f} Å", ha="center", va="center",
                  fontsize=11 if video else 10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.12, f"ε = {eps:.4f} eV", ha="center", va="center",
                  fontsize=11 if video else 10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.055, "electrostatics omitted", ha="center", va="center",
                  fontsize=10, color=DARK_GRAY)
    if returning:
        registry.text(ax, 0.50, 0.015, r"$n\;\rightarrow\;n+1$", ha="center", va="bottom",
                      fontsize=10, color=NAVY, weight="bold")


def draw_main(ax, registry: LayoutRegistry, data: dict[str, np.ndarray], scenes: dict[str, object], *, camera, stage: int | None, progress: float, returning: bool, video: bool) -> None:
    titles = ("measure O···O distance", "evaluate O–O LJ force", "carry velocity")
    panel_box(ax, registry, "TIP3P WATER DIMER" if returning else f"TIP3P WATER DIMER · {titles[stage]}", video=video)
    # The dimer source renders retain generous transparent margins.  A wider
    # placement rect intentionally crops only those margins so the two real
    # water molecules occupy the dominant central area.
    fitted_rect = (0.02, 0.18, 0.98, 0.82)
    if returning:
        registry.text(ax, 0.035, 0.035, "next step · pause", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(ax, scenes["plain"][-1], alpha=0.16, rect=fitted_rect)
        fitted = place_main(ax, scenes["force"], rect=fitted_rect)
        draw_world_segment(ax, data["oxygen_positions"][1][0], data["oxygen_positions"][1][1], camera=camera,
                           rect=fitted, color=NAVY, linewidth=5.0 if video else 3.0, image_aspect=1700.0 / 1180.0)
        return
    if stage == 0:
        registry.text(ax, 0.035, 0.035, "O···O separation", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(ax, scenes["plain"][0], alpha=0.18, rect=fitted_rect)
        idx = int(round(smoothstep(progress) * (len(scenes["position"]) - 1)))
        fitted = place_main(ax, scenes["position"][idx], rect=fitted_rect)
        oxy = data["motion_oxygen_positions"][idx]
        xy = draw_world_segment(ax, oxy[0], oxy[1], camera=camera, rect=fitted, color=NAVY,
                                linewidth=5.0 if video else 3.0, image_aspect=1700.0 / 1180.0)
        registry.text(ax, float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]) + 0.045), r"$r_{\mathrm{OO}}$",
                      ha="center", va="bottom", fontsize=13 if video else 11, color=NAVY, weight="bold")
    elif stage == 1:
        registry.text(ax, 0.035, 0.035, "equal + opposite O forces", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(ax, scenes["plain"][-1], alpha=0.15, rect=fitted_rect)
        fitted = place_main(ax, scenes["force"], rect=fitted_rect)
        draw_world_segment(ax, data["oxygen_positions"][1][0], data["oxygen_positions"][1][1], camera=camera,
                           rect=fitted, color=NAVY, linewidth=5.0 if video else 3.0, image_aspect=1700.0 / 1180.0)
    else:
        registry.text(ax, 0.035, 0.035, "force → acceleration → velocity", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(ax, scenes["plain"][0], alpha=0.15, rect=fitted_rect)
        fitted = place_main(ax, scenes["velocity"], rect=fitted_rect)
        draw_world_segment(ax, data["oxygen_positions"][1][0], data["oxygen_positions"][1][1], camera=camera,
                           rect=fitted, color=NAVY, linewidth=5.0 if video else 3.0, image_aspect=1700.0 / 1180.0)


def compose(fig, t: float, registry: LayoutRegistry, data: dict[str, np.ndarray], scenes: dict[str, object], *, video: bool) -> list[dict]:
    stage, progress, returning = timeline_stage(t, 9.0, n_stages=3, return_seconds=1.5)
    rail, main, info = story_axes(fig)
    stage_rail(rail, registry, active=None if returning else stage, video=video, equation=None, return_phase=returning)
    draw_main(main, registry, data, scenes, camera=scenes["camera"], stage=stage, progress=progress, returning=returning, video=video)
    draw_info(info, registry, data, video=video, active=None if returning else stage, returning=returning)
    draw_horizontal_key(
        main, registry,
        (("$\\Delta r_{OO}$", LAKE_BLUE), ("$F_{OO}$", PALE_OLIVE), ("$v$", EMERALD)),
        video=video, y=0.075,
    )
    return [{"id": "O-O LJ", "color": PALE_OLIVE, "min_pixels": 120}, {"id": "distance", "color": NAVY, "min_pixels": 100}]


def render_static(data: dict[str, np.ndarray], scenes: dict[str, object]) -> None:
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10.0, max_font_pt=16.0, edge_pad_px=18)
    # The paper still freezes the instructional centre on the interaction
    # itself (stage 1), while the movie starts with the distance measurement.
    compose(fig, 3.0, registry, data, scenes, video=False)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("static responsive layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def render_animation(data: dict[str, np.ndarray], scenes: dict[str, object]) -> None:
    render_video(stem=STEM, duration_seconds=9.0,
                 draw_frame=lambda fig, t, i, reg: compose(fig, t, reg, data, scenes, video=True),
                 audit_config=simple_audit(("rail", "structure", "lj_info")),
                 qa_directory=QA_DIR / "_qa", representative_times=[0.2, 2.8, 5.4, 8.4])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    scenes = prepare_mattervis(data)
    render_static(data, scenes)
    if not args.static_only:
        render_animation(data, scenes)


if __name__ == "__main__":
    main()
