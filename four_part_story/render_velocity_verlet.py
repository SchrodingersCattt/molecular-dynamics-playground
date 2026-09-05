"""Responsive Velocity--Verlet story rendered with MatterVis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static, smoothstep
from mattervis_story import camera_for_source, make_vector_group, render_structure, write_provenance_index
from responsive_story import (
    EMERALD,
    LAKE_BLUE,
    PALE_OLIVE,
    draw_horizontal_key,
    panel_box,
    place_main,
    simple_audit,
    stage_rail,
    stage_title,
    story_axes,
    timeline_stage,
)

ROOT = Path(__file__).resolve().parent
STEM = "01_velocity_verlet"
QA_DIR = ROOT / "_qa" / STEM
MATTERVIS_DIR = QA_DIR / "source" / "mattervis_v3"
SOURCE = ROOT / "data" / "vv_h2o_motion.extxyz"

ARROW_STYLE = {
    # World-space dimensions keep the head outside the atom sphere without
    # making a vector look like a second atom.
    "shaft_radius": 0.020,
    "head_length": 0.070,
    "head_radius": 0.046,
    "sides": 18,
}


def load_data() -> dict[str, np.ndarray]:
    with np.load(ROOT / "data" / "vv_h2o_step.npz", allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def prepare_mattervis(data: dict[str, np.ndarray]) -> dict[str, object]:
    """Create fixed-camera MatterVis renders for trajectory and three vectors."""
    MATTERVIS_DIR.mkdir(parents=True, exist_ok=True)
    frame_count = len(data["motion_positions"])
    cached_plain = [MATTERVIS_DIR / f"motion_{frame:02d}.png" for frame in range(frame_count)]
    cached_position = [MATTERVIS_DIR / f"position_{frame:02d}.png" for frame in range(frame_count)]
    cached_accel = MATTERVIS_DIR / "acceleration.png"
    cached_velocity = MATTERVIS_DIR / "velocity.png"
    if all(path.exists() for path in (cached_plain[0], cached_plain[-1], cached_position[0], cached_position[-1], cached_accel, cached_velocity)):
        return {"plain": cached_plain, "position": cached_position, "acceleration": cached_accel, "velocity": cached_velocity, "overview": cached_velocity}
    target = np.mean(data["motion_positions"], axis=(0, 1))
    # Tight orthographic framing suppresses empty paper-space while retaining
    # the same camera for every trajectory frame.
    camera = camera_for_source(SOURCE, target=target, ortho_scale=1.16, frame=0)
    displacement = data["positions"][1] - data["positions"][0]
    position_vectors = make_vector_group(
        "vv-position", data["motion_positions"][0], displacement,
        scale=float(data["display_displacement_scale"]), color=LAKE_BLUE,
        style=ARROW_STYLE,
    )
    acceleration_vectors = make_vector_group(
        "vv-acceleration", data["positions"][1], data["accelerations"][1],
        scale=float(data["display_acceleration_scale"]), color=PALE_OLIVE,
        style=ARROW_STYLE,
    )
    velocity_vectors = make_vector_group(
        "vv-velocity", data["positions"][1], data["velocities"][1],
        scale=float(data["display_velocity_scale"]), color=EMERALD,
        style=ARROW_STYLE,
    )
    plain: list[Path] = []
    position: list[Path] = []
    records: list[dict] = []
    for frame in range(frame_count):
        plain_path = MATTERVIS_DIR / f"motion_{frame:02d}.png"
        pos_path = MATTERVIS_DIR / f"position_{frame:02d}.png"
        records.append(render_structure(
            SOURCE, plain_path, camera=camera, frame=frame, view="cluster",
            width=1700, height=1180, atom_scale=1.12, bond_radius=0.14,
        ))
        records.append(render_structure(
            SOURCE, pos_path, camera=camera, frame=frame, view="cluster",
            width=1700, height=1180, atom_scale=1.12, bond_radius=0.14,
            vector_overlays=position_vectors,
        ))
        plain.append(plain_path)
        position.append(pos_path)
    final = len(plain) - 1
    acceleration_path = MATTERVIS_DIR / "acceleration.png"
    velocity_path = MATTERVIS_DIR / "velocity.png"
    for path, vectors in (
        (acceleration_path, acceleration_vectors),
        (velocity_path, velocity_vectors),
    ):
        records.append(render_structure(
            SOURCE, path, camera=camera, frame=final, view="cluster",
            width=1700, height=1180, atom_scale=1.12, bond_radius=0.14,
            vector_overlays=vectors,
        ))
    write_provenance_index(MATTERVIS_DIR, records)
    return {
        "plain": plain,
        "position": position,
        "acceleration": acceleration_path,
        "velocity": velocity_path,
        "overview": velocity_path,
    }


def draw_info(ax, registry: LayoutRegistry, *, video: bool, active: int | None, returning: bool = False) -> None:
    panel_box(ax, registry, "STATE", video=video)
    cards = [
        (0.72, "known", r"$\mathbf{r}_n,\;\mathbf{v}_n,\;\mathbf{a}_n$", LAKE_BLUE),
        (0.49, "new", r"$\mathbf{r}_{n+1},\;\mathbf{v}_{n+1},\;\mathbf{a}_{n+1}$", EMERALD),
    ]
    for index, (y, label, value, colour) in enumerate(cards):
        selected = active == index
        ax.add_patch(FancyBboxPatch(
            (0.08, y - 0.095), 0.84, 0.19,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#EAF2EE" if selected else "#F7F8F6",
            edgecolor=colour if selected else LINE_GRAY,
            linewidth=2.3 if video else 1.5,
            zorder=2,
        ))
        registry.text(ax, 0.18, y + 0.035, label, ha="left", va="center",
                      fontsize=12 if video else 11, color=colour, weight="bold", zorder=3)
        registry.text(ax, 0.18, y - 0.035, value, ha="left", va="center",
                      fontsize=11 if video else 10, color=INK, zorder=3)
    registry.text(ax, 0.50, 0.24, r"$\Delta t=0.5\;\mathrm{fs}$", ha="center", va="center",
                  fontsize=12 if video else 11, color=NAVY, weight="bold")
    registry.text(ax, 0.50, 0.15, "one real H₂O step", ha="center", va="center",
                  fontsize=11 if video else 10, color=DARK_GRAY)
    if returning:
        registry.text(ax, 0.50, 0.075, r"$n\;\rightarrow\;n+1$", ha="center", va="center",
                      fontsize=12 if video else 11, color=NAVY, weight="bold")


def draw_composition(fig, t: float, registry: LayoutRegistry, data: dict[str, np.ndarray], scenes: dict[str, object], *, video: bool) -> list[dict]:
    del data
    stage, progress, returning = timeline_stage(t, 9.0, n_stages=3, return_seconds=1.5)
    rail, main, info = story_axes(fig)
    active = None if returning else stage
    stage_rail(
        rail, registry, active=active, video=video,
        equation=None,
        return_phase=returning,
    )
    main_titles = ("update position", "evaluate acceleration", "update velocity")
    panel_box(main, registry, "next step" if returning else f"VELOCITY–VERLET · H₂O · {main_titles[stage]}", video=video)
    if returning:
        registry.text(main, 0.035, 0.035, "pause · then repeat", ha="left", va="bottom",
                      fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, scenes["plain"][-1], alpha=0.18)
        place_main(main, scenes["acceleration"], alpha=0.92)
        place_main(main, scenes["velocity"])
    elif stage == 0:
        registry.text(main, 0.035, 0.035, "old → new geometry", ha="left", va="bottom",
                      fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, scenes["plain"][0], alpha=0.20)
        index = int(round(smoothstep(progress) * (len(scenes["position"]) - 1)))
        place_main(main, scenes["position"][index])
    elif stage == 1:
        registry.text(main, 0.035, 0.035, "force / mass", ha="left", va="bottom",
                      fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, scenes["plain"][0], alpha=0.16)
        place_main(main, scenes["acceleration"])
    else:
        registry.text(main, 0.035, 0.035, "new acceleration closes loop", ha="left", va="bottom",
                      fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, scenes["plain"][0], alpha=0.16)
        place_main(main, scenes["velocity"])
    draw_info(info, registry, video=video, active=None if returning else (1 if stage == 2 else 0), returning=returning)
    draw_horizontal_key(
        main, registry,
        (("$\\Delta r$", LAKE_BLUE), ("$F/m$", PALE_OLIVE), ("$v$", EMERALD)),
        video=video, y=0.075,
    )
    return [
        {"id": "position", "color": LAKE_BLUE, "min_pixels": 120},
        {"id": "acceleration", "color": PALE_OLIVE, "min_pixels": 120},
        {"id": "velocity", "color": EMERALD, "min_pixels": 120},
    ]


def render_static(data: dict[str, np.ndarray], scenes: dict[str, object]) -> None:
    del data
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10.0, max_font_pt=16.0, edge_pad_px=18)
    draw_composition(fig, 0.0, registry, {}, scenes, video=False)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("static responsive layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def render_animation(data: dict[str, np.ndarray], scenes: dict[str, object]) -> None:
    render_video(
        stem=STEM,
        duration_seconds=9.0,
        draw_frame=lambda fig, t, i, reg: draw_composition(fig, t, reg, data, scenes, video=True),
        audit_config=simple_audit(("rail", "structure", "state")),
        qa_directory=QA_DIR / "_qa",
        representative_times=[0.2, 2.8, 5.4, 8.4],
    )


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
