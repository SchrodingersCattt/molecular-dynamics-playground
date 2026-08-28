from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    new_static_figure,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    STORY_STATIC_A,
    STORY_STATIC_B,
    STORY_STATIC_C,
    STORY_STATIC_D,
    STORY_VIDEO_A,
    STORY_VIDEO_B,
    STORY_VIDEO_C,
    STORY_VIDEO_D,
    camera_for_source,
    draw_vv_loop,
    make_vector_group,
    place_render,
    render_structure,
    write_provenance_index,
)


ROOT = Path(__file__).resolve().parent
STEM = "01_velocity_verlet"
QA_DIR = ROOT / "_qa" / "01_velocity_verlet"
MATTERVIS_DIR = QA_DIR / "source" / "mattervis"
SCENE_RECT = (0.04, 0.09, 0.96, 0.91)
EQUATIONS = (
    r"$\mathbf{r}_{n+1}=\mathbf{r}_n+\mathbf{v}_n\Delta t+\frac{1}{2}\mathbf{a}_n\Delta t^2$",
    r"$\mathbf{a}_{n+1}=\mathbf{F}(\mathbf{r}_{n+1})/m$",
    r"$\mathbf{v}_{n+1}=\mathbf{v}_n+\frac{1}{2}(\mathbf{a}_n+\mathbf{a}_{n+1})\Delta t$",
)


def load_data() -> dict[str, np.ndarray]:
    source = np.load(ROOT / "data" / "vv_h2o_step.npz", allow_pickle=True)
    return {key: source[key] for key in source.files}


def prepare_mattervis(data: dict[str, np.ndarray]) -> dict[str, list[Path] | Path]:
    source = ROOT / "data" / "vv_h2o_motion.extxyz"
    target = np.mean(data["positions"], axis=(0, 1))
    plain_paths: list[Path] = []
    position_paths: list[Path] = []
    records: list[dict] = []
    displacement_overlays = make_vector_group(
        "displacement",
        data["positions"][0],
        data["positions"][1] - data["positions"][0],
        scale=float(data["display_displacement_scale"]),
        color=NAVY,
    )
    for frame in range(len(data["motion_positions"])):
        camera = camera_for_source(source, target=target, ortho_scale=1.55, frame=frame)
        plain_path = MATTERVIS_DIR / f"motion_{frame:02d}.png"
        position_path = MATTERVIS_DIR / f"position_{frame:02d}.png"
        records.append(render_structure(source, plain_path, camera=camera, frame=frame))
        records.append(
            render_structure(
                source,
                position_path,
                camera=camera,
                frame=frame,
                vector_overlays=displacement_overlays,
            )
        )
        plain_paths.append(plain_path)
        position_paths.append(position_path)
    final_frame = len(data["motion_positions"]) - 1
    final_camera = camera_for_source(
        source,
        target=target,
        ortho_scale=1.55,
        frame=final_frame,
    )
    acceleration_path = MATTERVIS_DIR / "acceleration.png"
    velocity_path = MATTERVIS_DIR / "velocity.png"
    records.append(
        render_structure(
            source,
            acceleration_path,
            camera=final_camera,
            frame=final_frame,
            vector_overlays=make_vector_group(
                "acceleration",
                data["positions"][1],
                data["accelerations"][1],
                scale=float(data["display_acceleration_scale"]),
                color=CRIMSON,
            ),
        )
    )
    records.append(
        render_structure(
            source,
            velocity_path,
            camera=final_camera,
            frame=final_frame,
            vector_overlays=make_vector_group(
                "velocity",
                data["positions"][1],
                data["velocities"][1],
                scale=float(data["display_velocity_scale"]),
                color=GREEN,
            ),
        )
    )
    write_provenance_index(MATTERVIS_DIR, records)
    return {
        "plain": plain_paths,
        "position": position_paths,
        "acceleration": acceleration_path,
        "velocity": velocity_path,
    }


def draw_left(
    ax,
    registry: LayoutRegistry,
    *,
    video: bool,
    active: int | None,
) -> None:
    centre_text = EQUATIONS[active] if active is not None else "\n".join(EQUATIONS)
    draw_vv_loop(
        ax,
        registry,
        video=video,
        active_stage=active,
        centre_text=centre_text,
        centre_y=0.52,
        radius_x=0.40,
    )


def draw_state_panel(
    ax,
    registry: LayoutRegistry,
    *,
    video: bool,
    updated: bool,
    active: int | None,
) -> None:
    ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=2.0 if video else 1.1))
    registry.text(
        ax,
        0.50,
        0.84,
        "updated state" if updated else "known state",
        ha="center",
        va="center",
        fontsize=14 if video else 10,
        color=INK,
    )
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    aspect = (position.width * figure_width) / (position.height * figure_height)
    symbols = (r"$\mathbf{r}$", r"$\mathbf{a}$", r"$\mathbf{v}$")
    xs = (0.22, 0.50, 0.78)
    suffix = r"_{n+1}" if updated else r"_n"
    for index, (x, symbol) in enumerate(zip(xs, symbols)):
        is_active = bool(updated and active == index)
        ax.add_patch(Ellipse((x, 0.45), 0.20, 0.20 * aspect, fc=INK if is_active else WHITE, ec=INK if is_active else LINE_GRAY, lw=2.2 if video else 1.3))
        registry.text(
            ax,
            x,
            0.45,
            f"${symbol[1:-1]}{suffix}$",
            ha="center",
            va="center",
            fontsize=14 if video else 10,
            color=WHITE if is_active else INK,
            weight="bold" if is_active else "normal",
        )


def draw_stage(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    *,
    stage: int,
    rect: tuple[float, float, float, float],
    video: bool,
    progress: float = 1.0,
) -> None:
    del registry, data, video
    plain_paths = scenes["plain"]
    position_paths = scenes["position"]
    assert isinstance(plain_paths, list) and isinstance(position_paths, list)
    if stage == 0:
        place_render(ax, plain_paths[0], rect, alpha=0.16, zorder=3)
        index = int(round(smoothstep(progress) * (len(position_paths) - 1)))
        place_render(ax, position_paths[index], rect, alpha=1.0, zorder=5)
    elif stage == 1:
        acceleration_path = scenes["acceleration"]
        assert isinstance(acceleration_path, Path)
        place_render(ax, acceleration_path, rect, zorder=5)
    else:
        velocity_path = scenes["velocity"]
        assert isinstance(velocity_path, Path)
        place_render(ax, velocity_path, rect, zorder=5)


def draw_case_panel(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    *,
    stage: int,
    video: bool,
    progress: float = 1.0,
) -> None:
    ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=2.0 if video else 1.1, zorder=20))
    headings = ("Update position", "Evaluate acceleration", "Update velocity")
    registry.text(ax, 0.50, 0.925, headings[stage], ha="center", va="center", fontsize=16 if video else 11, color=INK, zorder=21)
    registry.text(ax, 0.075, 0.070, "Simulation step 01", ha="left", va="bottom", fontsize=12 if video else 10, color=INK, zorder=21)
    draw_stage(ax, registry, data, scenes, stage=stage, rect=SCENE_RECT, video=video, progress=progress)


def render_static(data: dict[str, np.ndarray], scenes: dict[str, list[Path] | Path]) -> None:
    render_source_panel(QA_DIR / "source" / "integrator.png", lambda ax, reg: draw_left(ax, reg, video=False, active=None), width_px=1000, height_px=1500)
    render_source_panel(QA_DIR / "source" / "case.png", lambda ax, reg: draw_case_panel(ax, reg, data, scenes, stage=0, video=False), width_px=1800, height_px=1500)
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    draw_left(axes_from_top_slot(fig, STORY_STATIC_A), registry, video=False, active=None)
    draw_case_panel(axes_from_top_slot(fig, STORY_STATIC_B), registry, data, scenes, stage=0, video=False)
    draw_state_panel(axes_from_top_slot(fig, STORY_STATIC_C), registry, video=False, updated=False, active=None)
    draw_state_panel(axes_from_top_slot(fig, STORY_STATIC_D), registry, video=False, updated=True, active=0)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def draw_video_frame(
    fig,
    time_seconds: float,
    frame_index: int,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
) -> list[dict]:
    del frame_index
    stage = min(int(time_seconds // 3.0), 2)
    progress = (time_seconds - 3.0 * stage) / 3.0
    left = axes_from_top_slot(fig, STORY_VIDEO_A)
    middle = axes_from_top_slot(fig, STORY_VIDEO_B)
    upper_right = axes_from_top_slot(fig, STORY_VIDEO_C)
    lower_right = axes_from_top_slot(fig, STORY_VIDEO_D)
    draw_left(left, registry, video=True, active=stage)
    headings = ("position update", "force → acceleration", "velocity update")
    colors = (NAVY, CRIMSON, GREEN)
    draw_case_panel(middle, registry, data, scenes, stage=stage, video=True, progress=progress)
    draw_state_panel(upper_right, registry, video=True, updated=False, active=None)
    draw_state_panel(lower_right, registry, video=True, updated=True, active=stage)
    return [{"id": headings[stage], "color": colors[stage], "min_pixels": 160}]


def render_animation(data: dict[str, np.ndarray], scenes: dict[str, list[Path] | Path]) -> None:
    audit = {
        "panels": [
            {"id": "integrator", "rect": list(STORY_VIDEO_A), "min_clearance_px": 12},
            {"id": "mattervis_h2o", "rect": list(STORY_VIDEO_B), "min_clearance_px": 12},
            {"id": "known_state", "rect": list(STORY_VIDEO_C), "min_clearance_px": 12},
            {"id": "updated_state", "rect": list(STORY_VIDEO_D), "min_clearance_px": 12},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.018, "min_panel_bbox_fill": 0.20, "grid_rows": 12, "grid_columns": 20},
        "bands": [
            {"id": "gap_a_b", "rect": [0.310, 0.055, 0.325, 0.955], "max_ink_pixels": 0},
            {"id": "gap_b_right", "rect": [0.715, 0.045, 0.745, 0.955], "max_ink_pixels": 0},
            {"id": "gap_c_d", "rect": [0.745, 0.405, 0.965, 0.445], "max_ink_pixels": 0},
        ],
    }
    render_video(
        stem=STEM,
        duration_seconds=9.0,
        draw_frame=lambda fig, t, i, reg: draw_video_frame(fig, t, i, reg, data, scenes),
        audit_config=audit,
        qa_directory=QA_DIR / "_qa",
        representative_times=[1.5, 4.5, 7.5],
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
